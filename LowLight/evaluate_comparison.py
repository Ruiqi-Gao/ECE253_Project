"""
Comprehensive evaluation script for comparing LIME and Zero-DCE++ results
Combines PSNR, SSIM, LPIPS, BRISQUE, and SPAQ metrics
"""

import os
import re
import numpy as np
from glob import glob
import cv2
from PIL import Image
import argparse
from tqdm import tqdm
from datetime import datetime
import sys

# Lazy import PyTorch to handle import errors gracefully
TORCH_AVAILABLE = False
torch = None
nn = None
torchvision = None
transforms = None

try:
    import torch
    import torch.nn as nn
    import torchvision
    from torchvision import transforms
    TORCH_AVAILABLE = True
except (ImportError, OSError, RuntimeError) as e:
    # Silently handle PyTorch import errors - will be reported later if needed
    TORCH_AVAILABLE = False

# Import metrics
try:
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
    from skimage.metrics import structural_similarity as compare_ssim
except ImportError:
    from skimage.measure import compare_psnr, compare_ssim

# Try to import lpips, but handle PyTorch DLL errors gracefully
LPIPS_AVAILABLE = False
lpips = None
try:
    import lpips
    LPIPS_AVAILABLE = True
except (ImportError, OSError, RuntimeError, AttributeError) as e:
    # lpips import failed - likely due to PyTorch issues
    lpips = None
    LPIPS_AVAILABLE = False

# Try to import BRISQUE - support both 'brisque' package and 'imquality.brisque'
BRISQUE_AVAILABLE = False
brisque = None
BRISQUE_OBJ = None

try:
    # First try the 'brisque' package (pip install brisque)
    from brisque import BRISQUE
    BRISQUE_OBJ = BRISQUE(url=False)
    BRISQUE_AVAILABLE = True
    print("BRISQUE: Using 'brisque' package")
except (ImportError, AttributeError, TypeError, RuntimeError, Exception) as e:
    # Catch all exceptions, not just ImportError, as object creation may fail
    try:
        # Fallback to 'imquality.brisque' package
        from skimage import img_as_float
        import imquality.brisque as brisque
        BRISQUE_AVAILABLE = True
        print("BRISQUE: Using 'imquality.brisque' package")
    except (ImportError, AttributeError, TypeError, RuntimeError, Exception) as e2:
        print("Warning: BRISQUE not installed. BRISQUE metric will be skipped.")
        print(f"  Error with 'brisque' package: {type(e).__name__}: {e}")
        print(f"  Error with 'imquality.brisque' package: {type(e2).__name__}: {e2}")
        print("  To install: pip install brisque  (or pip install imquality)")
        brisque = None
        BRISQUE_AVAILABLE = False

# Check scikit-image version for BRISQUE compatibility
try:
    import skimage
    SKIMAGE_VERSION = skimage.__version__
    # Simple version comparison (scikit-image >= 0.19 uses channel_axis instead of multichannel)
    version_parts = SKIMAGE_VERSION.split('.')[:2]
    major, minor = int(version_parts[0]), int(version_parts[1])
    BRISQUE_COMPATIBLE = major == 0 and minor < 19
except:
    BRISQUE_COMPATIBLE = False

if TORCH_AVAILABLE:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = None


class Tee:
    """Class to write output to both terminal and file"""
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.file = open(file_path, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.file.flush()
    
    def close(self):
        if self.file:
            self.file.close()


def get_unique_output_filename(base_name="evaluation_results"):
    """Generate a unique filename by appending timestamp if file exists"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}.txt"
    counter = 1
    while os.path.exists(filename):
        filename = f"{base_name}_{timestamp}_{counter}.txt"
        counter += 1
    return filename


def extract_frame_number(filename):
    """Extract base frame number from filename (e.g., 'frame_0105' from 'frame_0105_DUAL_g0.6_l0.15.jpg')"""
    match = re.search(r'frame_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None


def find_matching_files(original_dir, enhanced_dir1, enhanced_dir2):
    """Find matching files across three directories based on frame numbers or filename similarity"""
    original_files = {}
    enhanced1_files = {}
    enhanced2_files = {}
    
    # Load original files
    for f in glob(os.path.join(original_dir, '*.jpg')):
        frame_num = extract_frame_number(os.path.basename(f))
        if frame_num is not None:
            if frame_num not in original_files:
                original_files[frame_num] = []
            original_files[frame_num].append(f)
    
    # Load LIME enhanced files
    for f in glob(os.path.join(enhanced_dir1, '*.jpg')):
        frame_num = extract_frame_number(os.path.basename(f))
        if frame_num is not None:
            if frame_num not in enhanced1_files:
                enhanced1_files[frame_num] = []
            enhanced1_files[frame_num].append(f)
    
    # Load Zero-DCE++ files
    for f in glob(os.path.join(enhanced_dir2, '*.jpg')):
        frame_num = extract_frame_number(os.path.basename(f))
        if frame_num is not None:
            if frame_num not in enhanced2_files:
                enhanced2_files[frame_num] = []
            enhanced2_files[frame_num].append(f)
    
    # Find common frame numbers
    common_frames = set(original_files.keys()) & set(enhanced1_files.keys()) & set(enhanced2_files.keys())
    
    # If we found matches by frame number, use that method
    if len(common_frames) > 0:
        print(f"Found {len(original_files)} original images, {len(enhanced1_files)} LIME images, {len(enhanced2_files)} Zero-DCE++ images (matched by frame number)")
        matches = []
        for frame_num in sorted(common_frames):
            # Use first file for each frame (can be modified to use all variants)
            matches.append({
                'frame_num': frame_num,
                'original': original_files[frame_num][0],
                'lime': enhanced1_files[frame_num][0],
                'zerodce': enhanced2_files[frame_num][0]
            })
        return matches
    
    # Fallback: match by filename similarity (for files without frame_ prefix)
    print("Matching files by filename similarity...")
    matches = []
    all_original = list(glob(os.path.join(original_dir, '*.jpg')))
    all_lime = list(glob(os.path.join(enhanced_dir1, '*.jpg')))
    all_zerodce = list(glob(os.path.join(enhanced_dir2, '*.jpg')))
    
    for orig_file in all_original:
        orig_basename = os.path.basename(orig_file)
        orig_base = os.path.splitext(orig_basename)[0]
        
        # Try to find matching LIME file
        lime_match = None
        for lime_file in all_lime:
            lime_basename = os.path.basename(lime_file)
            lime_base = os.path.splitext(lime_basename)[0]
            # Check if original base name is contained in LIME filename or vice versa
            if orig_base in lime_base or lime_base.startswith(orig_base.split('(')[0]):
                lime_match = lime_file
                break
        
        # Try to find matching Zero-DCE++ file
        zerodce_match = None
        for zerodce_file in all_zerodce:
            zerodce_basename = os.path.basename(zerodce_file)
            zerodce_base = os.path.splitext(zerodce_basename)[0]
            # Check if original base name matches Zero-DCE++ filename
            if orig_base == zerodce_base or zerodce_base.startswith(orig_base.split('(')[0]):
                zerodce_match = zerodce_file
                break
        
        if lime_match and zerodce_match:
            matches.append({
                'frame_num': extract_frame_number(orig_basename),
                'original': orig_file,
                'lime': lime_match,
                'zerodce': zerodce_match
            })
    
    print(f"Found {len(matches)} matching image pairs (matched by filename similarity)")
    return matches


def transform(img):
    """Convert image to tensor format for LPIPS"""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available")
    def to_4d(img):
        assert len(img.shape) == 3
        assert img.dtype == np.uint8
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    return compare_psnr(img1, img2)


def calculate_ssim(img1, img2):
    """Calculate SSIM between two images (handles different skimage versions)"""
    # Get image dimensions
    h, w = img1.shape[:2]
    
    # Calculate appropriate win_size (default is 7, but must be smaller than image)
    # Use smaller window for small images
    win_size = min(7, min(h, w) - 1)
    if win_size < 3:
        win_size = 3  # Minimum window size
    
    # Ensure win_size is odd
    if win_size % 2 == 0:
        win_size -= 1
    
    try:
        # Newer version (scikit-image >= 0.19)
        return compare_ssim(img1, img2, channel_axis=2, data_range=255, win_size=win_size)
    except TypeError:
        try:
            # Older version with multichannel (scikit-image < 0.19)
            return compare_ssim(img1, img2, multichannel=True, data_range=255, win_size=win_size)
        except TypeError:
            # Very old version - try without win_size
            try:
                return compare_ssim(img1, img2, multichannel=True, win_size=win_size)
            except:
                return compare_ssim(img1, img2, multichannel=True)


def calculate_lpips(img1, img2, loss_fn):
    """Calculate LPIPS between two images"""
    if loss_fn is None:
        return None
    img1_tensor = transform(img1).to(device)
    img2_tensor = transform(img2).to(device)
    return loss_fn(img1_tensor, img2_tensor).item()


def calculate_brisque(img_path):
    """Calculate BRISQUE score for an image"""
    if not BRISQUE_AVAILABLE:
        return None
    
    try:
        img = Image.open(img_path)
        
        # Check if using 'brisque' package (BRISQUE_OBJ is not None)
        if BRISQUE_OBJ is not None:
            # Use 'brisque' package API
            import numpy as np
            img_array = np.asarray(img)
            score = BRISQUE_OBJ.score(img=img_array)
            return score
        else:
            # Use 'imquality.brisque' package API
            if brisque is None:
                return None
            from skimage import img_as_float
            img_float = img_as_float(img)
            score = brisque.score(img_float)
            return score
    except TypeError as e:
        # Handle version compatibility issue (multichannel parameter)
        if 'multichannel' in str(e) or 'channel_axis' in str(e):
            # Version incompatibility - return None silently
            return None
        raise
    except Exception as e:
        # Other errors - return None silently
        return None


# SPAQ related classes (only if PyTorch is available)
if TORCH_AVAILABLE:
    class Image_load(object):
        def __init__(self, size, stride, interpolation=Image.BILINEAR):
            assert isinstance(size, int)
            self.size = size
            self.stride = stride
            self.interpolation = interpolation

        def __call__(self, img):
            image = self.adaptive_resize(img)
            return self.generate_patches(image, input_size=self.stride)

        def adaptive_resize(self, img):
            h, w = img.size
            if h < self.size or w < self.size:
                img = transforms.ToTensor()(img)
                return img
            else:
                img = transforms.ToTensor()(transforms.Resize(self.size, self.interpolation)(img))
                return img

        def to_numpy(self, image):
            p = image.numpy()
            return p.transpose((1, 2, 0))

        def generate_patches(self, image, input_size, type=np.float32):
            img = self.to_numpy(image)
            img_shape = img.shape
            img = img.astype(dtype=type)
            if len(img_shape) == 2:
                H, W, = img_shape
                ch = 1
            else:
                H, W, ch = img_shape
            if ch == 1:
                img = np.asarray([img, ] * 3, dtype=img.dtype)

            stride = int(input_size / 2)
            hIdxMax = H - input_size
            wIdxMax = W - input_size

            hIdx = [i * stride for i in range(int(hIdxMax / stride) + 1)]
            if H - input_size != hIdx[-1]:
                hIdx.append(H - input_size)
            wIdx = [i * stride for i in range(int(wIdxMax / stride) + 1)]
            if W - input_size != wIdx[-1]:
                wIdx.append(W - input_size)
            patches_numpy = [img[hId:hId + input_size, wId:wId + input_size, :]
                        for hId in hIdx
                        for wId in wIdx]
            patches_tensor = [transforms.ToTensor()(p) for p in patches_numpy]
            patches_tensor = torch.stack(patches_tensor, 0).contiguous()
            return patches_tensor.squeeze(0)

    class Baseline(nn.Module):
        def __init__(self):
            super(Baseline, self).__init__()
            self.backbone = torchvision.models.resnet50(pretrained=False)
            fc_feature = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(fc_feature, 1, bias=True)

        def forward(self, x):
            result = self.backbone(x)
            return result
else:
    # Dummy classes when PyTorch is not available
    class Image_load:
        pass
    class Baseline:
        pass


class SPAQEvaluator:
    def __init__(self, checkpoint_dir=None):
        if not TORCH_AVAILABLE:
            print('Warning: PyTorch is not available. SPAQ will be skipped.')
            self.model = None
            return
        
        self.prepare_image = Image_load(size=512, stride=224)
        self.model = Baseline()
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        
        if checkpoint_dir and os.path.exists(checkpoint_dir):
            try:
                # PyTorch 2.6+ requires weights_only=False for checkpoints with numpy objects
                # This is safe as long as the checkpoint file is from a trusted source
                checkpoint = torch.load(checkpoint_dir, map_location=self.device, weights_only=False)
                # Try different checkpoint formats
                if 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                elif 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'net' in checkpoint:
                    self.model.load_state_dict(checkpoint['net'])
                else:
                    # Try loading directly as state_dict
                    self.model.load_state_dict(checkpoint)
                print(f'SPAQ checkpoint loaded successfully from {checkpoint_dir}')
            except Exception as e:
                print(f'Warning: Could not load SPAQ checkpoint: {e}')
                print(f'  Checkpoint file: {checkpoint_dir}')
                print('  SPAQ will be skipped.')
                self.model = None
        else:
            if checkpoint_dir:
                print(f'Warning: SPAQ checkpoint file not found: {checkpoint_dir}')
            else:
                print('Warning: SPAQ checkpoint not provided. SPAQ will be skipped.')
            self.model = None

    def predict_quality(self, img_path):
        if self.model is None:
            return None
        try:
            image = self.prepare_image(Image.open(img_path).convert("RGB"))
            image = image.to(self.device)
            score = self.model(image).mean()
            return score.item()
        except Exception as e:
            print(f"Error calculating SPAQ for {img_path}: {e}")
            return None


def main(args):
    # Setup output file
    output_filename = get_unique_output_filename("evaluation_results")
    tee = Tee(output_filename)
    original_stdout = sys.stdout
    sys.stdout = tee
    
    try:
        print("=" * 80)
        print("Comprehensive Evaluation: LIME vs Zero-DCE++")
        print("=" * 80)
        print(f"Output will be saved to: {output_filename}")
        print("=" * 80)
        
        # Validate directories
        if not os.path.exists(args.original_dir):
            print(f"Error: Original directory does not exist: {args.original_dir}")
            return
        if not os.path.exists(args.lime_dir):
            print(f"Error: LIME directory does not exist: {args.lime_dir}")
            return
        if not os.path.exists(args.zerodce_dir):
            print(f"Error: Zero-DCE++ directory does not exist: {args.zerodce_dir}")
            return
        
        print(f"\nDirectories:")
        print(f"  Original: {args.original_dir}")
        print(f"  LIME:     {args.lime_dir}")
        print(f"  Zero-DCE++: {args.zerodce_dir}")
        
        # Report status
        print(f"\nMetrics Status:")
        print(f"  PSNR/SSIM: Available")
        if TORCH_AVAILABLE:
            print(f"  PyTorch: Available")
        else:
            print(f"  PyTorch: Not available (LPIPS and SPAQ will be skipped)")
        if LPIPS_AVAILABLE:
            print(f"  LPIPS: Available")
        else:
            print(f"  LPIPS: Not available")
        if BRISQUE_AVAILABLE:
            print(f"  BRISQUE: Available")
        else:
            print(f"  BRISQUE: Not installed")
        
        # Check BRISQUE compatibility (only for imquality.brisque, not for brisque package)
        if BRISQUE_AVAILABLE and BRISQUE_OBJ is None and not BRISQUE_COMPATIBLE:
            print("\nWarning: BRISQUE (imquality) may not work due to scikit-image version incompatibility.")
            print("  To fix: pip install 'scikit-image<0.19' or use 'pip install brisque' instead")
            print("  BRISQUE will be skipped if errors occur.\n")
        
        # Initialize LPIPS
        loss_fn_alex = None
        if TORCH_AVAILABLE and LPIPS_AVAILABLE:
            try:
                loss_fn_alex = lpips.LPIPS(net='alex').to(device)
                print("LPIPS initialized successfully.")
            except Exception as e:
                print(f"Warning: Could not initialize LPIPS: {e}")
                loss_fn_alex = None
        else:
            if not TORCH_AVAILABLE:
                print("LPIPS skipped: PyTorch is not available.")
                print("  To enable: Install PyTorch first (see FIX_PYTORCH.md)")
            elif not LPIPS_AVAILABLE:
                print("LPIPS skipped: lpips package could not be imported.")
                print("  This may be due to PyTorch DLL errors. To install: pip install lpips")
        
        # Initialize SPAQ
        spaq_evaluator = None
        if TORCH_AVAILABLE:
            if args.spaq_checkpoint:
                spaq_evaluator = SPAQEvaluator(args.spaq_checkpoint)
            else:
                print("SPAQ skipped: No checkpoint file provided.")
                print("  To enable: Use --spaq_checkpoint path/to/checkpoint.pt")
        else:
            print("SPAQ skipped: PyTorch is not available.")
            print("  To enable: Install PyTorch first (see FIX_PYTORCH.md)")
        
        # Find matching files
        print("\nFinding matching files...")
        matches = find_matching_files(args.original_dir, args.lime_dir, args.zerodce_dir)
        print(f"Found {len(matches)} matching image pairs.")
        
        if len(matches) == 0:
            print("Error: No matching files found!")
            return
        
        # Test mode: only process first N images
        if args.test > 0:
            matches = matches[:args.test]
            print(f"\nTEST MODE: Processing only first {len(matches)} images")
        
        # Initialize result lists
        results_lime = {
            'psnr': [], 'ssim': [], 'lpips': [], 'brisque': [], 'spaq': []
        }
        results_zerodce = {
            'psnr': [], 'ssim': [], 'lpips': [], 'brisque': [], 'spaq': []
        }
        
        # Process each image pair
        print("\nProcessing images...")
        error_count = 0
        for match in tqdm(matches):
            try:
                # Read images
                img_original = cv2.imread(match['original'])
                img_lime = cv2.imread(match['lime'])
                img_zerodce = cv2.imread(match['zerodce'])
                
                if img_original is None or img_lime is None or img_zerodce is None:
                    print(f"\nWarning: Could not read one of the images for frame {match['frame_num']}")
                    error_count += 1
                    continue
                
                # Ensure same size (resize if necessary)
                h, w = img_original.shape[:2]
                if img_lime.shape[:2] != (h, w):
                    img_lime = cv2.resize(img_lime, (w, h))
                if img_zerodce.shape[:2] != (h, w):
                    img_zerodce = cv2.resize(img_zerodce, (w, h))
                
                # Reference-based metrics (compared to original)
                try:
                    psnr_lime = calculate_psnr(img_original, img_lime)
                    results_lime['psnr'].append(psnr_lime)
                except Exception as e:
                    print(f"\nError calculating PSNR for LIME (frame {match['frame_num']}): {e}")
                
                try:
                    ssim_lime = calculate_ssim(img_original, img_lime)
                    results_lime['ssim'].append(ssim_lime)
                except Exception as e:
                    print(f"\nError calculating SSIM for LIME (frame {match['frame_num']}): {e}")
                
                try:
                    lpips_lime = calculate_lpips(img_original, img_lime, loss_fn_alex)
                    if lpips_lime is not None:
                        results_lime['lpips'].append(lpips_lime)
                except Exception as e:
                    print(f"\nError calculating LPIPS for LIME (frame {match['frame_num']}): {e}")
                
                try:
                    psnr_zerodce = calculate_psnr(img_original, img_zerodce)
                    results_zerodce['psnr'].append(psnr_zerodce)
                except Exception as e:
                    print(f"\nError calculating PSNR for Zero-DCE++ (frame {match['frame_num']}): {e}")
                
                try:
                    ssim_zerodce = calculate_ssim(img_original, img_zerodce)
                    results_zerodce['ssim'].append(ssim_zerodce)
                except Exception as e:
                    print(f"\nError calculating SSIM for Zero-DCE++ (frame {match['frame_num']}): {e}")
                
                try:
                    lpips_zerodce = calculate_lpips(img_original, img_zerodce, loss_fn_alex)
                    if lpips_zerodce is not None:
                        results_zerodce['lpips'].append(lpips_zerodce)
                except Exception as e:
                    print(f"\nError calculating LPIPS for Zero-DCE++ (frame {match['frame_num']}): {e}")
                
                # No-reference metrics
                try:
                    brisque_lime = calculate_brisque(match['lime'])
                    if brisque_lime is not None:
                        results_lime['brisque'].append(brisque_lime)
                except Exception:
                    pass  # BRISQUE errors are already handled in the function
                
                try:
                    brisque_zerodce = calculate_brisque(match['zerodce'])
                    if brisque_zerodce is not None:
                        results_zerodce['brisque'].append(brisque_zerodce)
                except Exception:
                    pass  # BRISQUE errors are already handled in the function
                
                if spaq_evaluator:
                    try:
                        spaq_lime = spaq_evaluator.predict_quality(match['lime'])
                        if spaq_lime is not None:
                            results_lime['spaq'].append(spaq_lime)
                    except Exception as e:
                        pass  # SPAQ errors are already handled
                    
                    try:
                        spaq_zerodce = spaq_evaluator.predict_quality(match['zerodce'])
                        if spaq_zerodce is not None:
                            results_zerodce['spaq'].append(spaq_zerodce)
                    except Exception as e:
                        pass  # SPAQ errors are already handled
            except Exception as e:
                print(f"\nUnexpected error processing frame {match.get('frame_num', 'unknown')}: {e}")
                error_count += 1
                continue
        
        if error_count > 0:
            print(f"\nWarning: {error_count} images had errors during processing.")
        
        # Check if we have any results
        has_results = any(len(results_lime[key]) > 0 for key in results_lime.keys())
        if not has_results:
            print("\nError: No metrics were successfully calculated!")
            return
        
        # Print results
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        
        metrics_info = [
            ('PSNR', 'psnr', 'Higher is better', 'dB'),
            ('SSIM', 'ssim', 'Higher is better', ''),
            ('LPIPS', 'lpips', 'Lower is better', ''),
            ('BRISQUE', 'brisque', 'Lower is better', ''),
            ('SPAQ', 'spaq', 'Higher is better', '')
        ]
        
        print(f"\n{'Metric':<12} {'LIME':<20} {'Zero-DCE++':<20} {'Winner':<15}")
        print("-" * 80)
        
        for metric_name, metric_key, better, unit in metrics_info:
            if len(results_lime[metric_key]) == 0 or len(results_zerodce[metric_key]) == 0:
                print(f"{metric_name:<12} {'N/A':<20} {'N/A':<20} {'N/A':<15}")
                continue
            
            lime_mean = np.mean(results_lime[metric_key])
            zerodce_mean = np.mean(results_zerodce[metric_key])
            
            lime_std = np.std(results_lime[metric_key])
            zerodce_std = np.std(results_zerodce[metric_key])
            
            if better == 'Higher is better':
                winner = "LIME" if lime_mean > zerodce_mean else "Zero-DCE++"
            else:
                winner = "LIME" if lime_mean < zerodce_mean else "Zero-DCE++"
            
            unit_str = f" {unit}" if unit else ""
            lime_str = f"{lime_mean:.4f}±{lime_std:.4f}{unit_str}"
            zerodce_str = f"{zerodce_mean:.4f}±{zerodce_std:.4f}{unit_str}"
            print(f"{metric_name:<12} {lime_str:<20} {zerodce_str:<20} {winner:<15}")
        
        # Detailed statistics
        print("\n" + "=" * 80)
        print("DETAILED STATISTICS")
        print("=" * 80)
        
        for metric_name, metric_key, better, unit in metrics_info:
            if len(results_lime[metric_key]) == 0 or len(results_zerodce[metric_key]) == 0:
                print(f"\n{metric_name} ({better}): Not available")
                continue
            
            print(f"\n{metric_name} ({better})")
            print(f"  LIME:      Mean={np.mean(results_lime[metric_key]):.4f}, Std={np.std(results_lime[metric_key]):.4f}, "
                  f"Min={np.min(results_lime[metric_key]):.4f}, Max={np.max(results_lime[metric_key]):.4f}")
            print(f"  Zero-DCE++: Mean={np.mean(results_zerodce[metric_key]):.4f}, Std={np.std(results_zerodce[metric_key]):.4f}, "
                  f"Min={np.min(results_zerodce[metric_key]):.4f}, Max={np.max(results_zerodce[metric_key]):.4f}")
    
        print("\n" + "=" * 80)
        print(f"Total images evaluated: {len(matches)}")
        print("=" * 80)
        print(f"\nResults saved to: {output_filename}")
        print("=" * 80)
    
    finally:
        # Restore stdout and close file
        sys.stdout = original_stdout
        tee.close()
        print(f"\nResults saved to: {output_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comprehensive evaluation for LIME vs Zero-DCE++')
    parser.add_argument('--original_dir', type=str,
                        default='../selected_700_lowlight/original',
                        help='directory for original images')
    parser.add_argument('--lime_dir', type=str,
                        default='../selected_700_lowlight/LIME_enhanced',
                        help='directory for LIME enhanced images')
    parser.add_argument('--zerodce_dir', type=str,
                        default='../selected_700_lowlight/result_Zero_DCE++',
                        help='directory for Zero-DCE++ enhanced images')
    parser.add_argument('--spaq_checkpoint', type=str,
                        default='BL_release.pt',
                        help='path to SPAQ checkpoint file (default: BL_release.pt)')
    parser.add_argument('--test', type=int, default=0,
                        help='test mode: only process first N images (0 = process all)')
    args = parser.parse_args()
    main(args)

