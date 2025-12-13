CONFIG=$1

PYTHONPATH="./:${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=0 \
python -u basicsr/train.py -opt $CONFIG
