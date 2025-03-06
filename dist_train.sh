SESSION=$1
CONFIG=$2
GPUS=$3

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=63667 \
    scripts/$SESSION/train.py $CONFIG --launcher pytorch
