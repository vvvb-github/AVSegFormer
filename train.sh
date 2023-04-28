SESSION=$1
CONFIG=$2

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python scripts/$SESSION/train.py $CONFIG
