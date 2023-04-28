SESSION=$1
CONFIG=$2
WEIGHTS=$3

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python scripts/$SESSION/test.py \
        $CONFIG \
        $WEIGHTS \
        # --save_pred_mask
