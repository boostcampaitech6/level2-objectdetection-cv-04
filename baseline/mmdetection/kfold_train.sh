ROOT="../../dataset/"
ANNOTATION="k-fold/"
OUTPUT="./work_dirs/"
TRAIN=1
CONFIG_DIR="./configs/Outliers/faster_rcnn_r50_fpn_1x_trash.py"
WANDB_NAME="test"


for NAME in "train_fold_1" "train_fold_2" "train_fold_3"
do
    FOLD=$ANNOTATION$NAME".json" # "k-fold/train_fold_1.json"
    NEW_OUTPUT=$OUTPUT$NAME  # "./work_dirs/train_fold_1"
    python main.py \
    --root ${ROOT} \
    --annotation ${FOLD} \
    --output ${NEW_OUTPUT} \
    --train ${TRAIN} \
    --config_dir ${CONFIG_DIR} \
    --wandb_name ${NAME}  # "train_fold_1" "train_fold_2" "train_fold_3"
done