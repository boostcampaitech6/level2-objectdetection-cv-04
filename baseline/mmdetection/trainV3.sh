ROOT="../../dataset/"
ANNOTATION="k-fold/train_fold_7.json"
OUTPUT="./work_dirs/dino"
TRAIN=1
CONFIG_DIR="./configs/Outliers/dino-5scale_swin-l_8xb2-12e_coco.py"
WANDB_NAME="dino"

python tools/train.py \
--root ${ROOT} \
--annotation ${ANNOTATION} \
--output ${OUTPUT} \
--train ${TRAIN} \
--config_dir ${CONFIG_DIR} \
--wandb_name ${WANDB_NAME}