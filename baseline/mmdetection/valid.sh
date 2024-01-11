ROOT="../../dataset/"
ANNOTATION="k-fold/val_fold_7.json"
OUTPUT="./work_dirs/atss_swin-l_fold7_clean"
TRAIN=0
VALID=1
CHECKPOINT="latest"
CONFIG_DIR="./configs/Outliers/atss_swin-l-p4-w12_fpn_dyhead_1x_trash.py"
CONF_THRESHOLD=0.3
IOU_THRESHOLD=0.5


python main.py \
--root ${ROOT} \
--annotation ${ANNOTATION} \
--output ${OUTPUT} \
--train ${TRAIN} \
--valid ${VALID} \
--config_dir ${CONFIG_DIR} \
--checkpoint ${CHECKPOINT} \
--conf_threshold ${CONF_THRESHOLD} \
--iou_threshold ${IOU_THRESHOLD}