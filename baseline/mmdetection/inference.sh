ROOT="../../dataset/"
ANNOTATION="test.json"
OUTPUT="./work_dirs/atss_swin-l_fold7_clean"
TRAIN=0
VALID=0
CHECKPOINT="latest"
CONFIG_DIR="./configs/Outliers/atss_swin-l-p4-w12_fpn_dyhead_1x_trash.py"

python main.py \
--root ${ROOT} \
--annotation ${ANNOTATION} \
--output ${OUTPUT} \
--train ${TRAIN} \
--valid ${VALID} \
--config_dir ${CONFIG_DIR} \
--checkpoint ${CHECKPOINT}