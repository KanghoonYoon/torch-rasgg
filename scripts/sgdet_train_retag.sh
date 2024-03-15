export NUM_GUP=2

# Predictor="RA-PENetCorrect"
# Predictor="RA-PENetCorrectProto"
# Predictor="RA-PENetCorrectRelMix"
Predictor="RA-PENetCorrectProtoBeta"

# Predictor="RA-PENetCorrectProtoBetaBG"
# Predictor="RA-PENetCorrectProtoBetaFG"

# Predictor="RA-PENetCorrectProtoBetaEnhanceBG"


COEF1=0.0
COEF2=0.0
COEF3=1.0
CONTRA_LOSS_COEF=0.0

MEMORY_SIZE=8
NUM_CORRECT_BG=1
NUM_RETRIEVALS=10
THRESHOLD=0.3
# IMS_PER_BATCH=4
IMS_PER_BATCH=12

REL_LOSS_TYPE='ce'

MIXUP=True
MIXUP_RATIO=0.5


REWEIGHT_BETA=0.99999
# PREDICT_USE_BIAS=True
PREDICT_USE_BIAS=False

MAX_ITER=20000
VAL_PERIOD=1000

# BASE_LR=1e-3
BASE_LR=1e-4
# BASE_LR=5e-5


# MODEL_NAME='240219_CorrectLabel_FGBG_BASELR_SAMEMEMORY'
# MODEL_NAME='240219_CorrectLabelProto_FGBG_BASELR_SAMEMEMORY'
# MODEL_NAME='240222_CorrectLabelProtoBeta_FGBG_SMALLLR'
# MODEL_NAME='240215_CorrectLabelRelMix_FGBG_BASELR'
# MODEL_NAME='240224_CorrectLabelRelMix_MIDLR'
# MODEL_NAME='240224_CorrectLabelRelMix_SMALLLR'

# MODEL_NAME='240225_CorrectLabelAblation(BG)_BASELR'
# MODEL_NAME='240227_CorrectLabelAblation(FG)_BASELR'
MODEL_NAME='240302_CorrectLabelProtoBeta_BASELR'


PRE_TRAINED_PENET="checkpoints/PE-NET_SGDet/model_0050000.pth"
# PRE_TRAINED_PENET="/home/public/Datasets/CV/faster_ckpt/vg_faster_det.pth"

mkdir ./checkpoints/sgdet
mkdir ./checkpoints/sgdet/${MODEL_NAME}/

# OUTPUT_DIR="./checkpoints/sgdet/${MODEL_NAME}/MIXUP(${MIXUP})(${MIXUP_RATIO})_COEF(${CONTRA_LOSS_COEF})_NUMRET(${NUM_RETRIEVALS})_THRESH(${THRESHOLD})"

OUTPUT_DIR="./checkpoints/sgdet/${MODEL_NAME}/MIXUP(${MIXUP})(BETA(20_5))_NUM_CORRECT_BG(${NUM_CORRECT_BG})_NUMRET(${NUM_RETRIEVALS})_THRESH(${THRESHOLD})"

# /home/public/Datasets/CV/ckpt/faster_ckpt/vg_faster_det.pth
# CUDA_VISIBLE_DEVICES=3 python3 tools/relation_train_net.py \
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10025 tools/relation_train_net.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_rasgg.yaml" \
  TYPE "retag" \
  REL_LOSS_TYPE $REL_LOSS_TYPE \
  REWEIGHT_BETA $REWEIGHT_BETA \
  RASGG.MEMORY_SIZE $MEMORY_SIZE \
  RASGG.NUM_CORRECT_BG $NUM_CORRECT_BG \
  RASGG.CONTRA_LOSS_COEF $CONTRA_LOSS_COEF \
  RASGG.NUM_RETRIEVALS $NUM_RETRIEVALS \
  RASGG.THRESHOLD $THRESHOLD \
  RASGG.MIXUP $MIXUP \
  RASGG.MIXUP_RATIO $MIXUP_RATIO \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS $PREDICT_USE_BIAS \
  MODEL.ROI_RELATION_HEAD.PREDICTOR $Predictor \
  DTYPE "float32" \
  SOLVER.IMS_PER_BATCH $IMS_PER_BATCH TEST.IMS_PER_BATCH $NUM_GUP \
  SOLVER.MAX_ITER $MAX_ITER SOLVER.BASE_LR $BASE_LR \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
  SOLVER.STEPS "(28000, 48000)" SOLVER.VAL_PERIOD $VAL_PERIOD \
  SOLVER.CHECKPOINT_PERIOD $VAL_PERIOD \
  MODEL.PRETRAINED_DETECTOR_CKPT $PRE_TRAINED_PENET \
  OUTPUT_DIR $OUTPUT_DIR \
  SOLVER.PRE_VAL False \
  SOLVER.GRAD_NORM_CLIP 5.0;
