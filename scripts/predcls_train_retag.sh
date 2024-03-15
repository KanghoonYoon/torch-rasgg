export NUM_GUP=1

Predictor="ReTAGPENet"

## Retrieval Module
MEMORY_SIZE=8
NUM_RETRIEVALS=20

## Reliable Selection
THRESHOLD=0.3
NUM_CORRECT_BG=1

## Unbiased Augmentation
MIXUP=True
MIXUP_ALPHA=20
MIXUP_BETA=5

REL_LOSS_TYPE='ce'
IMS_PER_BATCH=6
REWEIGHT_BETA=0.99999

# PREDICT_USE_BIAS=True
PREDICT_USE_BIAS=False

MAX_ITER=60000
VAL_PERIOD=2500
BASE_LR=1e-3

MODEL_NAME='RETAG'

PRE_TRAINED_PENET="checkpoints/PE-NET_PredCls/model_final.pth"

mkdir ./checkpoints/predcls
mkdir ./checkpoints/predcls/${MODEL_NAME}/

OUTPUT_DIR="./checkpoints/predcls/${MODEL_NAME}/MIXUP(${MIXUP})BETA(${MIXUP_ALPHA}_${MIXUP_BETA})_NUMRET(${NUM_RETRIEVALS})_THRESH(${THRESHOLD})_MEMORY_SIZE(${MEMORY_SIZE})"


CUDA_VISIBLE_DEVICES=0 python3 tools/relation_train_net.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_rasgg.yaml" \
  TYPE "retag" \
  REL_LOSS_TYPE $REL_LOSS_TYPE \
  REWEIGHT_BETA $REWEIGHT_BETA \
  RASGG.MEMORY_SIZE $MEMORY_SIZE \
  RASGG.NUM_CORRECT_BG $NUM_CORRECT_BG \
  RASGG.NUM_RETRIEVALS $NUM_RETRIEVALS \
  RASGG.THRESHOLD $THRESHOLD \
  RASGG.MIXUP $MIXUP \
  RASGG.MIXUP_ALPHA $MIXUP_ALPHA \
  RASGG.MIXUP_BETA $MIXUP_BETA \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
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
