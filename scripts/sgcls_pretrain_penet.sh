export NUM_GUP=2

Predictor="PrototypeEmbeddingNetwork"
mkdir ./checkpoints/sgcls
mkdir ./checkpoints/sgcls/${MODEL_NAME}

REL_LOSS_TYPE="ce"
# REL_LOSS_TYPE="ce_rwt"

REWEIGHT_BETA=0.99999
# PREDICT_USE_BIAS=True
PREDICT_USE_BIAS=False

OUTPUT_DIR="./checkpoints/PE-NET_SGCls"
# MODEL_NAME="penet_rwt(${REWEIGHT_BETA})"
# OUTPUT_DIR="./checkpoints/sgcls/${MODEL_NAME}/"

# python3 tools/relation_train_net.py \
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29601 tools/relation_train_net.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
  TYPE None \
  REL_LOSS_TYPE $REL_LOSS_TYPE \
  REWEIGHT_BETA $REWEIGHT_BETA \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS $PREDICT_USE_BIAS \
  MODEL.ROI_RELATION_HEAD.PREDICTOR $Predictor \
  DTYPE "float32" \
  SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH $NUM_GUP \
  SOLVER.MAX_ITER 60000 SOLVER.BASE_LR 1e-3 \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
  SOLVER.STEPS "(28000, 48000)" SOLVER.VAL_PERIOD 2500 \
  SOLVER.CHECKPOINT_PERIOD 2500 \
  MODEL.PRETRAINED_DETECTOR_CKPT /home/public/Datasets/CV/faster_ckpt/vg_faster_det.pth \
  OUTPUT_DIR $OUTPUT_DIR \
  SOLVER.PRE_VAL False \
  SOLVER.GRAD_NORM_CLIP 5.0;
