export CUDA_VISIBLE_DEVICES=7
export NUM_GUP=1

output_dir="checkpoints/sgcls/RETAG/MIXUP(True)BETA(20_5)_NUMRET(10)_THRESH(0.3)_MEMORY_SIZE(8)"
model_dir="${output_dir}/model_0032500.pth"
config_file="${output_dir}/config.yml"

USE_PREDICT_BIAS=False

MODEL_LOGIT_COEF=1.0
RETRIEVAL_LOGIT_COEF=0.0
FREQUENCY_LOGIT_COEF=0.0

python tools/relation_test_net.py \
        --config-file $config_file \
        MODEL.WEIGHT $model_dir \
        TEST.ALLOW_LOAD_FROM_CACHE False \
        RASGG.MODEL_LOGIT_COEF $MODEL_LOGIT_COEF \
        RASGG.RETRIEVAL_LOGIT_COEF $RETRIEVAL_LOGIT_COEF \
        RASGG.FREQUENCY_LOGIT_COEF $FREQUENCY_LOGIT_COEF \
        MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS $USE_PREDICT_BIAS