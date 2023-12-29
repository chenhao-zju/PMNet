MODEL_MASK="[2,3]"
COMBINE_MODE="add"
MIX_OUT="mixer3"
DATASET='fss'
MIXING_MODE="concat"
SPLIT=0
BACKBONE='resnet50'

SHOT=1

POOL_MIX='concat'

SKIP_MODE='concat'

lr=0.001
WEIGHT=0.6

NAME="original_model_crossmixing_use6dconvhead4_${SKIP_MODE}skip_lr${lr}_k${WEIGHT}_${MIXING_MODE}_${BACKBONE}_${DATASET}_${SPLIT}_${MIX_OUT}_${COMBINE_MODE}_${MODEL_MASK}_01_11"


LOG_MODEL="./model/dcama/log/${NAME}/"

visualize_path="./dcama_visualize/standard_fss_vis/vis_1/"
# visualize_path="./dcama_visualize/${NAME}/vis_${SHOT}/"

mkdir -p -- "$LOG_MODEL"
mkdir -p -- "$visualize_path"

CUDA_VISIBLE_DEVICES=5 .python ./test.py --datapath "./data/" \
            --benchmark ${DATASET} \
            --fold ${SPLIT} \
            --bsz 6 \
            --nworker 20 \
            --backbone ${BACKBONE} \
            --feature_extractor_path "./model/dcama/resnet50_a1h-35c100f8.pth" \
            --logpath ${LOG_MODEL} \
            --lr ${lr} \
            --weight ${WEIGHT} \
            --original True \
            --add_4dconv True \
            --skip_mode ${SKIP_MODE} \
            --pooling_mix ${POOL_MIX} \
            --mixing_mode ${MIXING_MODE} \
            --mix_out ${MIX_OUT} \
            --combine_mode ${COMBINE_MODE} \
            --model_mask ${MODEL_MASK} \
            --load "./model/dcama/our_model/standard/fss/fss_resnet50/train/fold_0_0111_213127/best_model.pt" \
            --nshot ${SHOT} \
            --vispath ${visualize_path} \
            --visualize \
            2>&1 | tee ${LOG_MODEL}/log_test_split${SPLIT}_shot${SHOT}.log


            # --vispath ${visualize_path} \
            # --visualize \

            # --use_aspp True \
            # --upmix True \
            
            # --add_low True \
            
            # --use_aspp True \
            # --upmix True \
            