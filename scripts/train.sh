MODEL_MASK="[2,3]"
COMBINE_MODE="add"
MIX_OUT="mixer3"
DATASET='pascal'
MIXING_MODE="concat"
SPLIT=2
BACKBONE='resnet50'

SHOT=1

POOL_MIX='concat'

SKIP_MODE='concat'
lr=0.001
WEIGHT=0.6


# NAME="original_model_crossmixing_allqsaddlowfeat_use4dconvhead4_${SKIP_MODE}skip_singleloss_lr${lr}_k${WEIGHT}_complexnewqueryskip_upmix_${MIXING_MODE}_${BACKBONE}_${DATASET}_${SPLIT}_${MIX_OUT}_${COMBINE_MODE}_${MODEL_MASK}_02_11"
# NAME="original_model_crossmixing_4dconvhead4_${SKIP_MODE}skip_lr${lr}_k${WEIGHT}_${MIXING_MODE}_${BACKBONE}_${DATASET}_${SPLIT}_${MIX_OUT}_${COMBINE_MODE}_${MODEL_MASK}_testsuim_02_09"
NAME="original_model_crossmixing_layer1addlowfeat_use4dconvhead4_${SKIP_MODE}skip_lr${lr}_k${WEIGHT}_upmix_useaspp_${MIXING_MODE}_${BACKBONE}_${DATASET}_${SPLIT}_${MIX_OUT}_${COMBINE_MODE}_${MODEL_MASK}_02_12"

LOG_MODEL="./model/dcama/log/${NAME}/"

mkdir -p -- "$LOG_MODEL"

CUDA_VISIBLE_DEVICES=6 python -u -m torch.distributed.launch --nnodes=1 --nproc_per_node=1 --node_rank=0 --master_port=16006 \
./train.py --datapath "./data/" \
           --benchmark ${DATASET} \
           --fold ${SPLIT} \
           --bsz 36 \
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
           --nepoch 150 \
           --nshot ${SHOT} \
			2>&1 | tee ${LOG_MODEL}/log_${SPLIT}.log


        #    --cross_mix True \
        #    --add_4dconv True \


        # --add_low True \
        #    --use_aspp True \
        #    --upmix True \