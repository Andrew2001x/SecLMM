

export NNODES=1
export GPUS_PER_NODE=1
export WANDB__SERVICE_WAIT=60  
export CUDA_VISIBLE_DEVICES=5

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES"
torchrun $DISTRIBUTED_ARGS src/training/main.py \
 --save-frequency 1 \
 --report-to wandb \
 --train-data /home/gg/gg/MQBench-main/test/model/e1/split_2tar \
 --dataset-type webdataset \
 --imagenet-val ./ImageNet \
 --warmup 2000 \
 --batch-size 512 \
 --epochs 25 \
 --workers 16 \
 --model TinyCLIP-ViT-39M-16-Text-19M \
 --name exp_name \
 --seed 0 \
 --local-loss \
 --grad-checkpointing \
 --output ./outputs/TinyCLIP-ViT-39M-16-Text-19M \
 --lr 0.0001 \
 --gather-with-grad \
 --pretrained-image-file ViT-B-16@openai \
 --pretrained-text-file ViT-B-16@openai \
 --distillation-teacher ViT-B-32@laion2b_e16 \
 --norm_gradient_clip 5 \
 --train-num-samples  15000000 \
 --logit-scale 50
  