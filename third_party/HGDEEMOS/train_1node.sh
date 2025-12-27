export MASTER_ADDR=localhost
export MASTER_PORT=29500

torchrun --nnodes=1 \
    --nproc_per_node=1 \
    pretrainCD.py \
    --train_data_dir /data4/ruixu/LLMDATA/slim \
    --val_data_dir /data4/ruixu/LLMDATA/slim \
    --resume /deemos-research-area-d/meshgen/code/HG_DEEMOS/out/HY1024_tsz128x16k_100B_ScaleUp20k_unlockCondition_Diff_LLaMA_551M/Samba-DEEMOS-12-21-13/iter-010000-ckpt.pth \
    
