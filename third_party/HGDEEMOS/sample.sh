CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --master_port=12345 sample.py \
    --model_path "/deemos-research-area-d/meshgen/code/HG_DEEMOS/out/tsz128x16k_100B_ScaleUp20k_FixCondition_Diff_LLaMA_2121M/Samba-DEEMOS-12-09-09/iter-122500-ckpt.pth" \
    --model_id 2121 \
    --steps 50000 \
    --input_path /deemos-research-area-d/meshgen/code/Samba/data/testdata \
    --output_path mesh_output \
    --repeat_num 4 \
    --uid_list "" \
    --temperature 0.5 \
