# conda activate cad
CUDA_VISIBLE_DEVICES=1 xvfb-run --auto-servernum python third_party/HGDEEMOS/batch_infer_kvcache.py \
    --config_path src/configs/gpt/gpt_0101_conditioner_4090.yaml \
    --ckpt /data/ssd/CAD/checkpoints/GPT_INIT_142M_CONDITIONER/train0/iter-010620-final-ckpt.pth \
    --start_idx 0 \
    --num_samples 300 \
    --max_new_tokens 1000 \
    --max_seq_len 1000 \
    --temperature 0.0 \
    --device cuda \
    --dtype bf16 \
    --output_dir /data/ssd/CAD/checkpoints/GPT_INIT_142M_CONDITIONER/train0/iter-010620-final-ckpt-train/