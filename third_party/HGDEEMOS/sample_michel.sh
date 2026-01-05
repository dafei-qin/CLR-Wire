# conda activate cad
CUDA_VISIBLE_DEVICES=1 xvfb-run --auto-servernum python third_party/HGDEEMOS/batch_infer_kvcache.py \
    --config_path src/configs/gpt/gpt_0101_sht_4090.yaml \
    --ckpt /data/ssd/CAD/checkpoints/GPT_INIT_142M/train0/iter-060000-ckpt.pth \
    --start_idx 0 \
    --num_samples 300 \
    --max_new_tokens 1000 \
    --max_seq_len 1000 \
    --temperature 0.0 \
    --device cuda \
    --dtype bf16 \
    --output_dir /data/ssd/CAD/checkpoints/GPT_INIT_142M/train1/iter-060000-ckpt.pth-train8/