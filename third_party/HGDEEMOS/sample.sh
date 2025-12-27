# conda activate cad
xvfb-run python third_party/HGDEEMOS/batch_infer_kvcache.py \
    --config_path src/configs/gpt/gpt_1225.yaml \
    --ckpt /deemos-research-area-d/CADgen/CLR-Wire/out/GPT_INIT_142M/debug2/iter-004760-final-ckpt.pth \
    --num_samples 100 \
    --max_new_tokens 1000 \
    --max_seq_len 1000 \
    --temperature 0.0 \
    --device cuda \
    --dtype fp32 \
    --output_dir /deemos-research-area-d/CADgen/CLR-Wire/out/GPT_INIT_142M/debug2/