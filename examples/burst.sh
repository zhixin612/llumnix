export IP_PORTS=127.0.0.1:18000
export MODEL_PATH="/root/share/models/Qwen2.5-7B-Instruct"
export DATASET_PATH="/root/.cache/huggingface/hub/datasets--shibing624--sharegpt_gpt4/snapshots/3fb53354e02a931777556fb1da37e931d73af48a/sharegpt_gpt4.jsonl"
export CUDA_VISIBLE_DEVICES=0,1  # should same to online.sh

echo "[WARNING] Start GPU monitor on devices: $CUDA_VISIBLE_DEVICES"

# request generator:
#   --max_request_len: int = 8192  (16384 | 32768)
#   --random_prompt_count: int
# 1. prompt: random
#   *--gen_random_prompts: store ture
#    --random_prompt_lens_distribution: str = "uniform" | "exponential" | "capped_exponential" | "zipf"
#    --random_prompt_lens_mean: int
#    --random_prompt_lens_range: int (mean ± range//2)
# 2. prompt: dataset
#   *--dataset_type: str = "sharegpt" | "burstgpt" | "arxiv"
#    --dataset_path: str
# 3. response_len: random
#   *--allow_random_gen_len: store true (otherwise use max_request_len-prompt_len)
#    --random_response_lens_distribution: str = "uniform" | "exponential" | "capped_exponential" | "zipf"
#    --random_response_lens_mean: int
#    --random_response_lens_range: int (mean ± range//2)

# logdir
# "./logs/bench-0225/dispatch-rr_1P3D_SGPT1k_Q6_0228"
# name_PD_dataset-size_qps-distri_other_date


###################################################################################################################

python ../benchmark/bench_plus.py \
    --ip_ports "${IP_PORTS[@]}" \
    --tokenizer $MODEL_PATH \
    --random_prompt_count   4 \
    --warmup_request_count  0 \
    --max_request_len       8192 \
    --gen_random_prompts \
    --random_prompt_lens_distribution "uniform" \
    --random_prompt_lens_mean  256 \
    --random_prompt_lens_range 0 \
    --allow_random_gen_len \
    --random_response_lens_distribution "uniform" \
    --random_response_lens_mean  16 \
    --random_response_lens_range 0 \
    --qps                   1 \
    --distribution          "burst" \
    --log_latencies \
    --fail_on_response_failure \
    --log_dir               "./logs/test/overload-test-256-256-Q24"

#    --log_dir               "./logs/test-minimal-overhead/mini_bench-1k-256-Q8-1P1D-NCCL-bs3e5-seq1k"
#    --log_dir               "./logs/bench-migration-0402-sct/mig-sct-bw_exp-128-512_exp-512-4k_uni-Q6-500_2P4D"
#    --log_dir               "./logs/bench-migration-0331-pred/mig-pred-remain_exp-128-512_exp-384-1024_uni-Q6-1K_1P3D"
#    --log_dir               "./logs/0327-llumnix-base-benchmark/batch_size_check_1P1D_B1"

