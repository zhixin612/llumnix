export IP_PORTS=127.0.0.1:18000
export MODEL_PATH="/root/share/models/Qwen2.5-7B-Instruct"
export DATASET_PATH="/root/.cache/huggingface/hub/datasets--shibing624--sharegpt_gpt4/snapshots/3fb53354e02a931777556fb1da37e931d73af48a/sharegpt_gpt4.jsonl"
export CUDA_VISIBLE_DEVICES=0,1,2,3  # should same to online.sh

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


# dataset benchmark
python ../benchmark/bench_plus.py \
    --ip_ports "${IP_PORTS[@]}" \
    --tokenizer $MODEL_PATH \
    --random_prompt_count   1000 \
    --max_request_len       8192 \
    --dataset_type          "sharegpt" \
    --dataset_path          $DATASET_PATH \
    --qps                   6 \
    --distribution          "poisson" \
    --log_latencies \
    --fail_on_response_failure \
    --log_dir               "./logs/test/test_0303"


# random benchmark: exponential
#python ../benchmark/bench_plus.py \
#    --ip_ports "${IP_PORTS[@]}" \
#    --tokenizer $MODEL_PATH \
#    --random_prompt_count   1000 \
#    --max_request_len       8192 \
#    --gen_random_prompts \
#    --random_prompt_lens_distribution "exponential" \
#    --random_prompt_lens_mean  2048 \
#    --random_prompt_lens_range 4096 \
#    --allow_random_gen_len \
#    --random_response_lens_distribution "exponential" \
#    --random_response_lens_mean  2048 \
#    --random_response_lens_range 4096 \
#    --qps                   2 \
#    --distribution          "gamma" \
#    --coefficient_variation  2.0 \
#    --log_latencies \
#    --fail_on_response_failure \
#    --log_dir               "./logs/test/test_0303"
