export IP_PORTS=127.0.0.1:18000
export MODEL_PATH="/root/share/models/Qwen2.5-7B-Instruct"
export DATASET_PATH="/root/.cache/huggingface/hub/datasets--shibing624--sharegpt_gpt4/snapshots/3fb53354e02a931777556fb1da37e931d73af48a/sharegpt_gpt4.jsonl"
export NUM_PROMPTS=100
export QPS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3  # should same to online.sh

python ../benchmark/benchmark_logdir.py \
    --ip_ports "${IP_PORTS[@]}" \
    --tokenizer $MODEL_PATH \
    --random_prompt_count $NUM_PROMPTS \
    --dataset_type "sharegpt" \
    --dataset_path $DATASET_PATH \
    --qps $QPS \
    --distribution "poisson" \
    --log_latencies \
    --fail_on_response_failure \
    --log_dir "./logs/test_0220"
    # test_1P3D_sharegpt-2k_qps8-poi_0220
    # name_PD_dataset-size_qps-distri_other_date
