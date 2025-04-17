export CONFIG_PATH="/root/llm/disagg/llumnix/configs/bench.yml"
export MODEL_PATH="/root/share/models/Qwen2.5-14B-Instruct"
# Configure on all nodes.
export HEAD_NODE_IP_ADDRESS=127.0.0.1
export HEAD_NODE_IP=$HEAD_NODE_IP_ADDRESS
# Configure on head node.
export HEAD_NODE=1
export CUDA_VISIBLE_DEVICES=0,1
export RAY_DEDUP_LOGS=0

echo "[WARNING] Start serving on devices: $CUDA_VISIBLE_DEVICES"


# [Zhixin] set swap_space to 0 to remove cpu_block (kv cache) (default=4GB)
# [Zhixin] set gpu_memory_utilization to 0.99 to use all gpu memory

# [Zhixin] max-model-len=32768

HEAD_NODE=1 python -m llumnix.entrypoints.vllm.serve \
                --config-file $CONFIG_PATH \
                --model $MODEL_PATH \
                --worker-use-ray \
                --swap-space 0 \
                --max-model-len 32768 \
                --gpu-memory-utilization 0.95 \
                --max-num-seqs 1024 \
                --max-num-batched-tokens 300000
