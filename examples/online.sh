export CONFIG_PATH="/root/llm/disagg/llumnix/configs/bench.yml"
export MODEL_PATH="/root/share/models/Qwen2.5-7B-Instruct"
# Configure on all nodes.
export HEAD_NODE_IP_ADDRESS=127.0.0.1
export HEAD_NODE_IP=$HEAD_NODE_IP_ADDRESS
# Configure on head node.
export HEAD_NODE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

HEAD_NODE=1 python -m llumnix.entrypoints.vllm.serve \
                --config-file $CONFIG_PATH \
                --model $MODEL_PATH \
                --worker-use-ray \
                --max-model-len 4096 \
                --gpu-memory-utilization 0.95