
from dataclasses import dataclass
import json
import csv
import os

config_root = os.path.dirname(os.path.abspath(__file__))


class ModelConfig:
    def __init__(self, model_name):
        with open(os.path.join(config_root, f"configs/{model_name}.json"), "r") as f:
            config = json.load(f)
        self.H = config.get("hidden_size")
        self.L = config.get("num_hidden_layers")
        self.N = config.get("num_attention_heads") or config.get("attention_heads")  # attn_heads
        self.n = config.get("num_key_value_heads", self.N)  # kv_heads
        self.I = config.get("intermediate_size")
        self.d = self.H / self.N
        self.v = config.get("vocab_size")
        self.gate = False if model_name.startswith('Qwen-') else True


class HWConfig:
    def __init__(self, hw_name):
        with open(os.path.join(config_root, f"config_hardware/{hw_name}.json"), "r") as f:
            config = json.load(f)
        self.compute = config.get("compute_FP16_TFLOPS")
        self.memory_bw = config.get("memory_bandwidth_GBps")
        self.memory_size = config.get("memory_size_GB")
        self.bandwidth = config.get("gpu_bandwidth_GBps")


# def load_model_config(model_name: str) -> dict:
#     with open(f"configs/{model_name}.json", "r") as f:
#         config = json.load(f)
#     H = config.get("hidden_size")
#     L = config.get("num_hidden_layers")
#     N = config.get("num_attention_heads") or config.get("attention_heads")  # attn_heads
#     n = config.get("num_key_value_heads", N)  # kv_heads
#     I = config.get("intermediate_size")
#     d = H / N
#     v = config.get("vocab_size")
#     gate = False if model_name.startswith('Qwen-') else True
#     return H, L, N, n, I, d, v, gate
#
#
# def load_hw_config(hw_name: str) -> dict:
#     with open(f"config_hardware/{hw_name}.json", "r") as f:
#         config = json.load(f)
#     compute = config.get("compute_FP16_TFLOPS")
#     memory_bw = config.get("memory_bandwidth_GBps")
#     memory_size = config.get("memory_size_GB")
#     bandwidth = config.get("gpu_bandwidth_GBps")
#     return compute, memory_bw, memory_size, bandwidth


def compute_param_size(model_name, m: ModelConfig):
    # F = (2HH + 2Hnd+ KHI) * L + vH + H
    # K = 2 for Qwen, 3 for other

    H, L, N, n, I, d, v, gate = m.H, m.L, m.L, m.n, m.I, m.d, m.v, m.gate
    if model_name.startswith(('Qwen2.5', 'Llama-3', 'Llama-2', 'QwQ')):
        params = (H**2 + 2*(H*n*d) + H**2 + 3*(H*I)) * L  # W_q, W_kv, W_o, W_mlp
    elif model_name.startswith('Qwen-'):
        params = (H**2 + 2*(H*n*d) + H**2 + 2*(H*I)) * L  # W_q, W_kv, W_o, W_mlp
    else:
        raise ValueError(f"Unsupported model {model_name}")
    int_out = v * H + H
    return params + int_out


def compute_load_prefill(model_name, m: ModelConfig, S):
    # F = (2SHH + 4HndS + 4HSS + 2SHH + KHIS) * L + 4HvS = S*( (4HH + 4Hnd + 4HS + KHI) * L + 4Hv)
    # K = 4 for Qwen, 6 for other

    H, L, N, n, I, d, v, gate = m.H, m.L, m.L, m.n, m.I, m.d, m.v, m.gate
    embed_in_out = 2 * H * v * S * 2
    attn_proj = 2 * (H ** 2) * S + 2 * (H * n * d) * S * 2 + 2 * (H ** 2) * S  # Q, KV, O
    attn_attn = 4 * H * (S ** 2)  # Q * K^T * V
    if model_name.startswith(('Qwen2.5', 'Llama-3', 'Llama-2', 'QwQ')):
        mlp_cost = 6 * H * I * S
    elif model_name.startswith('Qwen-'):
        mlp_cost = 4 * H * I * S
    else:
        raise ValueError(f"Unsupported model {model_name}")
    return L * (attn_proj + attn_attn + mlp_cost) + embed_in_out


def compute_load_decode(model_name, m: ModelConfig, context_len):
    # F = (4HH + 4Hnd + 4HS + KHI) * L + 4Hv
    # K = 4 for Qwen, 6 for other

    H, L, N, n, I, d, v, gate = m.H, m.L, m.L, m.n, m.I, m.d, m.v, m.gate
    embed_in_out = 2 * H * v * 2
    attn_proj = 2 * (H ** 2) + 2 * (H * n * d) * 2 + 2 * (H ** 2)  # Q, KV, O
    attn_attn = 4 * H * context_len  # Q * K^T * V
    if model_name.startswith(('Qwen2.5', 'Llama-3', 'Llama-2', 'QwQ')):
        mlp_cost = 6 * H * I
    elif model_name.startswith('Qwen-'):
        mlp_cost = 4 * H * I
    else:
        raise ValueError(f"Unsupported model {model_name}")
    return L * (attn_proj + attn_attn + mlp_cost) + embed_in_out


def peak_mem_overhead_prefill(model_name, m: ModelConfig, S, bytes_per_param):
    # F = max{Q, hidden_state+up_proj+residual(+gate)} * bytes_per_param
    #   = max{ SH, SH+SI+SH(+SI) } * bytes_per_param
    # note: attn_score does not need to be stored in memory (flash-attn)
    H, L, N, n, I, d, v, gate = m.H, m.L, m.L, m.n, m.I, m.d, m.v, m.gate
    if model_name.startswith(('Qwen2.5', 'Llama-3', 'Llama-2', 'QwQ')):
        return max(S*H, S*(2*H+2*I)) * bytes_per_param
    elif model_name.startswith('Qwen-'):
        return max(S*H, S*(2*H+I)) * bytes_per_param
    else:
        raise ValueError(f"Unsupported model {model_name}")


def peak_mem_overhead_decode(model_name, m: ModelConfig, context_len, bytes_per_param):
    # F = max{Q+attn_score, hidden_state+up_proj+residual(+gate)} * bytes_per_param
    #   = max{ H+S, H+I+H(+I) } * bytes_per_param
    H, L, N, n, I, d, v, gate = m.H, m.L, m.L, m.n, m.I, m.d, m.v, m.gate
    if model_name.startswith(('Qwen2.5', 'Llama-3', 'Llama-2', 'QwQ')):
        return max(H+context_len, 2*H+2*I) * bytes_per_param
    elif model_name.startswith('Qwen-'):
        return max(H+context_len, 2*H+I) * bytes_per_param
    else:
        raise ValueError(f"Unsupported model {model_name}")


def kv_size_per_token(m: ModelConfig, bytes_per_param):
    # F = 2ndL * bytes * S
    H, L, N, n, I, d, v, gate = m.H, m.L, m.L, m.n, m.I, m.d, m.v, m.gate
    kv_size = 2 * (n * d)
    return L * kv_size * bytes_per_param


def analysis_single_model(model_name: str, bytes_per_param: int = 2) -> dict:
    m = ModelConfig(model_name)

    param_size = compute_param_size(model_name, m)
    weight_size = param_size * bytes_per_param
    compute_load_p1 = compute_load_prefill(model_name, m, S=1)
    compute_load_p1k = compute_load_prefill(model_name, m, S=1024)
    compute_load_p128k = compute_load_prefill(model_name, m, S=1024 * 128)
    compute_load_d1 = compute_load_decode(model_name, m, context_len=1)
    compute_load_d1k = compute_load_decode(model_name, m, context_len=1024)
    compute_load_d128k = compute_load_decode(model_name, m, context_len=1024 * 128)
    kv_cache_size = kv_size_per_token(m, bytes_per_param)
    overhead_p1 = peak_mem_overhead_prefill(model_name, m, 1, bytes_per_param)
    overhead_p1k = peak_mem_overhead_prefill(model_name, m, 1024, bytes_per_param)
    overhead_p128k = peak_mem_overhead_prefill(model_name, m, 1024 * 128, bytes_per_param)
    overhead_d1 = peak_mem_overhead_decode(model_name, m, 1, bytes_per_param)
    overhead_d1k = peak_mem_overhead_decode(model_name, m, 1024, bytes_per_param)
    overhead_d128k = peak_mem_overhead_decode(model_name, m, 1024 * 128, bytes_per_param)

    result = {
        "model_name": model_name,
        "vocab_size": m.v,
        "hidden_size": m.H,
        "attn_heads": m.N,
        "kv_heads": m.n,
        "inter_size": m.I,
        "num_layers": m.L,
        "params(B)": f'{param_size / 1e9:.1f}' if not model_name.startswith('Qwen-') else f'{param_size / (1024**3):.1f}',
        "weight(GB)": f'{weight_size / (1024 ** 3):.1f}',
        "kv/token (KB)": f'{kv_cache_size / (1024 ** 1):.3f}',
        "kv/1k (MB)": f'{kv_cache_size / (1024 ** 1):.3f}',
        "kv/128k (GB)": f'{kv_cache_size * 128 / (1024 ** 2):.3f}',
        "P_FLOPs/token(GFLOPs)": f'{compute_load_p1 / (1024 ** 3):.3f}',
        "P_FLOPs/1k(TFLOPs)": f'{compute_load_p1k / (1024 ** 4):.3f}',
        "P_FLOPs/128k(PFLOPs)": f'{compute_load_p128k / (1024 ** 5):.3f}',
        "D_FLOPs/S1(GFLOPs)": f'{compute_load_d1 / (1024 ** 3):.3f}',
        "D_FLOPs/S1k(GFLOPs)": f'{compute_load_d1k / (1024 ** 3):.3f}',
        "D_FLOPs/S128K(GFLOPs)": f'{compute_load_d128k / (1024 ** 3):.3f}',
        "overhead/P1(KB)": f'{overhead_p1 / (1024):.1f}',
        "overhead/P1k(MB)": f'{overhead_p1k / (1024 ** 2):.1f}',
        "overhead/P128k(GB)": f'{overhead_p128k / (1024 ** 3):.1f}',
        "overhead/D1(KB)": f'{overhead_d1 / 1024:.1f}',
        "overhead/D1k(KB)": f'{overhead_d1k / (1024):.1f}',
        "overhead/D128k(KB)": f'{overhead_d128k / (1024):.1f}',
    }

    print(result)
    return result


def analysis_models_csv(models: list, bytes_per_param: int = 2, csv_name='model_stats.csv') -> list:
    results = []
    for model in models:
        print(f"Processing {model}")
        res = analysis_single_model(model)
        results.append(res)

    csv_file = "model_stats.csv"
    header = list(results[0].keys())
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"save results in {csv_file}")


if __name__ == "__main__":
    models = ['Llama-3.1-8B', 'Llama-3.1-70B', 'Llama-3.1-405B', 'Llama-3.2-1B', 'Llama-3.2-8B',
              'Qwen2.5-0.5B', 'Qwen2.5-3B', 'Qwen2.5-7B', 'Qwen2.5-14B', 'Qwen2.5-32B', 'Qwen2.5-72B',
              'Qwen-7B', 'Qwen-14B', 'Qwen-72B', 'QwQ-32B']

    analysis_models_csv(models)
