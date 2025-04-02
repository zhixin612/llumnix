# comparison of different parallelism strategies

import os
import numpy as np
import matplotlib.pyplot as plt

from calculator import *


def prefill_latency_ms(model_name, hw_name, mem_usage, tp, pp, S, B: int=1, B_per_param: int=2):
    """
    Computational Load (FLOPs) and Memory Access (GB/s, only read?)
    - H: hidden_size, L: num_layers, N: num_attn_heads, n: num_kv_heads, I: inter_size, d: embed_size, v: vocab_size

    ----- PER LAYER -----   |   Compute Load    |   Memory Access   |   Communication   |
    1. Embedding            |   2BSvH           |   SvH             |   0               |
    2. LayerNorm            |   4BSH(RMS)       |   4BSH            |   0               |
    3. QKV                  |   2BSHH+4BHSnd    |   HH+2Hnd         |   0               |
    4. Attention            |   4BSSH           |   0               |   0               |
    5. (comm) AllGather     |   0               |   0               |   BSH(tp-1)/tp    |
    6. O_proj               |   2BSHH           |   HH              |   0               |
    7. (comm) AllGather     |   0               |   0               |   BSH(tp-1)/tp    |
    8. LayerNorm            |   4BSH(RMS)       |   4BSH            |   0               |
    9. Up and Gate          |   4BSHI / 2BSHI   |   HI / 2HI        |   0               |
    10. Gate*Up             |   BSI             |   0               |   0               |
    11. Down                |   2BSHI           |   HI              |   0               |
    12. (comm) AllReduce    |   0               |   0               |   2BSH(tp-1)/tp^2 |
    13. LayerNorm           |   4BSH(RMS)       |   4BSH            |   0               |
    14. Embedding           |   2BSvH           |   SvH             |   0               |

    note: ring all-reduce (A/tp on each GPU), comm = 2A(tp-1)/tp^2

    TODO(Zhixin): Do QKV, Up, Gate, Down read hidden_states from memory?
    TODO(Zhixin): Should we take memory write load into consideration?
    TODO(Zhixin): Does KV cache need to do all-gather?
    """

    m = ModelConfig(model_name)
    H, L, N, n, I, d, v, gate = m.H, m.L, m.L, m.n, m.I, m.d, m.v, m.gate
    hw = HWConfig(hw_name)
    compute, memory_bw, memory_size, bandwidth = hw.compute, hw.memory_bw, hw.memory_size, hw.bandwidth  # compute: TFLOPs/s  bandwidth: GB/s
    compute = compute * 1e9  # FLOPs/ms
    memory_bw = memory_bw * 1e6  # B/ms
    bandwidth = bandwidth *1e6  # B/ms
    memory_size *= mem_usage

    # operator
    lat_embed = 2 * B * S * v * H / tp / compute
    lat_ln  = 4 * B * S * H / tp / compute
    lat_qkv = max((2 * B * S * H * H + 4 * B * H * S * n * d)/tp/compute, (H * H + 2 * H * n * d)*B_per_param/tp/memory_bw)
    lat_attn = 4 * B * S * S * H / tp / compute
    lat_o_proj = max((2 * B * S * H * H)/tp/compute, (H * H)*B_per_param/tp/memory_bw)
    lat_ln += 4 * B * S * H / tp / compute
    if gate:
        lat_up_gate = max((4 * B * S * H * I)/tp/compute, (2 * H * I)*B_per_param/tp/memory_bw)
    else:
        lat_up_gate = max((2 * B * S * H * I)/tp/compute, (H * I)*B_per_param/tp/memory_bw)
    lat_mul = B * S * I / tp / compute
    lat_down = max((2 * B * S * H * I)/tp/compute, (H * I)*B_per_param/tp/memory_bw)
    lat_ln += 4 * B * S * H / tp / compute
    lat_embed += 2 * B * S * v * H / tp / compute
    # comm
    lat_comm = (2*B*S*H*(tp-1)/tp + 2*B*S*H*(tp-1)/tp/tp) * B_per_param / bandwidth
    lat_pp = (pp-1) * B * S * H * B_per_param / bandwidth  # TODO(Zhixin): check if this is correct

    # total latency
    lat_total_ms = lat_embed + L * (lat_ln + lat_qkv + lat_attn + lat_o_proj + lat_up_gate + lat_mul + lat_down + lat_comm) + lat_pp
    if lat_total_ms > 1000:
        print(f"[prefill] {model_name} \t{hw_name}-TP{tp}-PP{pp} \t(S={S},B={B}) \t{lat_total_ms/1000:.2f} s"
              f"\tT={1000*B*S/lat_total_ms:.0f} tokens/s")
    else:
        print(f"[prefill] {model_name} \t{hw_name}-TP{tp}-PP{pp} \t(S={S},B={B}) \t{lat_total_ms:.2f} ms"
              f"\tT={1000*B*S/lat_total_ms:.0f} tokens/s")
    return lat_total_ms


def decode_latency_ms(model_name, hw_name, mem_usage, tp, pp, S, B: int=1, B_per_param: int=2):
    """
    Computational Load (FLOPs) and Memory Access (GB/s, only read?)
    - H: hidden_size, L: num_layers, N: num_attn_heads, n: num_kv_heads, I: inter_size, d: embed_size, v: vocab_size

    ----- PER LAYER -----   |   Compute Load    |   Memory Access   |   Communication   |
    1. Embedding            |   2BSvH           |   SvH             |   0               |
    2. LayerNorm            |   4BSH(RMS)       |   4BSH            |   0               |
    3. QKV                  |   2BHH+4BHnd      |   HH+2Hnd         |   0               |
    4. Attention            |   4BSH            |   0               |   0               |
    5. (comm) AllGather     |   0               |   0               |   BH(tp-1)/tp     |
    6. O_proj               |   2BHH            |   HH              |   0               |
    7. (comm) AllGather     |   0               |   0               |   BH(tp-1)/tp     |
    8. LayerNorm            |   4BSH(RMS)       |   4BSH            |   0               |
    9. Up and Gate          |   4BHI / 2BHI     |   HI / 2HI        |   0               |
    10. Gate*Up             |   BI              |   0               |   0               |
    11. Down                |   2BHI            |   HI              |   0               |
    12. (comm) AllReduce    |   0               |   0               |   2BH(tp-1)/tp^2  |
    13. LayerNorm           |   4BSH(RMS)       |   4BSH            |   0               |
    14. Embedding           |   2BSvH           |   SvH             |   0               |

    note: ring all-reduce (A/tp on each GPU), comm = 2A(tp-1)/tp^2

    TODO(Zhixin): Do QKV, Up, Gate, Down read hidden_states from memory?
    TODO(Zhixin): Should we take memory write load into consideration?
    TODO(Zhixin): Does KV cache need to do all-gather?
    """
    m = ModelConfig(model_name)
    H, L, N, n, I, d, v, gate = m.H, m.L, m.L, m.n, m.I, m.d, m.v, m.gate
    hw = HWConfig(hw_name)
    compute, memory_bw, memory_size, bandwidth = hw.compute, hw.memory_bw, hw.memory_size, hw.bandwidth  # compute: TFLOPs/s  bandwidth: GB/s
    compute = compute * 1e9  # FLOPs/ms
    memory_bw = memory_bw * 1e6  # B/ms
    bandwidth = bandwidth *1e6  # B/ms
    memory_size *= mem_usage

    # operator
    lat_embed = 2 * B * S * v * H / tp / compute
    lat_ln = 4 * B * S * H / tp / compute
    lat_qkv = max((2 * B * H * H + 4 * B * H * n * d)/tp/compute, (H * H + 2 * H * n * d)*B_per_param/tp/memory_bw)
    lat_attn = 4 * B * S * H / tp / compute
    lat_o_proj = max((2 * B * H * H)/tp/compute, (H * H)*B_per_param/tp/memory_bw)
    lat_ln += 4 * B * S * H / tp / compute
    if gate:
        lat_up_gate = max((4 * B * H * I)/tp/compute, (2 * H * I)*B_per_param/tp/memory_bw)
    else:
        lat_up_gate = max((2 * B * H * I)/tp/compute, (H * I)*B_per_param/tp/memory_bw)
    lat_mul = B * I / tp / compute
    lat_down = max((2 * B * H * I)/tp/compute, (H * I)*B_per_param/tp/memory_bw)
    lat_ln += 4 * B * S * H / tp / compute
    lat_embed += 2 * B * S * v * H / tp / compute

    # comm
    lat_comm = (2*B*H*(tp-1)/tp + 2*B*H*(tp-1)/tp/tp) * B_per_param / bandwidth  # FIXME(Zhixin): check if this is correct
    lat_pp = (pp-1) * B * H * B_per_param / bandwidth  # FIXME(Zhixin): check if this is correct

    # total latency
    lat_total_ms = L * (lat_qkv + lat_attn + lat_o_proj + lat_up_gate + lat_mul + lat_down + lat_comm) + lat_pp
    if lat_total_ms > 1000:
        print(f"[decode] {model_name} \t{hw_name}-TP{tp}-PP{pp} \t(S={S},B={B}) \t{lat_total_ms/1000:.2f} s"
              f"\tT={1000*B/lat_total_ms:.0f} tokens/s")
    else:
        print(f"[decode] {model_name} \t{hw_name}-TP{tp}-PP{pp} \t(S={S},B={B}) \t{lat_total_ms:.2f} ms"
              f"\tT={1000*B/lat_total_ms:.0f} tokens/s")
    return lat_total_ms


def plot_throughput(model, hw_name, mem_usage, para_configs, S, B):
    # plot throughput under different parallelism configurations
    pass


if __name__ == '__main__':
    # FIXME(Zhixin): memory usage and memory size is not considered in the calculation

    models = ['Qwen2.5-14B']
    hw_name = 'A800_practical'
    for model in models:
        prefill_latency_ms(model, hw_name, mem_usage=1, tp=2, pp=1, S=128, B=512)
        prefill_latency_ms(model, hw_name, mem_usage=1, tp=2, pp=1, S=256, B=512)
        prefill_latency_ms(model, hw_name, mem_usage=1, tp=2, pp=1, S=512, B=512)
        prefill_latency_ms(model, hw_name, mem_usage=1, tp=1, pp=1, S=128, B=128)

    for model in models:
        prefill_latency_ms(model, hw_name, mem_usage=1, tp=4, pp=1, S=1024*60, B=1)
        decode_latency_ms(model, hw_name, mem_usage=1, tp=4, pp=1, S=1024, B=1000)
