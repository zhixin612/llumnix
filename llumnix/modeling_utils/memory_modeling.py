import numpy as np
from typing import List
from calculator import *
import matplotlib.pyplot as plt


def prefill_weight_kv_size_plot(model, bytes_per_param=2):
    """Plot the memory size of weight, KV cache, and activation during prefill stage."""
    m = ModelConfig(model)
    seq_lens = [2 ** i for i in range(1, 20)]
    weight_size = compute_param_size(model, m) * bytes_per_param
    weight_size /= 1024 ** 3  # GB
    kv_sizes, overhead = [], []
    for seq_len in seq_lens:
        kv_sizes.append(kv_size_per_token(m, bytes_per_param) * seq_len / 1024 ** 3)  # GB
        overhead.append(peak_mem_overhead_prefill(model, m, seq_len, bytes_per_param) / 1024 ** 3)  # GB

    # plot
    plt.figure(figsize=(3, 2), dpi=300)
    plt.plot(seq_lens, kv_sizes, label='KV cache')
    plt.plot(seq_lens, overhead, label='activation')
    plt.axhline(y=weight_size, color='r', linestyle='--', label='weight')

    plt.xlabel(f'sen_len (prefill, {model})')
    plt.ylabel('Size (GB)')
    plt.xscale('log')
    plt.yscale('log')

    plt.legend()
    plt.tight_layout(pad=0.5)
    plt.show()


def decode_weight_kv_size_plot(model, bytes_per_param=2):
    """Plot the memory size of weight, KV cache, and activation during decode stage."""
    m = ModelConfig(model)
    seq_lens = [2 ** i for i in range(1, 20)]
    weight_size = compute_param_size(model, m) * bytes_per_param
    weight_size /= 1024 ** 3  # GB
    kv_sizes, overhead = [], []
    for seq_len in seq_lens:
        kv_sizes.append(kv_size_per_token(m, bytes_per_param) * seq_len / 1024 ** 3)  # GB
        overhead.append(peak_mem_overhead_decode(model, m, seq_len, bytes_per_param) / 1024 ** 3)  # GB

    # plot
    plt.figure(figsize=(3, 2), dpi=300)
    plt.plot(seq_lens, kv_sizes, label='KV cache')
    plt.plot(seq_lens, overhead, label='activation')
    plt.axhline(y=weight_size, color='r', linestyle='--', label='weight')

    plt.xlabel(f'sen_len (decode, {model})')
    plt.ylabel('Size (GB)')
    plt.xscale('log')
    plt.yscale('log')

    plt.legend()
    plt.tight_layout(pad=0.5)
    plt.show()


if __name__ == '__main__':
    prefill_weight_kv_size_plot('Qwen2.5-7B')
    decode_weight_kv_size_plot('Qwen2.5-7B')
    prefill_weight_kv_size_plot('Llama-2-7B')
    decode_weight_kv_size_plot('Llama-2-7B')
