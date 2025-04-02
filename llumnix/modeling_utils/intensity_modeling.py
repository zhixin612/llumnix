import json
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from calculator import *


def plot_intensity(model_name: str, hw_name: str, mem_usage=1.0):
    m = ModelConfig(model_name)
    hw = HWConfig(hw_name)
    compute, memory_bw, memory_size, bandwidth = hw.compute, hw.memory_bw, hw.memory_size, hw.bandwidth

    # compute the maximum number of tokens that can be stored in memory
    model_weigths_GB = compute_param_size(model_name, m) * 2 / 1024**3
    kv_cache_GB = kv_size_per_token(m, 2) / 1024 ** 3
    max_tokens = (memory_size - model_weigths_GB) // kv_cache_GB
    if max_tokens < 1:
        print(f"Memory size too small to hold model weights and kv cache: {model_name} {hw_name}")
        return

    context_label = ['64', '256', '1k', '8k', '64k', '128k']
    context_length = np.array([64, 256, 1024, 8192, 65536, 131072])
    prefill_FLOPS = np.array([compute_load_prefill(model_name, m, S) for S in context_length])
    decode_FLOPS = np.array([compute_load_decode(model_name, m, S) for S in context_length])

    # prefill compute intensity (FLOPs/byte)
    plt.subplots(1, 2, figsize=(6, 3), dpi=300)
    plt.subplot(1, 2, 1)
    plt.axhline(y=compute * 1024 / memory_bw, color='r', linestyle='--', label=f'{hw_name}')
    print(compute * 1024 / memory_bw)
    for i, S in enumerate(context_length):
        if S > max_tokens:
            continue
        batch_sizes = np.linspace(1, max_tokens // S, min(100, int(max_tokens//S))).astype(int)
        intensities = [batch_size * prefill_FLOPS[i] / (model_weigths_GB * 1024**3) for batch_size in batch_sizes]
        plt.plot(batch_sizes, intensities, label=f'{context_label[i]} tokens')
    plt.xlabel('batch size')
    plt.ylabel('FLOPs/byte (prefill)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='lower right')

    # prefill compute intensity (FLOPs/byte)
    plt.subplot(1, 2, 2)
    plt.axhline(y=compute * 1024 / memory_bw, color='r', linestyle='--', label=f'{hw_name}')
    for i, S in enumerate(context_length):
        batch_sizes = np.linspace(1, max_tokens // S, min(100, int(max_tokens // S))).astype(int)
        print(batch_sizes)
        intensities = [batch_size * decode_FLOPS[i] / ((kv_cache_GB * S * batch_size + model_weigths_GB) * 1024 ** 3) for batch_size in batch_sizes]
        plt.plot(batch_sizes, intensities, label=f'{context_label[i]} tokens')
    plt.xlabel('batch size')
    plt.ylabel('FLOPs/byte (decode)')
    plt.xscale('log')
    plt.yscale('log')

    plt.title(f'{model_name} | {hw_name} ({memory_size:.0f}GB*{mem_usage})' + ' '*60)
    plt.tight_layout()
    plt.savefig(f'figs/{model_name}_{hw_name}GB_{memory_size:.0f}*{mem_usage}.png')
    plt.show()


def prefill_resource_usage_modeling(model_name: str):
    m = ModelConfig(model_name)
    weight = compute_param_size(model_name, m) * 2 / 1024 ** 3  # GB

    seq_lens = [2 ** i for i in range(8, 18)]
    xticks = [str(_) if _ < 1024 else str(_//1024)+'K' for _ in seq_lens]

    mem_usages, mem_read_per_step, compute_per_step = [], [], []
    for seq_len in seq_lens:
        kv_size = kv_size_per_token(m, 2) * seq_len / 1024 ** 3
        activation = peak_mem_overhead_prefill(model_name, m, seq_len, 2) / 1024 ** 3
        mem_usages.append(weight + kv_size + activation)
        mem_read_per_step.append(kv_size + weight + activation)
        compute_per_step.append(compute_load_prefill(model_name, m, seq_len) / 1024 ** 3)

    # plot: left axis -> mem_usage & mem_rw_per_step (GB), right axis -> compute_per_step (GFLOPs)
    plt.figure(figsize=(4, 2.5), dpi=300)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.plot(seq_lens, mem_usages, label='mem_usage')
    plt.plot(seq_lens, mem_read_per_step, label='mem_read_per_step')
    plt.xlabel(f'seq_len (prefill, {model_name})')
    plt.ylabel('size (GB)')
    plt.xscale('log', base=2)
    plt.xticks(seq_lens[::2], xticks[::2])
    plt.legend(loc='upper left')
    plt.twinx()
    plt.plot(seq_lens, compute_per_step, label='compute_per_step', color='r')
    plt.ylabel('GFLOPs')
    # plt.yscale('log')
    plt.legend(loc='center left')

    plt.tight_layout(pad=0.5)
    plt.show()


if __name__ == "__main__":
    models = ['Qwen2.5-7B', 'Qwen2.5-14B', 'Llama-3.2-3B']
    hardware = ['A800', '4090', 'A10']

    # intensity modeling
    # for model in models:
    #     for hw in hardware:
    #         plot_intensity(model, hw, mem_usage=1.0)

    # resource usage modeling
    prefill_resource_usage_modeling('Qwen2.5-7B')
    prefill_resource_usage_modeling('Llama-2-7B')
