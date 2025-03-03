# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# !/usr/bin/env python3
import subprocess
import threading
import multiprocessing
import aiohttp
import argparse
import asyncio
import json
import os
import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import tensorboardX

from scipy.stats import zipf
from enum import Enum
from transformers import AutoTokenizer
from typing import List

multiprocessing.set_start_method('spawn', True)  # avoid tokenizer warning

num_finished_requests = 0
server_num_requests = {}


def get_wait_time(mean_time_between_requests: float, distribution: str, coefficient_variation: float = 0.0) -> float:
    if distribution == "uniform":
        return mean_time_between_requests
    elif distribution == "gamma":
        variance = (coefficient_variation * mean_time_between_requests) ** 2
        shape = mean_time_between_requests ** 2 / variance
        scale = variance / mean_time_between_requests
        return np.random.gamma(shape, scale)
    elif distribution == "poisson":
        return np.random.exponential(mean_time_between_requests)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def request_gen(generator, qps: float, distribution="uniform"):
    while True:
        try:
            item = next(generator)
            yield item
            if distribution != "burst":
                time.sleep(get_wait_time(1.0 / qps, distribution))
        except StopIteration:
            return


async def async_request_gen(generator, qps: float, distribution="uniform", coefficient_variation: float = 0.0):
    while True:
        try:
            item = next(generator)
            yield item
            if distribution != "burst":
                await asyncio.sleep(get_wait_time(1.0 / qps, distribution, coefficient_variation))
        except StopIteration:
            return


class GenerationBackend(str, Enum):
    vLLM = "vLLM"
    NaiveHfPipeline = "NaiveHfPipeline"
    RayGen = "RayGen"
    FasterTransformer = "FasterTransformer"


async def query_model_vllm(prompt, verbose, ip_ports):
    prompt, prompt_len, expected_response_len = prompt

    # Evenly dispatch request to the given api servers.
    global server_num_requests
    server_id = min(server_num_requests, key=server_num_requests.get)
    server_num_requests[server_id] += 1
    timeout = aiohttp.ClientTimeout(total=4 * 60 * 60)
    global num_finished_requests

    async with aiohttp.ClientSession(timeout=timeout) as session:
        best_of = 1
        output_len = expected_response_len
        request_dict = {
            "prompt": prompt,
            "n": 1,
            "best_of": best_of,
            "temperature": 1.0,
            "top_k": 1,
            "max_tokens": max(output_len, 1),
            "ignore_eos": True,
            "stream": False,
        }
        if verbose:
            print('Querying model')
        try:
            async with session.post(f'http://{ip_ports[server_id]}/generate_benchmark', json=request_dict) as resp:
                if verbose:
                    print('Done')

                output = await resp.json()
                # necessary for latency calc
                output['response_len'] = expected_response_len
                if verbose and 'generated_text' in output:
                    print(json.dumps(output['generated_text']))
                num_finished_requests += 1
                print("num_finised_requests: {}".format(num_finished_requests))
                return (prompt, prompt_len, output)
        except aiohttp.ClientError as e:
            print(f"Connect to {ip_ports[server_id]} failed with: {str(e)}")
            sys.exit(1)


def load_prompts(prompt_file):
    with open(prompt_file) as f:
        prompts = [json.loads(l) for l in f.readlines()]
    return prompts


def get_tok_id_lens(tokenizer, batch):
    tokenized = tokenizer.batch_encode_plus(batch)
    lens = [len(s) for s in tokenized['input_ids']]
    return lens


def calculate_throughput(queries,
                         dur_s,
                         backend,
                         tokenizer,
                         median_token_latency,
                         median_e2e_latency,
                         median_inference_latency,
                         all_e2e_latencies,
                         all_per_token_latencies,
                         all_inference_latencies,
                         all_request_ids,
                         all_decode_token_latencies,
                         all_request_lens,
                         log_latencies,
                         fail_on_response_failure):
    prompts = []
    responses = []
    expected_response_lens = []
    cf_gen_lens = []
    for prompt, response in queries:
        if 'generated_text' in response:
            prompts.append(prompt)
            responses.append(response['generated_text'])
        if 'num_output_tokens_cf' in response:
            cf_gen_lens.append(response['num_output_tokens_cf'])
        if 'response_len' in response:
            expected_response_lens.append(response['response_len'])
    # prompt_ids = [p for p in tokenizer.batch_encode_plus(prompts)['input_ids']]
    # response_ids = [r for r in tokenizer.batch_encode_plus(responses)['input_ids']]

    # print(f'check_len actual {list(sorted(len(response) for response in response_ids))}')
    # print(f'check_len expect {list(sorted(expected_response_lens))}')
    # print(f'self-reported {list(sorted(cf_gen_lens))}')
    # for prompt, response, expected_response_len in zip(prompt_ids, response_ids, expected_response_lens):
    #     print(f'check lens {len(prompt)=} {len(response)=} {expected_response_len=}')

    try:
        prompt_lens = get_tok_id_lens(tokenizer, prompts)
        response_lens = get_tok_id_lens(tokenizer, responses)
    except Exception:
        print(prompts)
        print(responses)
        raise

    print(f'prompt_lens {list(sorted(prompt_lens))}')
    print(f'response_lens {list(sorted(response_lens))}')
    print(f'expected_response_lens {list(sorted(expected_response_lens))}')

    prompt_token_count = sum(prompt_lens)
    response_token_count = sum(response_lens)

    all_prompt_lens = prompt_lens
    all_response_lens = response_lens
    all_total_tokens = [all_prompt_lens[i] + all_response_lens[i] for i in range(len(all_prompt_lens))]
    all_waiting_latencies = [all_e2e_latencies[i] - all_inference_latencies[i] for i in range(len(all_e2e_latencies))]

    if backend == GenerationBackend.NaiveHfPipeline:
        # It returns the prompt in the output.
        prompt_token_count = 0
    if backend == GenerationBackend.FasterTransformer:
        response_token_count = sum(expected_response_lens)
    if cf_gen_lens:
        response_token_count = sum(cf_gen_lens)

    # print(f'prompt_token_count {prompt_token_count} response_token_count {response_token_count}')
    throughput_tok_s = (prompt_token_count + response_token_count) / dur_s
    # TODO(Zhixin): need more accurate calculation (dur_s is not accurate)
    throughput_prefill = prompt_token_count / dur_s
    throughput_decode = response_token_count / dur_s

    print(f'throughput_tok_s {throughput_tok_s:.02f}')
    qps = len(responses) / dur_s
    msg1 = f'backend {backend} dur_s {dur_s:.04f} tokens_per_s {throughput_tok_s:.02f} qps {qps:.04f}\n'
    msg2 = f'successful_responses {len(responses)} prompt_token_count {prompt_token_count} response_token_count {response_token_count}\n'
    msg3 = f'{median_token_latency=:.04f}, {median_e2e_latency=:.04f}, {median_inference_latency=:.04f}\n'
    msg = msg1 + msg2 + msg3
    if log_latencies:
        msg += f'{all_request_lens=}\n{all_request_ids=}\n'
        msg += f'{all_total_tokens=}\n{all_prompt_lens=}\n{all_response_lens=}\n'
        msg += f'{all_e2e_latencies=}\n{all_per_token_latencies=}\n{all_inference_latencies=}\n{all_waiting_latencies=}\n{all_decode_token_latencies=}\n'
    print(msg)

    if fail_on_response_failure:
        assert len(responses) == len(queries), f"{fail_on_response_failure=}, expected number of successful respones to equal number of queries, got {len(responses)} vs {len(queries)} "

    return throughput_tok_s, throughput_prefill, throughput_decode


def calculate_cdf(latencies):
    hist, bin_edges = np.histogram(latencies, bins=50)
    cumsum = np.cumsum(hist)
    print("Latency: ")
    print(f"{bin_edges=}")
    print(f"{hist=}")
    print(f"{cumsum=}")


def plot_latency_cdf(req_latencies, prefill_latencies, decode_latencies, log_dir):
    fig_filename = os.path.join(log_dir, "latency.png")
    fig, (ax_req, ax_prefill, ax_decode) = plt.subplots(1, 3, figsize=(3 * 7, 4.8))

    def plot_single(ax, latencies, is_prefill=False):
        hist, bin_edges = np.histogram(latencies, bins=50)
        cumsum = np.cumsum(hist)
        p50 = np.percentile(latencies, 50)
        p80 = np.percentile(latencies, 80)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        p999 = np.percentile(latencies, 99.9)
        ax.plot(bin_edges[1:], cumsum / np.sum(hist) * 100, color='red')
        ax.axvline(p50, color='blue', linestyle='--', label='P50')
        ax.text(p50, ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f"{p50:.2f}", va='bottom',
                ha='right', color='blue')
        ax.axvline(p80, color='green', linestyle='--', label='P80')
        ax.text(p80, ax.get_ylim()[0] + 0.10 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f"{p80:.2f}", va='bottom',
                ha='right', color='green')
        ax.axvline(p95, color='orange', linestyle='--', label='P95')
        ax.text(p95, ax.get_ylim()[0] + 0.15 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f"{p95:.2f}", va='bottom',
                ha='right', color='orange')
        ax.axvline(p99, color='purple', linestyle='--', label='P99')
        ax.text(p99, ax.get_ylim()[0] + 0.20 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f"{p99:.2f}", va='bottom',
                ha='right', color='purple')
        ax.axvline(p999, color='gray', linestyle='--', label='P99.9')
        ax.text(p999, ax.get_ylim()[0] + 0.25 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f"{p999:.2f}", va='bottom',
                ha='right', color='gray')
        mean = np.mean(latencies)
        mean_value = bin_edges[:-1][np.where(bin_edges[:-1] <= mean)][-1]
        mean_percentage = cumsum[np.where(bin_edges[:-1] <= mean)][-1] / np.sum(hist) * 100
        ax.axvline(mean_value, color='black', linestyle='-', label='mean={:.2f}'.format(mean))
        ax.text(mean_value, mean_percentage, f"{mean_percentage:.2f}", va='bottom', ha='right', color='black')
        ax.legend(loc='upper right')
        ax.set_ylabel('Cumulative Percentage(%)')

    plot_single(ax_req, req_latencies)
    plot_single(ax_prefill, prefill_latencies, is_prefill=True)
    plot_single(ax_decode, decode_latencies)
    ax_req.set_xlabel('Latency/req(s)')
    ax_req.set_title('request cdf')
    ax_prefill.set_xlabel('Latency/token(ms)')
    ax_prefill.set_title('prefill cdf')
    ax_decode.set_xlabel('Latency/token(ms)')
    ax_decode.set_title('decode cdf')
    index1 = fig_filename.rfind('/')
    index2 = fig_filename.rfind('/', 0, index1)
    fig_filename_title = fig_filename[index2 + 1:]
    plt.suptitle(fig_filename_title, fontsize=6)
    fig.savefig(fig_filename)

    writer = tensorboardX.SummaryWriter(log_dir, filename_suffix='.client')
    writer.add_figure('plot/latency_cdf', fig)
    writer.close()


def plot_len_cdf(prompt_lens, response_lens, total_tokens, log_dir):
    fig_filename = os.path.join(log_dir, "len.png")
    fig, (ax_prompt, ax_response, ax_total) = plt.subplots(1, 3, figsize=(3 * 7, 4.8))

    def plot_single(ax, lens, x_label_str, title_str):
        hist, bin_edges = np.histogram(lens, bins=50)
        cumsum = np.cumsum(hist)
        p50 = np.percentile(lens, 50)
        p80 = np.percentile(lens, 80)
        p95 = np.percentile(lens, 95)
        p99 = np.percentile(lens, 99)
        ax.plot(bin_edges[1:], cumsum / np.sum(hist) * 100, color='red')
        ax.axvline(p50, color='blue', linestyle='--', label='P50')
        ax.text(p50, ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f"{p50:.2f}", va='bottom',
                ha='right', color='blue')
        ax.axvline(p80, color='green', linestyle='--', label='P80')
        ax.text(p80, ax.get_ylim()[0] + 0.10 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f"{p80:.2f}", va='bottom',
                ha='right', color='green')
        ax.axvline(p95, color='orange', linestyle='--', label='P95')
        ax.text(p95, ax.get_ylim()[0] + 0.15 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f"{p95:.2f}", va='bottom',
                ha='right', color='orange')
        ax.axvline(p99, color='purple', linestyle='--', label='P99')
        ax.text(p99, ax.get_ylim()[0] + 0.20 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f"{p99:.2f}", va='bottom',
                ha='right', color='purple')
        mean = np.mean(lens)
        mean_value = bin_edges[:-1][np.where(bin_edges[:-1] <= mean)][-1]
        mean_percentage = cumsum[np.where(bin_edges[:-1] <= mean)][-1] / np.sum(hist) * 100
        ax.axvline(mean_value, color='black', linestyle='-', label='mean={:.2f}'.format(mean))
        ax.text(mean_value, mean_percentage, f"{mean_percentage:.2f}", va='bottom', ha='right', color='black')
        ax.legend(loc='upper right')
        ax.set_xlabel(x_label_str)
        ax.set_ylabel('Cumulative Percentage(%)')
        ax.set_title(title_str)

    plot_single(ax_prompt, prompt_lens, 'prompt len', 'prompt len cdf')
    plot_single(ax_response, response_lens, 'response len', 'response len cdf')
    plot_single(ax_total, total_tokens, 'total token', 'total token cdf')
    index1 = fig_filename.rfind('/')
    index2 = fig_filename.rfind('/', 0, index1)
    fig_filename_title = fig_filename[index2 + 1:]
    plt.suptitle(fig_filename_title, fontsize=6)
    fig.savefig(fig_filename)

    writer = tensorboardX.SummaryWriter(log_dir, filename_suffix='.client')
    writer.add_figure('plot/req_len_cdf', fig)
    writer.close()


def plot_instance(log_dir):
    current_dir = os.path.dirname(log_dir)
    log_files = glob.glob(os.path.join(current_dir, '*.log_instance.csv'))
    log_files.sort(key=os.path.getmtime, reverse=True)
    df_0 = pd.read_csv(log_files[0]).sort_values(by=["timestamp"])
    timestamp_list_0 = df_0["timestamp"].to_numpy()
    num_instances_list_0 = df_0["num_instances"].to_numpy()
    time_0 = 0
    sum_0 = 0
    for idx, t in enumerate(timestamp_list_0):
        if t > time_0:
            time_0 += 1
            sum_0 += num_instances_list_0[idx]
    print(f"{sum_0 / time_0} gpu/s")
    avg_instance_num = np.round(sum_0 / time_0, 2)

    fig, ax = plt.subplots()
    ax.plot(timestamp_list_0, num_instances_list_0, color="red", label=f"instance_num(avg {avg_instance_num} /s)")
    ax.legend(loc='upper left')
    fig_filename = os.path.join(log_dir, "instance.png")
    index1 = fig_filename.rfind('/')
    index2 = fig_filename.rfind('/', 0, index1)
    fig_filename_title = fig_filename[index2 + 1:]
    plt.suptitle(fig_filename_title, fontsize=6)
    fig.savefig(fig_filename)

    writer = tensorboardX.SummaryWriter(log_dir, filename_suffix='.client')
    writer.add_figure('plot/instance_num', fig)
    writer.close()

    return avg_instance_num


def save_all_decode_token_latencies_npy(all_token_latencies: List[np.ndarray], log_dir):
    dtype = [('timestamp', float), ('latency', float)]
    all_lat_pairs = []
    for arr in all_token_latencies:
        # use decode latencies
        for pair in arr[1:]:
            all_lat_pairs.append((pair[0], pair[1]))
    all_lat_pairs = np.array(all_lat_pairs, dtype=dtype)
    all_lat_pairs = np.sort(all_lat_pairs, order='timestamp')
    np.save(os.path.join(log_dir, 'TBT.npy'), all_lat_pairs)


class MeasureLatency:
    def __init__(self, logdir=None):
        self._request_ids = []
        self._request_lens = []
        self._request_latencies = []
        self._per_token_latencies = []
        self._decode_token_latencies = []
        self._prefill_token_latencies = []
        self._all_token_latencies = []
        self._decode_sum_latencies = []
        self._all_decode_token_latencies = []
        self._inference_latencies = []
        self._per_token_latencies_breakdown_dict = []

        self.writer = tensorboardX.SummaryWriter(logdir, filename_suffix='.client', flush_secs=5)
        self.writer_step = 0
        self.total_decode_latency = 0
        self.total_decode_tokens = 0

    def measure(self, f):
        async def measured(*args, **kwargs):
            start = time.time()
            prompt, prompt_len, output = await f(*args, **kwargs)
            # Do not record latency if request failed.
            if 'generated_text' in output:
                latency = time.time() - start
                self._request_latencies.append(latency)
                self.writer.add_scalar('client/request_latency', latency, self.writer_step)
                try:
                    self._per_token_latencies.append(
                        latency / output['response_len'])
                    # self.writer.add_scalar('client/per_token_latency', latency, self.writer_step)
                except ZeroDivisionError:
                    pass
            if 'request_id' in output:
                self._request_ids.append(output['request_id'])
            if 'per_token_latency' in output:
                lat_arr = np.array(output['per_token_latency'])
                mean_decode_token_latency = 0 if len(lat_arr) == 1 else np.mean(lat_arr[1:, 1])
                decode_sum_latency = 0 if len(lat_arr) == 1 else np.sum(lat_arr[1:, 1])
                self.total_decode_latency += decode_sum_latency
                self.total_decode_tokens += len(lat_arr[1:, 1])
                # decode
                self._request_lens.append(len(lat_arr[1:, 1]))  # output length? zhixin
                self._decode_token_latencies.append(mean_decode_token_latency)
                self._decode_sum_latencies.append(decode_sum_latency)
                self._all_decode_token_latencies.extend(lat_arr[1:, 1])
                # prefill & all
                self._prefill_token_latencies.append(lat_arr[0][1])
                self._all_token_latencies.append(lat_arr)

                if self.total_decode_tokens > 0:
                    self.writer.add_scalar('client/TPOT_avg', self.total_decode_latency / self.total_decode_tokens,
                                           self.writer_step)
                self.writer.add_scalar('client/TPOT_req', mean_decode_token_latency, self.writer_step)
                self.writer.add_scalar('client/TTFT_avg', np.mean(self._prefill_token_latencies), self.writer_step)
                self.writer.add_scalar('client/TTFT_req', lat_arr[0][1], self.writer_step)
                self.writer.add_scalar('client/len_prompt', prompt_len, self.writer_step)
                self.writer.add_scalar('client/len_response', len(lat_arr[1:, 1]), self.writer_step)

                # self.writer.add_scalar('client/decode_token_latency', mean_decode_token_latency, self.writer_step)
                # self.writer.add_scalar('client/decode_sum_latency', decode_sum_latency, self.writer_step)
                # self.writer.add_scalar('client/prefill_token_latency', lat_arr[0][1], self.writer_step)
                # self.writer.add_scalar('client/request_len', len(lat_arr[1:, 1]), self.writer_step)
            if 'per_token_latency_breakdown_dict' in output:
                self._inference_latencies.append(
                    np.mean(output['per_token_latency_breakdown_dict']['step_latency_engine']))
                self._per_token_latencies_breakdown_dict.append(output['per_token_latency_breakdown_dict'])
            self.writer_step += 1
            return prompt, output

        return measured


def get_token_ids(input_str, tokenizer):
    t = tokenizer(input_str)
    return t['input_ids']


async def benchmark(
        backend: GenerationBackend,
        tokenizer,
        prompts: List[str],
        allow_variable_generation_length: bool,
        verbose: bool,
        log_dir: str,
        ip_ports: List[int],
        distribution: str,
        qps: float,
        coefficient_variation: float,
        log_latencies: bool,
        fail_on_response_failure: bool,
):
    if backend == GenerationBackend.vLLM:
        query_model = query_model_vllm
    else:
        raise ValueError(f'unknown backend {backend}')

    global server_num_requests
    num_servers = len(ip_ports)
    for server_id in range(num_servers):
        server_num_requests[server_id] = 0

    m = MeasureLatency(logdir=log_dir)

    query_model = m.measure(query_model)

    if distribution == "burst":
        qps = float('inf')
    if distribution != "gamma":
        print(f'[WARNING] coefficient_variation is only supported for gamma distribution. Setting it to 0.0.')
        coefficient_variation = 0.0

    print(f'Starting with backend={backend}, num_prompts={len(prompts)}, allow_variable_gen_length={allow_variable_generation_length}')
    print(f'traffic distribution={distribution}, qps={qps}, coefficient_variation={coefficient_variation}')

    async_prompts = async_request_gen(
        iter(prompts), qps=qps, distribution=distribution, coefficient_variation=coefficient_variation)

    start_time = time.time()
    tasks = []
    async for prompt in async_prompts:
        tasks.append(asyncio.create_task(query_model(prompt, verbose, ip_ports)))
    queries = await asyncio.gather(*tasks)
    dur_s = time.time() - start_time
    median_token_latency = np.median(m._per_token_latencies)
    median_e2e_latency = np.median(m._request_latencies)
    median_inference_latency = np.median(m._inference_latencies)

    throughput = calculate_throughput(queries,
                                      dur_s,
                                      backend,
                                      tokenizer,
                                      median_token_latency,
                                      median_e2e_latency,
                                      median_inference_latency,
                                      m._request_latencies,
                                      m._per_token_latencies,
                                      m._inference_latencies,
                                      m._request_ids,
                                      m._decode_token_latencies,
                                      m._request_lens,
                                      log_latencies,
                                      fail_on_response_failure)
    calculate_cdf(m._request_latencies)
    plot_latency_cdf(m._request_latencies, m._prefill_token_latencies, m._decode_token_latencies, log_dir)
    save_all_decode_token_latencies_npy(m._all_token_latencies, log_dir)
    # avg_instance_num = plot_instance(log_dir)
    avg_instance_num = 0.0

    return throughput, \
           m._prefill_token_latencies, \
           m._decode_token_latencies, \
           m._inference_latencies, \
           avg_instance_num, \
           m._request_latencies, \
           m._request_ids, \
           m._decode_sum_latencies, \
           m._request_lens, \
           m._all_decode_token_latencies, \
           m._per_token_latencies_breakdown_dict


def gen_random_response_lens(distribution: str, len_mean, len_range, num_prompts):
    if distribution == 'uniform':
        if len_range == 0:
            return [len_mean for _ in range(num_prompts)]

        low = len_mean - (len_range // 2)
        high = len_mean + (len_range // 2)
        response_lens = list(
            map(lambda _: random.randint(low, high), range(num_prompts)))
    elif distribution == 'exponential':
        response_lens = [min(round(s), len_range) for s in np.random.exponential(scale=len_mean, size=num_prompts)]
    elif distribution == 'capped_exponential':
        response_lens = []
        while len(response_lens) < num_prompts:
            sample = round(np.random.exponential(scale=len_mean))
            if sample <= len_range and sample >= 1:
                response_lens.append(sample)
    elif distribution == 'zipf':
        rank = np.arange(1, len_mean * 2)
        if len_mean == 1024 and len_range == 6144:
            alpha = 1.0005
        elif len_mean == 512 and len_range == 6144:
            alpha = 1.15
        elif len_mean == 256 and len_range == 6144:
            alpha = 1.5
        elif len_mean == 128 and len_range == 6144:
            alpha = 2.0
        else:
            alpha = 1.0
        probabilities = zipf.pmf(rank, alpha)
        probabilities /= np.sum(probabilities)
        response_lens = np.random.choice(np.arange(1, len_mean * 2), size=num_prompts, p=probabilities)
    else:
        raise ValueError(f'unknown distribution {distribution=}')

    scaling_factor = len_mean / np.mean(response_lens)
    response_lens = np.ceil(np.array(response_lens) * scaling_factor).astype(int)
    if distribution == 'zipf':
        response_lens = [response_len if response_len <= len_range else len_range for response_len in response_lens]
    elif distribution == 'uniform':
        capped_response_lens = []
        for response_len in response_lens:
            if response_len < low:
                capped_response_lens.append(low)
            elif response_len > high:
                capped_response_lens.append(high)
            else:
                capped_response_lens.append(response_len)
        response_lens = capped_response_lens
    else:
        response_lens = [response_len if response_len <= len_range else len_range for response_len in response_lens]
    response_lens = [int(x) for x in response_lens]

    return response_lens


def gen_random_prompts(tokenizer, len_mean, len_range, num_prompts, vocab_ids_to_exclude=[]):
    prompts, _ = gen_random_prompts_return_lens(
        tokenizer, len_mean, len_range, num_prompts, vocab_ids_to_exclude)
    return prompts


def gen_random_prompts_return_lens(tokenizer, distribution: str, len_mean, len_range, num_prompts,
                                   vocab_ids_to_exclude=[]):
    def gen_prompt_ids(length):
        return [random.randint(10, tokenizer.vocab_size) for _ in range(length)]

    # prompt_lens = list(
    #     map(lambda _: random.randint(low, high), range(num_prompts)))
    prompt_lens = gen_random_response_lens(distribution, len_mean, len_range, num_prompts)
    prompts_as_ids = list(
        map(lambda prompt_len: gen_prompt_ids(prompt_len), prompt_lens))
    prompts = list(
        map(lambda prompt_ids: tokenizer.decode(prompt_ids), prompts_as_ids))

    # Because tokens do not map 1:1 to words, sometimes we get more tokens than desired.
    # This removes the additional tokens by tokenizing the prompt and cutting off additional tokens.
    # Confusingly, it works with a single iteration per prompt.
    for i, (p, l) in enumerate(zip(prompts, prompt_lens)):
        encoded = tokenizer(p)['input_ids']
        if len(encoded) > l:
            # I am not sure why l-1 works, but it does..
            encoded = encoded[:l - 1]
        decoded = tokenizer.decode(encoded)
        encoded = tokenizer(decoded)['input_ids']
        # assert len(
        #     encoded) == l, f"Expected prompt to contain exactly {l} tokens, got {len(encoded)=}"
        prompts[i] = decoded

    return prompts, prompt_lens


def sample_sharegpt_requests(
        dataset_path: str,
        num_requests: int,
        tokenizer,
        max_seqlen: int,
):
    # Load the dataset.
    prompts = []
    prompt_lens = []
    response_lens = []
    with open(dataset_path) as f:
        for line in f:
            data = json.loads(line)
            if len(data["conversations"]) >= 2:
                prompt = data["conversations"][0]["value"]
                res = data["conversations"][1]["value"]
                prompt_token_ids = tokenizer(prompt).input_ids
                completion_token_ids = tokenizer(res).input_ids
                if len(prompt_token_ids) + len(completion_token_ids) < max_seqlen and \
                        len(prompt_token_ids) > 0 and len(completion_token_ids) > 0:
                    prompts.append(prompt)
                    prompt_lens.append(len(prompt_token_ids))
                    response_lens.append(len(completion_token_ids))
            if len(prompts) > num_requests:
                break
    sampled_ids = [random.randint(0, len(prompts) - 1) for _ in range(num_requests)]
    sampled_prompts = [prompts[idx] for idx in sampled_ids]
    sampled_prompt_lens = [prompt_lens[idx] for idx in sampled_ids]
    sampled_response_lens = [response_lens[idx] for idx in sampled_ids]
    # print(f"max len:{max(a+b for a,b in zip(prompt_lens, response_lens))}")
    return sampled_prompts, sampled_prompt_lens, sampled_response_lens


def sample_burstgpt_request(
        dataset_path: str,
        num_requests: int,
        tokenizer,
        max_seqlen: int,
):
    data = pd.read_csv(dataset_path)
    request_tokens = data['Request tokens'].tolist()
    response_tokens = data['Response tokens'].tolist()
    num_prompts_sampled = min(num_requests, len(data))
    sampled_ids = random.sample(range(len(request_tokens)), num_prompts_sampled)
    random.shuffle(sampled_ids)
    # sampled_ids = range(num_prompts_sampled)
    prompt_lens = []
    response_lens = []
    for idx in sampled_ids:
        if request_tokens[idx] + response_tokens[idx] < max_seqlen and \
                request_tokens[idx] > 0 and response_tokens[idx] > 0:
            prompt_lens.append(request_tokens[idx])
            response_lens.append(response_tokens[idx])
    prompts = [tokenizer.decode([20] * prompt_len) for prompt_len in prompt_lens]
    return prompts, prompt_lens, response_lens


def sample_arxiv_request(
        dataset_path: str,
        num_requests: int,
        tokenizer,
        max_seqlen: int,
):
    prompts = []
    prompt_lens = []
    response_lens = []
    with open(dataset_path) as f:
        for id_, row in enumerate(f):
            data = json.loads(row)
            prompt = " ".join(data["article_text"])
            res = " ".join(data["abstract_text"])
            prompt_token_ids = tokenizer(prompt).input_ids
            completion_token_ids = tokenizer(res).input_ids
            if len(prompt_token_ids) + len(completion_token_ids) < max_seqlen and \
                    len(prompt_token_ids) > 0 and len(completion_token_ids) > 0:
                prompts.append(prompt)
                prompt_lens.append(len(prompt_token_ids))
                response_lens.append(len(completion_token_ids))
            if len(prompts) > num_requests:
                break
    return prompts, prompt_lens, response_lens


def gpu_utilization_monitor(log_dir, devices: str, running_event: threading.Event, interval_sec=1):
    # Get the GPU utilization and save to tensorboard
    def get_gpu_util():
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", f"--id={devices}"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, encoding='utf-8')
            return [int(x) for x in result.stdout.strip().split('\n')]
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to get GPU utilization: {e}")
            return None

    writer = tensorboardX.SummaryWriter(log_dir, filename_suffix='.client', flush_secs=3)
    gpu_ids = devices.split(',')
    steps = 0
    while running_event.is_set():
        gpu_util = get_gpu_util()
        writer.add_scalars('scheduler.req/gpu_util', {f'gpu_{gid}': util for gid, util in zip(gpu_ids, gpu_util)}, steps)
        steps += 1
        time.sleep(interval_sec)
    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Name or path of the tokenizer.")
    parser.add_argument('--trust_remote_code',
                        action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--backend', type=GenerationBackend,
                        choices=[e.name for e in GenerationBackend], default='vLLM')
    parser.add_argument('--log_dir', type=str, default='./results/default')
    parser.add_argument('--ip_ports', nargs='+', required=True, help='List of ip:port')
    parser.add_argument('--random_prompt_lens_mean', type=int)
    parser.add_argument('--random_prompt_lens_range', type=int)
    parser.add_argument('--variable_prompt_lens_distribution', choices=[
        "uniform", "exponential", "capped_exponential", "zipf"], default="uniform")
    parser.add_argument('--random_prompt_count', type=int)
    parser.add_argument('--max_request_len', type=int, default=8192)

    parser.add_argument(
        '--distribution', choices=["burst", "uniform", "poisson", "gamma"], default="poisson",
        help="burst: qps=inf, uniform: fixed qps, poisson & gamma: variable qps (only gamma support CV)")
    parser.add_argument('--qps', type=float, default=4.0)
    parser.add_argument('--coefficient_variation', type=float, default=0.0,
                        help="Coefficient of variation for the gamma distribution. "
                             "(0<CV<1.0 for steady workload, =1.0 for poisson distribution, >1.0 for bursty workload)")
    parser.add_argument('--log_latencies', action="store_true",
                        help="Whether or not to write all latencies to the log file.")
    parser.add_argument('--fail_on_response_failure', action="store_true",
                        help="Whether or not to fail the benchmarking script if any request fails")

    parser.add_argument('--variable_response_lens_mean', type=int)
    parser.add_argument('--variable_response_lens_range', type=int)
    parser.add_argument('--variable_response_lens_distribution', choices=[
        "uniform", "exponential", "capped_exponential", "zipf"], default="uniform")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset_type', type=str, choices=['sharegpt', 'burstgpt', 'arxiv'])
    group.add_argument('--gen_random_prompts', action='store_true')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--allow_variable_generation_length',
                       action='store_true')
    group.add_argument('--dataset_path', type=str)

    parser.add_argument('--print_generation_lens_and_exit',
                        action='store_true')

    # parser.add_argument('--enable_migration', type=int, default=0)
    # parser.add_argument('--priority_ratio', type=float, default=0.0)

    args = parser.parse_args()

    # save all configs into tensorboard log
    writer = tensorboardX.SummaryWriter(args.log_dir, filename_suffix='.client')
    writer.add_text('config/client', str(args))
    writer.flush()
    # writer.close()  # do not close writer, keep it open for logging

    if args.gen_random_prompts:
        assert args.random_prompt_count is not None

    backend = GenerationBackend[args.backend]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
    print(tokenizer)

    os.makedirs(args.log_dir, exist_ok=True)

    if args.dataset_type:
        random.seed(0xCADE)
        np.random.seed(0xCADE)
        if args.dataset_type == "sharegpt":
            prompts, prompt_lens, response_lens = sample_sharegpt_requests(args.dataset_path, args.random_prompt_count,
                                                                           tokenizer, args.max_request_len)
        elif args.dataset_type == "burstgpt":
            prompts, prompt_lens, response_lens = sample_burstgpt_request(args.dataset_path, args.random_prompt_count,
                                                                          tokenizer, args.max_request_len)
        elif args.dataset_type == "arxiv":
            prompts, prompt_lens, response_lens = sample_arxiv_request(args.dataset_path, args.random_prompt_count,
                                                                       tokenizer, args.max_request_len)
        num_prompts = len(prompts)
    elif args.gen_random_prompts:
        num_prompts = args.random_prompt_count
        random.seed(0xCADE)
        np.random.seed(0xCADE)
        prompts, prompt_lens = gen_random_prompts_return_lens(
            tokenizer,
            distribution=args.variable_prompt_lens_distribution,
            len_mean=args.random_prompt_lens_mean,
            len_range=args.random_prompt_lens_range,
            num_prompts=num_prompts,
            vocab_ids_to_exclude=tokenizer.all_special_ids,
        )
    else:
        raise ValueError("unknown prompts")

    if args.allow_variable_generation_length:
        response_lens = gen_random_response_lens(
            args.variable_response_lens_distribution, args.variable_response_lens_mean,
            args.variable_response_lens_range, num_prompts=num_prompts)
        args.fixed_max_tokens = -1

    for i, (prompt_len, gen_len) in enumerate(zip(prompt_lens, response_lens)):
        total = prompt_len + gen_len
        if total > args.max_request_len:
            print(f'truncating long prompt+gen_len {prompt_len=} {gen_len=}')
            gen_len = args.max_request_len - prompt_len
        response_lens[i] = gen_len

    if args.print_generation_lens_and_exit:
        print(f'{prompt_lens=}')
        print(f'{response_lens=}')
        print('Exiting...')
        return

    if args.verbose or True:
        print('prompt lens', sorted(list(prompt_lens)))
        print('response lens', sorted(list(response_lens)))
        total_tokens = []
        for i, (prompt_len, gen_len) in enumerate(zip(prompt_lens, response_lens)):
            total_tokens.append(prompt_len + gen_len)
        print('total tokens', sorted(list(total_tokens)))

    plot_len_cdf(prompt_lens, response_lens, total_tokens, args.log_dir)

    prompts = list(zip(prompts, prompt_lens, response_lens))

    # gpu utilization monitor, use subprocess to avoid tokenizer deadlocks
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    assert devices is not None, "CUDA_VISIBLE_DEVICES must be set"
    running_event = multiprocessing.Event()
    running_event.set()
    monitor = multiprocessing.Process(target=gpu_utilization_monitor, args=(args.log_dir, devices, running_event))
    monitor.start()

    throughput, \
    prefill_token_latencies, \
    decode_token_latencies, \
    inference_latencies, \
    avg_instance_num, \
    request_latencies, \
    request_ids, \
    decode_sum_latencies, \
    request_lens, \
    all_decode_token_latencies, \
    per_token_latencies_breakdown_dict = asyncio.run(benchmark(
        backend,
        tokenizer,
        prompts,
        args.allow_variable_generation_length,
        args.verbose,
        args.log_dir,
        args.ip_ports,
        args.distribution,
        args.qps,
        args.coefficient_variation,
        args.log_latencies,
        args.fail_on_response_failure,
    ))

    running_event.clear()
    monitor.join()

    results = []
    file_name = os.path.join(args.log_dir, "latency_info.json")
    try:
        with open(file_name, 'r') as f:
            results = json.load(f)
    except json.decoder.JSONDecodeError:
        pass
    except FileNotFoundError:
        os.mknod(file_name)

    with open(file_name, 'w', encoding='utf-8') as f:
        data = {"qps": args.qps,
                "cv": args.coefficient_variation,
                "request_ids": request_ids,  # [num_requests]
                "request_lens": request_lens,  # [num_requests]
                "request_latencies": request_latencies,  # [num_requests]
                "prefill_token_latencies": prefill_token_latencies,  # [num_requests]
                "decode_token_latencies": decode_token_latencies,  # [num_requests]
                "decode_sum_latencies": decode_sum_latencies,  # [num_requests]
                "all_decode_token_latencies": all_decode_token_latencies,  # [SUM(tokens)]
                "inference_latencies": inference_latencies,  # [num_requests]
                "per_token_latencies_breakdown_dict": per_token_latencies_breakdown_dict,
                "throughput": throughput[0],  # tokens/s
                "throughput_prefill": throughput[1],  # tokens/s
                "throughput_decode": throughput[2],  # tokens/s
                "instance_num": avg_instance_num  # gpu/s
                }
        # per_token_latencies_breakdown_dict: [{
        #     'step_latency_engine': [],
        #     'step_postprocess_latency': [],
        #     'across_async_put_queue_thread_latency': [],
        #     'across_async_put_queue_actor_latency': [],
        #     'queue_rpc_latency': [],
        #     'background_process_get_queue_latency': [],
        #     'generate_benchmark_return_output_latency': []
        # }]

        results.append(data)
        json.dump(results, f)

    # Zhixin: log latency and throughput data to tensorboard
    def cal_lat(latencies):
        return "[P50, P80, P95, P99, P99.9, mean]: [{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(
            np.percentile(latencies, 50),
            np.percentile(latencies, 80),
            np.percentile(latencies, 95),
            np.percentile(latencies, 99),
            np.percentile(latencies, 99.9),
            np.mean(latencies),
        )
    workload = {
        'qps': args.qps,
        'cv': args.coefficient_variation,
        'dataset': args.dataset_type,
        'datapath': args.dataset_path,
        'random_prompt_count': args.random_prompt_count,
    }
    writer.add_text('metric.workload', '\n'.join([f'{k}: {v}' for k, v in workload.items()]))
    writer.add_text('metric.throughput', str(throughput[0]))
    writer.add_text('metric.throughput_prefill', str(throughput[1]))
    writer.add_text('metric.throughput_decode', str(throughput[2]))
    writer.add_text('metric.latency_TTFT_ms', cal_lat(prefill_token_latencies))
    writer.add_text('metric.latency_TBT_ms', cal_lat(decode_token_latencies))
    writer.add_text('metric.latency_req_sec', cal_lat(request_latencies))


if __name__ == '__main__':
    main()
