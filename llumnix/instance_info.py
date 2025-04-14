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

from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple
import numpy as np

from llumnix.logging.logger import init_logger
from llumnix.llumlet.request import RequestInferenceType
from llumnix.modeling_utils.calculator import *

logger = init_logger(__name__)


class InstanceType(str, Enum):
    NO_CONSTRAINTS = "no_constraints"
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class InstanceInfo:
    instance_id: str = ""
    instance_id_str: str = "NONE"  # ZhiXin: add instance_id_str for tensorboard
    instance_type: InstanceType = None

    step_id: int = None
    timestamp: float = None
    num_batched_tokens: int = None
    num_seqs = None
    running_seq_lens: List[int] = field(default_factory=list)
    last_inference_latency: float = None
    inference_type: RequestInferenceType = None

    num_total_gpu_blocks: int = 0
    num_watermark_blocks: int = 0
    num_used_gpu_blocks: int = 0
    num_free_gpu_blocks: int = 0
    gpu_cache_usage: float = 0.0
    num_running_requests: int = 0
    num_waiting_requests: int = 0
    num_killed_requests: int = 0
    num_blocks_first_waiting_request: int = 0
    waiting_time_first_waiting_request: int = 0
    num_blocks_all_waiting_requests: int = 0
    num_blocks_last_running_request: int = 0

    # Zhixin: [PRED] num_preserved_blocks used for predicted but not yet generated tokens
    num_preserved_blocks: int = 0
    running_seq_predicted_lens: List[int] = field(default_factory=list)

    # on-demand init infos
    dispatch_load_metric: float = -np.inf
    migration_load_metric: float = np.inf
    migration_load_metric_after_migrate_in: float = -np.inf
    migration_load_metric_after_migrate_out: float = np.inf

    # [Zhixin] extra instance load metric (calculated in manager)
    load_100_memory: float = 0
    load_100_bandwidth: float = 0
    load_100_compute: float = 0

    # lazy init infos
    num_available_gpu_blocks: int = 0
    num_available_gpu_blocks_waiting: int = 0

    # manual init infos
    profiling_data: Tuple[str, int, int, float] = None

    def __post_init__(self) -> None:
        self.num_available_gpu_blocks = self.num_free_gpu_blocks - self.num_watermark_blocks
        self.num_available_gpu_blocks_waiting = self.num_available_gpu_blocks - self.num_blocks_all_waiting_requests


class InstanceLoadCalculator:
    def __init__(self, dispatch_load_metric: str, migration_load_metric: str, enable_defrag: bool) -> None:
        self.dispatch_load_calculator = DispatchLoadComputation(dispatch_load_metric)
        self.migration_load_calculator = MigrationLoadComputation(migration_load_metric, enable_defrag)

    def compute_instance_load(self, instance_info: InstanceInfo):
        instance_info.dispatch_load_metric = self.dispatch_load_calculator.compute_instance_load(instance_info)
        instance_info.migration_load_metric = self.migration_load_calculator.compute_instance_load(instance_info)

        # [Zhixin] only used for balanced migration policy
        instance_info.migration_load_metric_after_migrate_out = self.migration_load_calculator. \
            compute_instance_load_after_migrate(instance_info, is_migrate_in=False)
        instance_info.migration_load_metric_after_migrate_in = self.migration_load_calculator. \
            compute_instance_load_after_migrate(instance_info, is_migrate_in=True)


class CustomLoadCalculator:
    # Fixme(Zhixin): should consider waiting request (with small max_num_seqs)
    def __init__(self,
                 model_name: str = 'Qwen2.5-7B',
                 hardware_name: str = 'A800_practical',
                 block_size: str = 16,
                 TPOT_ms: float = 50,
                 TTFT_ms: float = 1000
                 ):
        # [Zhixin] currently only support decode phase
        self.model_name = model_name
        self.model_config = ModelConfig(model_name)
        self.hardware_name = hardware_name
        self.hw_config = HWConfig(hardware_name)
        self.block_size = block_size
        self.TPOT_s = TPOT_ms / 1000
        self.TTFT_s = TTFT_ms / 1000

        # TODO(Zhixin): add a factor to max_bandwidth_gbps and max_compute_gflops to simulate the real situation

        self.max_bandwidth_gbps = self.hw_config.memory_bw  # GB/s
        self.max_compute_gflops = self.hw_config.compute * 1e3  # GFLOPS
        self.weight_size_gb = compute_param_size(model_name, self.model_config) * 2 / (1024 ** 3)  # GB
        self.kv_size_per_block_gb = kv_size_per_token(self.model_config, 2) * self.block_size / (1024 ** 3)  # GB

    def compute_instance_load(self, info: InstanceInfo):
        info.load_100_memory = self.get_memory_load(
            info.num_used_gpu_blocks, info.num_preserved_blocks, info.num_total_gpu_blocks)
        if info.instance_type == 'decode':
            info.load_100_bandwidth = self.get_bandwidth_load(
                info.num_blocks_last_running_request, info.num_preserved_blocks, is_decode=True)
            info.load_100_compute = self.get_compute_load(info.running_seq_lens, is_decode=True)
        else:
            info.load_100_bandwidth = 0
            info.load_100_compute = 0

    def get_memory_load(self, blocks_used, blocks_preserved, blocks_total) -> float:
        return (blocks_used + blocks_preserved) / blocks_total

    def get_bandwidth_load(self, blocks_generated, blocks_preserved, is_decode: bool) -> float:
        # Fixme(Zhixin): this may not precise, since the activation also need to be transferred
        if not is_decode:
            raise ValueError("Currently only support decode phase.")
        usable_memory_bw_per_step = self.max_bandwidth_gbps * self.TPOT_s - self.weight_size_gb
        return (blocks_generated + blocks_preserved) * self.kv_size_per_block_gb / usable_memory_bw_per_step

    def get_compute_load(self, running_seq_lens: List[int], is_decode: bool) -> float:
        if not is_decode:
            raise ValueError("Currently only support decode phase.")
        usable_compute_per_step = self.max_compute_gflops * self.TPOT_s
        compute_per_step = 0
        for seq_len in running_seq_lens:
            compute_per_step += compute_load_decode(self.model_name, self.model_config, seq_len)
        compute_per_step /= (1000 ** 3)  # GFLOPs
        return compute_per_step / usable_compute_per_step


class LoadComputationStrategy(ABC):
    def __init__(self, load_metric: str, enable_defrag: bool = False) -> None:
        self.load_metric = load_metric
        self.enable_defrag = enable_defrag

    @abstractmethod
    def compute_instance_load(self, instance_info: InstanceInfo) -> float:
        pass


class DispatchLoadComputation(LoadComputationStrategy):
    def compute_instance_load(self, instance_info: InstanceInfo) -> float:
        instance_load = -np.inf
        if self.load_metric == 'usage_ratio':
            instance_load = (instance_info.num_used_gpu_blocks + instance_info.num_blocks_all_waiting_requests) \
                            / instance_info.num_total_gpu_blocks
        elif self.load_metric == 'remaining_steps':
            num_requests = instance_info.num_running_requests + instance_info.num_waiting_requests
            num_available_gpu_blocks = instance_info.num_available_gpu_blocks - instance_info.num_blocks_all_waiting_requests
            if num_requests == 0:
                # return -np.inf
                return num_available_gpu_blocks * -2  # ZhiXin: change to return the number of available blocks
            instance_load = (num_available_gpu_blocks / num_requests) * (-1)
        elif self.load_metric == 'predicted_remaining_blocks':
            raise ValueError('predicted_remaining_blocks should not be used for dispatch load computation.')
            num_requests = instance_info.num_running_requests + instance_info.num_waiting_requests
            num_available_gpu_blocks = instance_info.num_available_gpu_blocks - instance_info.num_blocks_all_waiting_requests
            # Zhixin: only decode instances have valid num_preserved_blocks
            if instance_info.instance_type == InstanceType.DECODE:
                num_available_gpu_blocks -= instance_info.num_preserved_blocks
            if num_requests == 0:
                return num_available_gpu_blocks * -2
            instance_load = num_available_gpu_blocks * -1

            temp = (instance_info.num_available_gpu_blocks - instance_info.num_blocks_all_waiting_requests) \
                   / num_requests * (-1)
            logger.warning(f'[LOAD] predicted_remaining_blocks vs remaining_steps: {instance_load} | {temp}')
        else:
            logger.error(f"Invalid dispatch load metric: {self.load_metric}")
        return instance_load


class MigrationLoadComputation(LoadComputationStrategy):
    def compute_instance_load_after_migrate(self, instance_info: InstanceInfo, is_migrate_in: bool) -> float:
        # used for balanced migration policy -> estimate instance load after migrate_in / migrate_out
        instance_info_after_migrate = copy.deepcopy(instance_info)
        num_blocks_last_running_request = instance_info_after_migrate.num_blocks_last_running_request

        if is_migrate_in:
            instance_info_after_migrate.num_running_requests += 1
            instance_info_after_migrate.num_available_gpu_blocks -= num_blocks_last_running_request
        else:
            instance_info_after_migrate.num_running_requests -= 1
            instance_info_after_migrate.num_available_gpu_blocks += num_blocks_last_running_request

        return self.compute_instance_load(instance_info_after_migrate)

    def compute_instance_load(self, instance_info: InstanceInfo) -> float:
        instance_load = -np.inf
        if self.load_metric == 'usage_ratio':
            instance_load = (instance_info.num_used_gpu_blocks + instance_info.num_blocks_first_waiting_request) \
                            / instance_info.num_total_gpu_blocks
        elif self.load_metric == 'remaining_steps':
            if not self.enable_defrag:
                num_requests = instance_info.num_running_requests
                num_available_gpu_blocks = instance_info.num_available_gpu_blocks
            else:
                num_requests = instance_info.num_running_requests
                if instance_info.num_waiting_requests != 0:
                    num_requests += 1
                num_available_gpu_blocks = instance_info.num_available_gpu_blocks - \
                                           instance_info.num_blocks_first_waiting_request
            if num_requests == 0:
                # return -np.inf
                return num_available_gpu_blocks * -2  # ZhiXin: change to return the number of available blocks
            instance_load = (num_available_gpu_blocks / num_requests) * (-1)
        elif self.load_metric == 'predicted_remaining_blocks':
            if not self.enable_defrag:
                raise ValueError('predicted_remaining_blocks is not supported without defrag')
            else:
                num_requests = instance_info.num_running_requests + instance_info.num_waiting_requests
                if num_requests == 0:
                    return instance_info.num_available_gpu_blocks * -2
                num_available_gpu_blocks = instance_info.num_available_gpu_blocks - \
                                           instance_info.num_blocks_all_waiting_requests
                if instance_info.instance_type == InstanceType.DECODE:
                    num_available_gpu_blocks -= instance_info.num_preserved_blocks

            instance_load = num_available_gpu_blocks * (-1)
        elif self.load_metric == 'predicted_used_blocks':
            if not self.enable_defrag:
                raise ValueError('predicted_used_blocks is not supported without defrag')
            # Fixme(Zhixin): "used_blocks" should be a percentage instead of an absolute number
            instance_load = instance_info.num_used_gpu_blocks + instance_info.num_preserved_blocks

        elif self.load_metric == 'sct_max':
            # [Zhixin] custom load for migration
            instance_load = max(
                instance_info.load_100_bandwidth, instance_info.load_100_compute, instance_info.load_100_memory)
        elif self.load_metric == 'sct_mem':
            instance_load = instance_info.load_100_memory
        elif self.load_metric == 'sct_bw':
            instance_load = instance_info.load_100_bandwidth
        elif self.load_metric == 'sct_comp':
            instance_load = instance_info.load_100_compute

        else:
            logger.error(f"Invalid migration load metric: {self.load_metric}")
        return instance_load


# TODO(KuilongCui): currently scaling and dispatch use the same load calculator, leave
# it in the future to refine
class ScalingLoadComputation(LoadComputationStrategy):
    def __init__(self, load_metric):
        super().__init__(load_metric)
        self.load_calculator = DispatchLoadComputation(load_metric)

    def compute_instance_load(self, instance_info: InstanceInfo) -> float:
        return self.load_calculator.compute_instance_load(instance_info)
