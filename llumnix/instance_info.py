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

    # on-demand init infos
    dispatch_load_metric: float = -np.inf
    migration_load_metric: float = np.inf
    migration_load_metric_after_migrate_in: float = -np.inf
    migration_load_metric_after_migrate_out: float = np.inf

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
        instance_info.migration_load_metric_after_migrate_out = self.migration_load_calculator. \
            compute_instance_load_after_migrate(instance_info, is_migrate_in=False)
        instance_info.migration_load_metric_after_migrate_in = self.migration_load_calculator. \
            compute_instance_load_after_migrate(instance_info, is_migrate_in=True)


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
        instance_info_after_migrate = copy.deepcopy(instance_info)
        num_blocks_last_running_request = instance_info_after_migrate.num_blocks_last_running_request

        if is_migrate_in:
            instance_info_after_migrate.num_running_requests += 1
            # TODO(Zhixin): Why minus num_blocks_last_running_request?
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
            if instance_info.instance_type not in [InstanceType.DECODE, InstanceType.NO_CONSTRAINTS]:
                logger.error(f'predicted_remaining_blocks is not supported for {instance_info.instance_type}')

            if not self.enable_defrag:
                raise ValueError('predicted_remaining_blocks is not supported without defrag')
            else:
                num_requests = instance_info.num_running_requests + instance_info.num_waiting_requests
                if num_requests == 0:
                    return instance_info.num_available_gpu_blocks * -2
                num_available_gpu_blocks = instance_info.num_available_gpu_blocks - \
                                           instance_info.num_blocks_all_waiting_requests
                if instance_info.instance_type == InstanceType.DECODE:
                    logger.warning(
                        f'init num_available_gpu_blocks: {num_available_gpu_blocks}  |  new num_available_gpu_blocks: {num_available_gpu_blocks - instance_info.num_preserved_blocks}')
                    num_available_gpu_blocks -= instance_info.num_preserved_blocks

            instance_load = num_available_gpu_blocks * (-1)
        elif self.load_metric == 'predicted_used_blocks':
            if not self.enable_defrag:
                raise ValueError('predicted_used_blocks is not supported without defrag')
            # Fixme(Zhixin): "used_blocks" should be a percentage instead of an absolute number
            instance_load = instance_info.num_used_gpu_blocks + instance_info.num_preserved_blocks


        # TODO(Zhixin): add compute load metric here!!!!!!!!!!!


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
