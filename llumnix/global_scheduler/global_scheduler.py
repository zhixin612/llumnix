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

from typing import Dict, List, Tuple, Union, Iterable, Set
import numpy as np
import math

from llumnix.logging.logger import init_logger
from llumnix.internal_config import GlobalSchedulerConfig
from llumnix.instance_info import InstanceInfo, InstanceType
from llumnix.global_scheduler.dispatch_scheduler import DispatchScheduler
from llumnix.global_scheduler.migration_scheduler import MigrationScheduler
from llumnix.global_scheduler.migration_policy import PairMigrationConstraints
from llumnix.global_scheduler.scaling_scheduler import ScalingScheduler
from llumnix.arg_utils import InstanceArgs

logger = init_logger(__name__)


class GlobalScheduler:
    def __init__(self, global_scheduler_config: GlobalSchedulerConfig) -> None:
        self.global_scheduler_config = global_scheduler_config
        self.num_instances = 0
        self.instance_id_set: Set[str] = set()

        # [Zhixin] used for minimal overhead dispatch & migration
        self.instance_ids_prefill = []
        self.instance_ids_decode = []
        self.num_requests = 0

    def dispatch(self) -> str:
        # minimal overhead dispatch: RR
        instance_id = self.instance_ids_prefill[self.num_requests % len(self.instance_ids_prefill)]
        request_expected_steps = 1 if self.global_scheduler_config.enable_pd_disagg else math.inf
        self.num_requests += 1
        return instance_id, request_expected_steps

    def pair_migration(self, pair_migration_type: PairMigrationConstraints) -> List[Tuple[str, str]]:
        # minimal overhead migration: p -> rand(D)
        assert pair_migration_type == PairMigrationConstraints.PREFILL_2_DECODING, "Only P2D is supported."
        migrate_instance_pairs = []
        for instance_id in self.instance_ids_prefill:
            dst_instance_id = np.random.choice(self.instance_ids_decode)
            migrate_instance_pairs.append((instance_id, dst_instance_id))
        return migrate_instance_pairs

    def scale_up(self, instance_id: Union[str, Iterable[str]], instance_args: List[InstanceArgs]) -> int:
        if isinstance(instance_id, str):
            instance_id = [instance_id,]
        instance_ids = list(instance_id)
        for ins_id, ins_args in zip(instance_ids, instance_args):
            if ins_id not in self.instance_id_set:
                logger.info("Scale up instance: {}.".format(ins_id))
                self._add_instance(ins_id, ins_args)
        logger.info("num_instances: {}, instances: {}".format(self.num_instances, self.instance_id_set))
        return self.num_instances

    def scale_down(self, instance_id: Union[str, Iterable[str]]) -> int:
        if isinstance(instance_id, str):
            instance_id = [instance_id,]
        instance_ids = list(instance_id)
        for ins_id in instance_ids:
            if ins_id in self.instance_id_set:
                logger.info("Scale down instance: {}.".format(ins_id))
                self._remove_instance(ins_id)
        logger.info("num_instances: {}, instances: {}".format(self.num_instances, self.instance_id_set))
        return self.num_instances

    def _add_instance(self, instance_id: str, instance_args: InstanceArgs) -> None:
        self.instance_id_set.add(instance_id)
        if instance_args.instance_type == InstanceType.PREFILL:
            self.instance_ids_prefill.append(instance_id)
        elif instance_args.instance_type == InstanceType.DECODE:
            self.instance_ids_decode.append(instance_id)
        else:
            raise ValueError(f"Unknown instance type: {instance_args.instance_type}")
        self.num_instances = len(self.instance_id_set)

    def _remove_instance(self, instance_id: str) -> None:
        self.instance_id_set.remove(instance_id)
        if instance_id in self.instance_ids_prefill:
            self.instance_ids_prefill.remove(instance_id)
        elif instance_id in self.instance_ids_decode:
            self.instance_ids_decode.remove(instance_id)
        else:
            raise ValueError(f"Unknown instance id: {instance_id}")
        self.num_instances = len(self.instance_id_set)
