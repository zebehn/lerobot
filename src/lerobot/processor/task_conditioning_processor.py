#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.utils.constants import OBS_ENV_STATE

from .core import TransitionKey
from .pipeline import ObservationProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="task_index_to_env_state")
class TaskIndexToEnvStateStep(ObservationProcessorStep):
    """Inject a one-hot task id as `observation.environment_state`.

    Uses `task_index` from complementary data when available, and falls back to
    `info.task_id` or `info.task_index` if present.
    """

    num_tasks: int
    key: str = OBS_ENV_STATE

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.key in observation:
            return observation

        transition = self.transition
        task_index = None

        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        if "task_index" in complementary_data:
            task_index = complementary_data["task_index"]
        else:
            info = transition.get(TransitionKey.INFO) or {}
            if "task_id" in info:
                task_index = info["task_id"]
            elif "task_index" in info:
                task_index = info["task_index"]

        if task_index is None:
            return observation

        one_hot = self._make_one_hot(task_index)
        new_obs = dict(observation)
        new_obs[self.key] = one_hot
        return new_obs

    def _make_one_hot(self, task_index: Any) -> torch.Tensor:
        if isinstance(task_index, torch.Tensor):
            task_idx = task_index
        elif isinstance(task_index, np.ndarray):
            task_idx = torch.from_numpy(task_index)
        elif isinstance(task_index, (list, tuple)):
            task_idx = torch.tensor(task_index)
        else:
            task_idx = torch.tensor(task_index)

        task_idx = task_idx.long()
        if task_idx.dim() == 0:
            task_idx = task_idx.unsqueeze(0)
        elif task_idx.dim() > 1:
            task_idx = task_idx.view(-1)

        if task_idx.numel() == 0:
            return torch.zeros((0, self.num_tasks), dtype=torch.float32)

        max_idx = int(task_idx.max().item())
        min_idx = int(task_idx.min().item())
        if max_idx >= self.num_tasks or min_idx < 0:
            raise ValueError(
                f"task_index out of range [0, {self.num_tasks - 1}]: min={min_idx}, max={max_idx}"
            )

        one_hot = torch.zeros((task_idx.shape[0], self.num_tasks), dtype=torch.float32)
        one_hot.scatter_(1, task_idx.unsqueeze(1), 1.0)
        return one_hot
