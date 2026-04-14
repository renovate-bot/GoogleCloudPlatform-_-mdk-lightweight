# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines the abstract base class (interface) for all model monitoring providers."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional


class MonitoringProvider(ABC):
    """
    Abstract base class defining the interface for a model monitoring provider.
    """

    @abstractmethod
    def set_up_monitoring(
        self,
        baseline_dataset_uri: str,
        notification_channels: list[str] = [],
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        The main method to set up and run the entire monitoring process.

        Args:
            baseline_dataset_uri (str): The baseline dataset to compare drifts to.
                This is typically your training dataset.
            notification_channels (list[str]): Channels to which errors should be
                sent (e.g. for Webex notification)

            tuple: The elements of the returned tuple are, respectively:
                - The URL or resource name for the created monitoring job.
                - The URL or resource name for the created monitoring schedule.
        """
        pass
