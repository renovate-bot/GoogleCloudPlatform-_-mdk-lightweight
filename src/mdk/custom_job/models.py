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

"""Defines the Pydantic models used for configuration validation."""

from typing import List, Dict, Optional, Union
import logging
from pydantic import BaseModel, model_validator, ConfigDict

# Import specific types from the SDK for accurate type hinting
from google.cloud.aiplatform_v1.types.custom_job import Scheduling
from google.cloud.aiplatform_v1.types.service_networking import PscInterfaceConfig
from google.cloud.aiplatform.metadata.experiment_resources import Experiment
from google.cloud.aiplatform.metadata.experiment_run_resource import ExperimentRun

logger = logging.getLogger(__name__)


class CustomJobCommonConfig(BaseModel):
    """Common configuration parameters for a Vertex AI Custom Job."""

    display_name: str
    args: Optional[List[str]] = None
    image_uri: str
    env_vars: Optional[Dict[str, str]] = None
    machine_type: str = "n1-standard-4"
    accelerator_type: Optional[str] = None
    accelerator_count: int = 0
    replica_count: int = 1
    labels: Optional[Dict[str, str]] = None
    project: Optional[str] = None
    location: Optional[str] = None
    base_output_dir: Optional[str] = None
    encryption_spec_key_name: Optional[str] = None
    staging_bucket: Optional[str] = None
    service_account: Optional[str] = None
    network: Optional[str] = None
    timeout: Optional[int] = None  # in seconds
    enable_web_access: bool = False
    experiment: Optional[Union[Experiment, str]] = None
    experiment_run: Optional[Union[ExperimentRun, str]] = None
    tensorboard: Optional[str] = None
    restart_job_on_worker_restart: bool = False
    create_request_timeout: Optional[float] = None
    disable_retries: bool = False
    persistent_resource_id: Optional[str] = None
    scheduling_strategy: Optional[Scheduling.Strategy] = None
    max_wait_duration: Optional[int] = None
    psc_interface_config: Optional[PscInterfaceConfig] = None
    # set 'arbitrary_types_allowed' to support Vertex AI input types
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_accelerator_config(self) -> "CustomJobCommonConfig":
        if self.accelerator_type and self.accelerator_count <= 0:
            # If type is specified, count should be at least 1, default will make it 1
            logger.warning(
                f"Accelerator type '{self.accelerator_type}' specified, but count is {self.accelerator_count}. "
                "Setting accelerator_count to 1."
            )
            self.accelerator_count = 1
        return self


class DirectJobSpecificConfig(BaseModel):
    """Specific configuration parameters for a direct container Custom Job."""

    command: Optional[List[str]] = None


class ScriptJobSpecificConfig(BaseModel):
    """Specific configuration parameters for a local script Custom Job."""

    script_path: str
    requirements: Optional[List[str]] = None
    python_module_name: Optional[str] = None

    @model_validator(mode="after")
    def validate_script_path(self) -> "ScriptJobSpecificConfig":
        if not self.script_path:
            raise ValueError("`script_path` must be provided for a script-based job.")
        return self
