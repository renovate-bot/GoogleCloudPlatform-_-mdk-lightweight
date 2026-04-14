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

"""Pydantic models for the model.deployment module."""

import mdk.config
from typing import Dict, Optional
from pydantic import BaseModel, Field, model_validator, conint
import yaml


class DeploymentConfig(BaseModel):
    """Configuration for deploying a model to an endpoint."""

    model_reference_config: mdk.config.ModelReferenceConfig = Field(
        default_factory=mdk.config.ModelReferenceConfig,
        description="Specifies how to retrieve a given model.",
    )
    endpoint_name: Optional[str] = Field(
        None,
        description="Display name of the Vertex AI Endpoint. Defaults to model_reference_config.model_name.",
    )
    machine_type: Optional[str] = Field(
        "n2-standard-4",
        description="The machine type for the deployment (e.g., 'n1-standard-2').",
    )
    min_replica_count: conint(ge=1) = 1
    max_replica_count: conint(ge=1) = 1

    # Deployment strategy flags
    is_primary_deployment: bool = Field(
        False, description="If true, deploys model with 100% traffic, demoting others."
    )
    shadow_mode: bool = Field(
        False,
        description="If true, deploys to a new, separate endpoint for shadow testing.",
    )
    traffic_split: Optional[Dict[str, int]] = Field(
        None, description="Explicit traffic split. Overrides other logic if set."
    )

    @model_validator(mode="after")
    def validate_deployment_config(self) -> "DeploymentConfig":
        """Set default endpoint_name and validate replica counts."""
        # If endpoint_name is not provided, default it to model_reference_config.model_name.
        if not self.endpoint_name:
            self.endpoint_name = self.model_reference_config.model_name

        # Ensure min_replica_count is not greater than max_replica_count.
        if self.min_replica_count > self.max_replica_count:
            raise ValueError(
                "min_replica_count cannot be greater than max_replica_count"
            )
        return self


class DeploymentAppConfig(BaseModel):
    """A single, unified configuration object for the deployment application."""

    gcp: mdk.config.GCPConfig
    deployment: DeploymentConfig

    @classmethod
    def from_yaml_files(
        cls, gcp_config_path: str, general_config_path: str
    ) -> "DeploymentAppConfig":
        """Loads and validates configuration from YAML files."""
        with open(gcp_config_path, "r") as f:
            gcp_data = yaml.safe_load(f)
        environment = gcp_data["deployment_environment"]

        general_config = mdk.config.readAndMergeYAMLConfig(
            config_filename=general_config_path, environment=environment
        )

        general_data = general_config["general"]
        deployment_data = general_config["deployment"]

        # Handle the nested Pydantic Models
        model_ref_fields = set(mdk.config.ModelReferenceConfig.model_fields.keys())
        model_reference_data = {}

        for key, value in general_data.items():
            if key in model_ref_fields:
                model_reference_data[key] = value

        # Get deployment_environment from the GCP config
        model_reference_data["deployment_environment"] = environment
        # Combine model_reference_config with the deployment config data
        deployment_data["model_reference_config"] = model_reference_data

        return cls(
            gcp=mdk.config.GCPConfig(**gcp_data),
            deployment=DeploymentConfig(**deployment_data),
        )
