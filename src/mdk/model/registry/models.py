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

import yaml
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

import mdk.config


class VertexAISettings(BaseModel):
    """Settings specific to the Vertex AI Model Registry."""

    model_name: str
    model_version_aliases: Optional[List[str]] = []
    model_version_description: Optional[str] = None
    # Default serving container image:
    serving_container_image_uri: Optional[
        str
    ]  # = "us-docker.pkg.dev/project-id/ar-repo-name/placeholder-serving-container:latest"
    serving_container_ports: Optional[List[int]] = [8080]
    model_labels: Dict[str, str] = {}


class ExpandedModelRegistrySettings(BaseModel):
    """Settings specific to the  expanded Model Registry."""

    major_version: int
    minor_version: int
    training_data_uri: Optional[str] = None
    model_semantic_version: Optional[str] = None
    path_to_production_projects: Dict[str, str] = {}
    is_sensitive_data: bool = False
    model_status: str = "training"
    trained_by: Optional[str] = None
    use_case: Optional[str] = None
    general_notes: Optional[str] = None
    git_repo_url: Optional[str] = None
    git_commit_hash: Optional[str] = None
    git_branch: Optional[str] = None
    vertex_ai_experiment_name: Optional[str] = None
    service_now_ticket_id: Optional[str] = None
    deployment_environment: str = "dev"
    deployment_endpoint_id: Optional[str] = None
    is_primary_deployment: bool = False
    deployment_status: Dict[str, Any] = {"status": "active"}
    publish_status: Optional[str] = None
    challenger_model_ids: Optional[List[str]] = None


class ModelRegistryConfig(BaseModel):
    """A container for all model metadata settings, parsed from the YAML."""

    vertex_mr: VertexAISettings = Field(alias="vertex_ai_model_registry_settings")
    expanded_mr: ExpandedModelRegistrySettings = Field(
        alias="expanded_model_registry_settings"
    )


class RegistryAppConfig(BaseModel):
    """A single, unified configuration object for the model registry application."""

    gcp: mdk.config.GCPConfig
    metadata: ModelRegistryConfig

    @classmethod
    def from_yaml_files(
        cls, gcp_config_path: str, general_config_path: str
    ) -> "RegistryAppConfig":
        """Loads and validates configuration from the specified YAML files."""
        with open(gcp_config_path, "r") as f:
            gcp_data = yaml.safe_load(f)
        environment = gcp_data["deployment_environment"]

        general_config = mdk.config.readAndMergeYAMLConfig(
            config_filename=general_config_path, environment=environment
        )
        model_name = general_config["general"]["model_name"]
        metadata_dict = general_config["model_registry"]

        # Logic to structure the flat metadata YAML into the nested Pydantic model
        vertex_mr_fields = ModelRegistryConfig.model_fields[
            "vertex_mr"
        ].annotation.model_fields.keys()
        expanded_mr_fields = ModelRegistryConfig.model_fields[
            "expanded_mr"
        ].annotation.model_fields.keys()

        structured_metadata = {
            "vertex_ai_model_registry_settings": {
                k: v for k, v in metadata_dict.items() if k in vertex_mr_fields
            },
            "expanded_model_registry_settings": {
                k: v for k, v in metadata_dict.items() if k in expanded_mr_fields
            },
        }
        # Get deployment_environment from the GCP config
        structured_metadata["expanded_model_registry_settings"][
            "deployment_environment"
        ] = environment
        structured_metadata["vertex_ai_model_registry_settings"]["model_name"] = (
            model_name
        )

        return cls(
            gcp=mdk.config.GCPConfig(**gcp_data),
            metadata=ModelRegistryConfig.model_validate(structured_metadata),
        )
