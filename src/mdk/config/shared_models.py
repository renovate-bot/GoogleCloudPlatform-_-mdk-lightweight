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

"""Shared Pydantic models used across mdk sub-packages."""

import re
from typing import Optional
from pydantic import BaseModel, Field, model_validator

from mdk.config import _util


class GCPConfig(BaseModel):
    """Configuration for Google Cloud Platform resources"""

    project_id: str = Field(..., description="Google Cloud project ID")
    region: str = Field(..., description="Google Cloud region")
    pipeline_service_account: str = Field(
        ..., description="Service account to use for Vertex AI Pipelines"
    )
    data_bucket: str = Field(
        ...,
        description="Google Cloud Storage (GCS) bucket used for general purpose storage",
    )
    pipeline_staging_dir: str = Field(
        ...,
        description="Google Cloud Storage (GCS) bucket and folder used as temp storage for Vertex AI Pipelines",
    )
    artifact_registry_repo: str = Field(
        ...,
        description="Artifact Registry repository in which to store container images",
    )
    expanded_model_registry_endpoint: str = Field(
        ..., description="URL for the Expanded Model Registry service"
    )
    experiment_name: str = Field(
        ..., description="Name to use for Vertex AI Experiments"
    )
    deployment_environment: str = Field(
        "dev", description="Deployment environment (e.g., 'dev', 'stage', 'prod')."
    )

    @classmethod
    def from_yaml_file(
        cls,
        filename: str,
    ) -> "GCPConfig":
        """Loads and validates configuration from YAML files.

        Args:
            filename (str): Filename containing YAML which shoudl be loaded
                and validated.

        Returns:
            GCPConfig: Loaded and validated GCPConfig object.
        """
        data = _util.readYAMLConfig(filename)
        try:
            obj = cls(**data)
        except Exception as e:
            e.add_note(f"Error reading file: {filename}")
            raise

        return obj


class ModelReferenceConfig(BaseModel):
    """Pydantic model for model reference configuration.

    This configures how a model URI should be resolved, either directly
    or by querying the Expanded Model Registry.
    """

    vertex_ai_model_resource_name: Optional[str] = Field(
        None,
        description="Explicit Vertex AI model resource name (e.g., projects/.../locations/.../models/...) for direct model reference.",
    )
    vertex_ai_model_version_id: Optional[str] = Field(
        None,
        description="Explicit Vertex AI model version ID (e.g., '1', 'v1.0.0') to use with vertex_ai_model_resource_name. If None, Vertex AI might use the default version.",
    )
    model_name: Optional[str] = Field(
        None,
        description="Logical name of the model, required for EMR lookup (latest, primary, semantic version).",
    )
    model_inference_reference: Optional[str] = Field(
        None,
        description="Strategy for model inference reference: 'latest', 'primary', semantic version (e.g., '1.0.0'), or GCS URI (e.g., 'gs://...').",
    )
    deployment_environment: str  # This field mirrors GCPConfig.deployment_environment

    @model_validator(mode="after")
    def validate_model_inference_reference_format(self) -> "ModelReferenceConfig":
        """Validates the format of the model_inference_reference string."""
        v = self.model_inference_reference
        if v is None:
            return self  # No validation needed if reference is not provided

        # Check for allowed string values or patterns
        if v in ["latest", "primary"] or v.startswith("gs://"):
            return self
        if re.fullmatch(r"^\d+\.\d+\.\d+$", v):
            return self

        # If none of the above conditions are met, raise an error
        raise ValueError(
            f"model_inference_reference '{v}' does not match expected formats: "
            "'latest', 'primary', semantic version (e.g., '1.0.0'), or GCS URI (starting with 'gs://')."
        )
