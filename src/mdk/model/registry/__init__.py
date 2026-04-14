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

"""Public facade function for the model.registry module."""

import logging
from typing import Optional, Dict, Any, Tuple

from google.cloud import aiplatform

from mdk.model.registry.models import RegistryAppConfig
from mdk.model.registry.providers.factory import ProviderFactory
from mdk.model.registry.clients.expanded_model_registry import (
    ExpandedModelRegistryClient,
)

logger = logging.getLogger(__name__)


def upload_model(
    config: RegistryAppConfig,
    artifact_folder_uri: str,
    registry_provider_name: str = "vertex",
    performance_metrics_summary: dict | None = None,
    vertex_ai_pipeline_job_run_id: str | None = None,
    access_token: str | None = None,
) -> str:
    """
    High-level function to upload a model. This is the main entry point
    for using this package as a library for model registration.

    Args:
        config: The fully populated RegistryAppConfig object.
        artifact_folder_uri: The URI of the model to upload.
        registry_provider_name: The registry provider. Defaults to 'vertex'.
        performance_metrics_summary: Performance metrics associated with the trained model.
        vertex_ai_pipeline_job_run_id: Vertex AI pipeline job ID associated with the trained model.
        access_token:  Optional. An auth token. If not provided, one will be generated.

    Returns:
        The resource name of the uploaded model in Vertex AI Model Registry.
    """
    logger.info(f"Starting model upload using '{registry_provider_name}' provider.")

    # Extract required GCP parameters from RegistryAppConfig for the provider
    gcp_project_id = config.gcp.project_id
    gcp_region = config.gcp.region
    expanded_model_registry_endpoint = config.gcp.expanded_model_registry_endpoint

    provider = ProviderFactory.get_provider(
        provider_name=registry_provider_name,
        project_id=gcp_project_id,
        region=gcp_region,
        expanded_model_registry_endpoint=expanded_model_registry_endpoint,
        app_config_for_upload=config,  # Pass the full RegistryAppConfig for upload operations
        access_token=access_token,
    )
    uploaded_model_resource_name = provider.upload(
        artifact_folder_uri=artifact_folder_uri,
        performance_metrics=performance_metrics_summary,
        pipeline_job_id=vertex_ai_pipeline_job_run_id,
    )

    logger.info("Model upload completed successfully.")
    return uploaded_model_resource_name


def get_model_uri_for_inference(
    model_reference_config_data: Dict[str, Any],
    gcp_project_id: str,
    gcp_region: str,
    expanded_model_registry_endpoint: str,
    access_token: Optional[str] = None,
    registry_provider_name: str = "vertex",
) -> str:
    """
    High-level function to retrieve the GCS URI of a model artifact for batch inference.
    This function acts as the main entry point for using the package to locate models.

    Args:
        model_reference_config_data: A dictionary containing the parameters for identifying the model.
                               This data will be validated against the ModelReferenceConfig Pydantic model.
                               Expected keys: "model_inference_reference", "model_name",
                               "deployment_environment", "vertex_ai_model_resource_name", etc.
        gcp_project_id: The Google Cloud project ID where the model registry operates.
        gcp_region: The Google Cloud region where the model registry operates.
        expanded_model_registry_endpoint: The endpoint URL for the Expanded Model Registry service.
        access_token:  Optional. An auth token. If not provided, one will be generated.
        registry_provider_name: The registry provider to use for model lookup. Defaults to 'vertex'.

    Returns:
        The GCS URI of the model artifacts.

    Raises:
        ValueError: If the model reference configuration is invalid or the model URI cannot be determined.
        Exception: Any exception raised by the underlying EMR client or Vertex AI SDK.
    """
    logger.info(
        f"Starting model URI retrieval using '{registry_provider_name}' provider."
    )

    # Get the provider instance, which will initialize necessary clients (like EMR)
    provider = ProviderFactory.get_provider(
        provider_name=registry_provider_name,
        project_id=gcp_project_id,
        region=gcp_region,
        expanded_model_registry_endpoint=expanded_model_registry_endpoint,
        app_config_for_upload=None,  # No RegistryAppConfig needed for model reference retrieval
        access_token=access_token,
    )

    # Delegate the retrieval to the provider's specific method
    model_uri = provider.get_model_uri_for_inference(model_reference_config_data)

    logger.info(f"Model URI retrieval completed successfully: {model_uri}")
    return model_uri


def get_emr_model_object(
    model_reference_config_data: Dict[str, Any],
    gcp_project_id: str,
    gcp_region: str,
    expanded_model_registry_endpoint: str,
    access_token: Optional[str] = None,
    registry_provider_name: str = "vertex",
) -> Dict[str, Any]:
    """
    High-level function to retrieve the entire model record from the Expanded Model Registry
    as a dictionary based on the provided configuration.

    Args:
        model_reference_config_data: A dictionary containing the parameters for identifying the model.
                                     Expected keys: "model_inference_reference", "model_name",
                                     "deployment_environment", etc., depending on the strategy.
        gcp_project_id: The Google Cloud project ID where the model registry operates.
        gcp_region: The Google Cloud region where the model registry operates.
        expanded_model_registry_endpoint: The endpoint URL for the Expanded Model Registry service.
        access_token:  Optional. An auth token. If not provided, one will be generated.
        registry_provider_name: The registry provider to use for model lookup. Defaults to 'vertex'.

    Returns:
        A dictionary representing the full model record from the Expanded Model Registry.
        Returns an empty dictionary if the strategy does not involve an EMR lookup
        (e.g., GCS URI or direct Vertex AI model reference).

    Raises:
        ValueError: If the model reference configuration is invalid or the model cannot be determined.
        Exception: Any exception raised by the underlying EMR client.
    """
    logger.info(
        f"Starting EMR model object retrieval using '{registry_provider_name}' provider."
    )

    provider = ProviderFactory.get_provider(
        provider_name=registry_provider_name,
        project_id=gcp_project_id,
        region=gcp_region,
        expanded_model_registry_endpoint=expanded_model_registry_endpoint,
        app_config_for_upload=None,
        access_token=access_token,
    )

    model_response_dict = provider.get_emr_model_object(model_reference_config_data)

    logger.info(f"EMR model object retrieved successfully: {model_response_dict}")
    return model_response_dict


def get_vertex_ai_model_object_for_inference(
    model_reference_config_data: Dict[str, Any],
    gcp_project_id: str,
    gcp_region: str,
    expanded_model_registry_endpoint: str,
    access_token: Optional[str] = None,
    registry_provider_name: str = "vertex",
) -> Optional[aiplatform.Model]:
    """
    High-level function to retrieve the aiplatform.Model object for a registered Vertex AI model.

    Args:
        model_reference_config_data: A dictionary containing the parameters for identifying the model.
        gcp_project_id: The Google Cloud project ID.
        gcp_region: The Google Cloud region.
        expanded_model_registry_endpoint: The endpoint URL for the Expanded Model Registry service.
        access_token:  Optional. An auth token. If not provided, one will be generated.
        registry_provider_name: The registry provider to use for model lookup. Defaults to 'vertex'.

    Returns:
        An instance of aiplatform.Model if the configuration resolves to a registered
        Vertex AI model, otherwise None (e.g., if it's a direct GCS URI).

    Raises:
        ValueError: If the model reference configuration is invalid or the model cannot be determined.
        Exception: Any exception raised by the underlying EMR client or Vertex AI SDK.
    """
    logger.info(
        f"Starting Vertex AI Model object retrieval using '{registry_provider_name}' provider."
    )

    provider = ProviderFactory.get_provider(
        provider_name=registry_provider_name,
        project_id=gcp_project_id,
        region=gcp_region,
        expanded_model_registry_endpoint=expanded_model_registry_endpoint,
        app_config_for_upload=None,
        access_token=access_token,
    )

    model_object = provider.get_vertex_ai_model_object(model_reference_config_data)

    if model_object:
        logger.info(
            f"Vertex AI Model object retrieved successfully: {model_object.resource_name}"
        )
    else:
        logger.info(
            "Vertex AI Model object could not be retrieved (e.g., not a registered model)."
        )
    return model_object


def get_vertex_ai_model_resource_name_and_version_for_inference(
    model_reference_config_data: Dict[str, Any],
    gcp_project_id: str,
    gcp_region: str,
    expanded_model_registry_endpoint: str,
    access_token: Optional[str] = None,
    registry_provider_name: str = "vertex",
) -> Tuple[Optional[str], Optional[str]]:
    """
    High-level function to retrieve the Vertex AI model's resource name and version ID.

    Args:
        model_reference_config_data: A dictionary containing the parameters for identifying the model.
        gcp_project_id: The Google Cloud project ID.
        gcp_region: The Google Cloud region.
        expanded_model_registry_endpoint: The endpoint URL for the Expanded Model Registry service.
        access_token:  Optional. An auth token. If not provided, one will be generated.
        registry_provider_name: The registry provider to use for model lookup. Defaults to 'vertex'.

    Returns:
        A tuple (resource_name, version_id) if the configuration resolves to a registered
        Vertex AI model, otherwise (None, None).

    Raises:
        ValueError: If the model reference configuration is invalid or the model cannot be determined.
        Exception: Any exception raised by the underlying EMR client or Vertex AI SDK.
    """
    logger.info(
        f"Starting Vertex AI Model resource name and version retrieval using '{registry_provider_name}' provider."
    )

    provider = ProviderFactory.get_provider(
        provider_name=registry_provider_name,
        project_id=gcp_project_id,
        region=gcp_region,
        expanded_model_registry_endpoint=expanded_model_registry_endpoint,
        app_config_for_upload=None,
        access_token=access_token,
    )

    resource_name, version_id = provider.get_vertex_ai_model_resource_name_and_version(
        model_reference_config_data
    )

    if resource_name:
        logger.info(
            f"Vertex AI Model resource details retrieved: resource='{resource_name}', version='{version_id}'"
        )
    else:
        logger.info(
            "Vertex AI Model resource details could not be retrieved (e.g., not a registered model)."
        )
    return resource_name, version_id


# Define what symbols are exposed when 'from mdk.model.registry import *' is used.
__all__ = [
    "RegistryAppConfig",
    "upload_model",
    "get_model_uri_for_inference",
    "get_emr_model_object",
    "get_vertex_ai_model_object_for_inference",
    "get_vertex_ai_model_resource_name_and_version_for_inference",
    "ExpandedModelRegistryClient",
]

# Optional: Set the version of your package.
__version__ = "0.1.0"
