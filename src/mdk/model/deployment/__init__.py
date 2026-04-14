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

"""Public facade function for the model.deployment module."""

import logging
from typing import Optional

from mdk.model.deployment.models import DeploymentAppConfig
from mdk.model.deployment.providers.factory import ProviderFactory


def deploy_model(
    config: DeploymentAppConfig,
    deployment_provider_name: Optional[str] = "vertex",
    access_token: Optional[str] = None,
) -> str:
    """
    High-level function to deploy a model. This is the main entry point
    for using this package as a library.

    Args:
        config: The fully populated DeploymentAppConfig object.
        deployment_provider_name: The deployment provider. Defaults to 'vertex'.
        access_token:  Optional. An auth token. If not provided, one will be generated.

    Returns:
        The resource name of the deployed endpoint.
    """
    logger = logging.getLogger(__name__)
    logger.info(
        f"Starting model deployment using '{deployment_provider_name}' provider."
    )

    provider = ProviderFactory.get_provider(
        deployment_provider_name, config, access_token
    )
    endpoint_resource_name = provider.deploy()

    logger.info("Model deployment completed successfully.")
    return endpoint_resource_name


def delete_model_endpoint(
    config: DeploymentAppConfig,
    endpoint_resource_name: str,
    deployment_provider_name: str = "vertex",
):
    """
    High-level function to delete an endpoint. This is the main entry point
    for using this package as a library.

    Args:
        config: The fully populated DeploymentAppConfig object.
        endpoint_resource_name: The full resource name of the endpoint to delete
                                (e.g., 'projects/PROJECT_ID/locations/REGION/endpoints/ENDPOINT_ID').
        deployment_provider_name: The deployment provider. Defaults to 'vertex'.

    Returns:
        The resource name of the deployed endpoint.
    """
    logger = logging.getLogger(__name__)
    logger.info(
        f"Starting endpoint deletion using '{deployment_provider_name}' provider."
    )

    provider = ProviderFactory.get_provider(deployment_provider_name, config)
    provider.delete_endpoint(endpoint_resource_name=endpoint_resource_name)

    logger.info("Endpoint deletion completed successfully.")


__all__ = ["DeploymentAppConfig", "deploy_model", "delete_model_endpoint"]

__version__ = "0.1.0"
