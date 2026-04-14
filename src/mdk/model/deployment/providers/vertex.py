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

"""Implements the VertexAIDeploymentProvider for Google Cloud Vertex AI."""

import logging
from typing import Dict, Any, Optional

from google.cloud import aiplatform

import os

import mdk.config
from mdk.model.deployment.providers.base import DeploymentProvider
from mdk.model.deployment.models import DeploymentAppConfig
from mdk.model.registry.clients.expanded_model_registry import (
    ExpandedModelRegistryClient,
)
from mdk.model.deployment import strategies

logger = logging.getLogger(__name__)


class VertexAIDeploymentProvider(DeploymentProvider):
    """Concrete provider for deploying models to Google Cloud Vertex AI Endpoints."""

    def __init__(self, config: DeploymentAppConfig, access_token: Optional[str] = None):
        self.config = config
        self.access_token = access_token
        is_lite = os.environ.get("MDK_LITE_MODE") == "True"

        if not config.gcp.expanded_model_registry_endpoint or is_lite:
            logger.info(
                "Disabling Expanded Model Registry Client due to Lite Mode or empty endpoint."
            )
            self.registry_client = None
        else:
            self.registry_client = ExpandedModelRegistryClient(
                base_url=config.gcp.expanded_model_registry_endpoint,
                access_token=self.access_token,
            )
        logger.info(
            f"Initializing Vertex AI for project '{config.gcp.project_id}' in '{config.gcp.region}'"
        )
        aiplatform.init(project=config.gcp.project_id, location=config.gcp.region)

    def _select_endpoint_strategy(self) -> strategies.EndpointHandlingStrategy:
        if self.config.deployment.shadow_mode:
            logger.info("Deploying Model in Shadow Mode")
            return strategies.ShadowModeEndpointStrategy()
        return strategies.ExistingOrCreateEndpointStrategy()

    def _select_traffic_strategy(
        self, endpoint: aiplatform.Endpoint
    ) -> strategies.TrafficSplitStrategy:
        cfg = self.config.deployment
        if cfg.traffic_split:
            return strategies.ExplicitTrafficSplitStrategy()
        if cfg.is_primary_deployment:
            return strategies.PrimaryDeploymentTrafficSplitStrategy()
        if endpoint.traffic_split:
            return strategies.ChallengerTrafficSplitStrategy()
        return strategies.InitialDeploymentTrafficSplitStrategy()

    def _select_registry_update_strategy(
        self, traffic_strategy: strategies.TrafficSplitStrategy
    ) -> strategies.RegistryUpdateStrategy:
        # The registry update logic directly mirrors the traffic split logic
        if isinstance(traffic_strategy, strategies.ExplicitTrafficSplitStrategy):
            return strategies.ExplicitTrafficRegistryUpdateStrategy()
        if isinstance(
            traffic_strategy, strategies.PrimaryDeploymentTrafficSplitStrategy
        ):
            return strategies.PrimaryDeploymentRegistryUpdateStrategy()
        if isinstance(traffic_strategy, strategies.ChallengerTrafficSplitStrategy):
            return strategies.ChallengerRegistryUpdateStrategy()
        # Default/Initial is also a primary deployment in the registry
        return strategies.PrimaryDeploymentRegistryUpdateStrategy()

    def deploy(self) -> str:
        """Orchestrates the entire Vertex AI deployment process."""
        from mdk.model.registry import get_vertex_ai_model_object_for_inference

        # 1. Get or Create Endpoint
        endpoint_strategy = self._select_endpoint_strategy()
        endpoint = endpoint_strategy.get_or_create_endpoint(
            self.config.deployment, self.config.gcp.project_id
        )

        # 2. Get Model and Calculate Traffic Split
        logger.info(
            f"Attempting to load Vertex AI Model with model_reference_config: '{self.config.deployment.model_reference_config}'."
        )

        model_ref_fields = set(mdk.config.ModelReferenceConfig.model_fields.keys())
        model_reference_config_data = (
            self.config.deployment.model_reference_config.model_dump(
                include=model_ref_fields
            )
        )
        try:
            model_to_deploy = get_vertex_ai_model_object_for_inference(
                model_reference_config_data=model_reference_config_data,
                gcp_project_id=self.config.gcp.project_id,
                gcp_region=self.config.gcp.region,
                expanded_model_registry_endpoint=self.config.gcp.expanded_model_registry_endpoint,
                access_token=self.access_token,
            )
            logger.info(
                f"Successfully retrieved Vertex AI model with name: '{model_to_deploy.resource_name}', "
                f"Version: '{model_to_deploy.version_id}'"
            )
        except Exception as e:
            log_message = (
                f"An unexpected error occurred during Vertex AI Model loading for "
                f"model_reference_config: '{self.config.deployment.model_reference_config}'. "
                f"Original error: {e}"
            )
            logger.error(log_message, exc_info=True)
            raise RuntimeError(
                f"Deployment preparation failed due to an unexpected error while loading the model. "
                f"See logs for details. Error: {e}"
            ) from e
        logger.info(f"Preparing to deploy model: {model_to_deploy.resource_name}")

        # Re-fetch endpoint to ensure we have the latest traffic info before calculation
        endpoint = aiplatform.Endpoint(endpoint.resource_name)
        deployed_models_details = self._get_models_from_endpoint(endpoint)
        traffic_strategy = self._select_traffic_strategy(endpoint)
        traffic_split_dict = traffic_strategy.calculate_traffic_split(
            self.config.deployment, endpoint
        )

        # 3. Update or Deploy Model to Endpoint
        logger.info(
            f"Updating or deploying model to endpoint '{endpoint.display_name}' with traffic: {traffic_split_dict}"
        )
        deployment_strategy: strategies.DeploymentActionStrategy = (
            strategies.DeployNewModelToEndpointStrategy()
        )

        # Check if the exact model version is already deployed
        found_deployed_model_id = None
        for deployed_model_id, details in deployed_models_details.items():
            if details[
                "vertex_ai_model_resource_name"
            ] == model_to_deploy.resource_name and str(
                details["vertex_ai_model_version_id"]
            ) == str(model_to_deploy.version_id):
                found_deployed_model_id = deployed_model_id
                break

        if found_deployed_model_id:
            deployment_strategy = strategies.UpdateExistingModelDeploymentStrategy(
                deployed_model_id=found_deployed_model_id
            )

        # Execute the chosen strategy
        deployment_strategy.execute_deployment_action(
            endpoint=endpoint,
            model_to_deploy=model_to_deploy,
            traffic_split_dict=traffic_split_dict,
            deployed_model_display_name=self.config.deployment.model_reference_config.model_name,
            machine_type=self.config.deployment.machine_type,
            min_replica_count=self.config.deployment.min_replica_count,
            max_replica_count=self.config.deployment.max_replica_count,
            # **kwargs # Pass through any extra kwargs
        )
        logger.info(
            f"Successfully updated or deployed model to endpoint: {endpoint.resource_name}"
        )

        # 4. Update Expanded Model Registry
        if self.registry_client:
            registry_strategy = self._select_registry_update_strategy(traffic_strategy)
            registry_strategy.update_registry(
                client=self.registry_client,
                config=self.config.deployment,
                endpoint=endpoint,
                model_to_deploy=model_to_deploy,
                deployed_models_details=deployed_models_details,
            )
        else:
            logger.info(
                "Skipping Expanded Model Registry update (Lite Mode or empty endpoint)."
            )

        return endpoint.resource_name

    def delete_endpoint(self, endpoint_resource_name: str) -> None:
        """
        Deletes a Vertex AI endpoint along with any deployed models on it.

        Args:
            endpoint_resource_name: The full resource name of the endpoint to delete
                                    (e.g., 'projects/PROJECT_ID/locations/REGION/endpoints/ENDPOINT_ID').
        """
        if not endpoint_resource_name:
            raise ValueError("Endpoint resource name cannot be empty for deletion.")

        logger.info(
            f"Attempting to delete Vertex AI endpoint: {endpoint_resource_name}"
        )
        try:
            # Instantiate the Endpoint object using its full resource name
            endpoint = aiplatform.Endpoint(endpoint_resource_name)

            # Undeploy all models first, as an endpoint cannot be deleted if it has deployed models
            deployed_models = endpoint.list_models()
            if deployed_models:
                logger.info(
                    f"Undeploying {len(deployed_models)} models from endpoint {endpoint_resource_name} before deletion."
                )
                for deployed_model in deployed_models:
                    logger.info(
                        f"Undeploying model '{deployed_model.display_name}' (ID: {deployed_model.id}) from endpoint."
                    )
                    endpoint.undeploy(deployed_model_id=deployed_model.id)
                logger.info(f"All models undeployed from {endpoint_resource_name}.")
            else:
                logger.info(
                    f"No models found on endpoint {endpoint_resource_name} to undeploy."
                )

            # Delete the empty endpoint
            endpoint.delete(sync=True)
            logger.info(
                f"Successfully deleted Vertex AI endpoint: {endpoint_resource_name}"
            )

        except Exception as e:
            logger.error(
                f"An unexpected error occurred while deleting endpoint '{endpoint_resource_name}'. "
                f"Original error: {e}",
                exc_info=True,
            )
            raise RuntimeError(
                f"Failed to delete endpoint '{endpoint_resource_name}' due to an unexpected error. "
                f"See logs for details. Error: {e}"
            ) from e

    def undeploy_model(
        self, endpoint_resource_name: str, deployed_model_id: str
    ) -> None:
        """
        Deletes a Vertex AI endpoint along with any deployed models on it.

        Args:
            endpoint_resource_name: The full resource name of the endpoint to delete
                                    (e.g., 'projects/PROJECT_ID/locations/REGION/endpoints/ENDPOINT_ID').
            deployed_model_id: The ID of the DeployedModel to be undeployed from the Endpoint
                                    (e.g. 6678664524553256960).
        """
        if not endpoint_resource_name:
            raise ValueError("Endpoint resource name cannot be empty for deletion.")

        logger.info(
            f"Attempting to undeploy deployed model {deployed_model_id} from Vertex AI endpoint: {endpoint_resource_name}"
        )
        try:
            # Instantiate the Endpoint object using its full resource name
            endpoint = aiplatform.Endpoint(endpoint_resource_name)
            # Undeploy
            endpoint.undeploy(deployed_model_id=deployed_model_id)
            logger.info(
                f"Successfully undeployed deployed model {deployed_model_id} from Vertex AI endpoint: {endpoint_resource_name}"
            )

        except Exception as e:
            logger.error(
                f"An unexpected error occurred while undeploying model {deployed_model_id} from endpoint '{endpoint_resource_name}'. "
                f"Original error: {e}",
                exc_info=True,
            )
            raise RuntimeError(
                f"Failed to undeploy model {deployed_model_id} from endpoint '{endpoint_resource_name}' due to an unexpected error. "
                f"See logs for details. Error: {e}"
            ) from e

    def _get_models_from_endpoint(
        self, endpoint: aiplatform.Endpoint
    ) -> Dict[str, Any]:
        """Helper to get details of models currently on the endpoint."""
        models_dict = {}
        for model in endpoint.list_models():
            models_dict[model.id] = {
                "vertex_ai_model_resource_name": model.model,
                "vertex_ai_model_version_id": model.model_version_id,
            }
        return models_dict
