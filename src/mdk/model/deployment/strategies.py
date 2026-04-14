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

"""Strategies for the model.deployment module."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from google.cloud import aiplatform

from mdk.model.deployment.models import DeploymentConfig
from mdk.model.registry.clients.expanded_model_registry import (
    ExpandedModelRegistryClient,
)

logger = logging.getLogger(__name__)

# --- Endpoint Handling Strategies ---


class EndpointHandlingStrategy(ABC):
    @abstractmethod
    def get_or_create_endpoint(self, config: DeploymentConfig) -> aiplatform.Endpoint:
        pass


class ShadowModeEndpointStrategy(EndpointHandlingStrategy):
    def get_or_create_endpoint(
        self, config: DeploymentConfig, project_id
    ) -> aiplatform.Endpoint:
        logger.info(f"Shadow mode: Creating new endpoint '{config.endpoint_name}'.")
        endpoint = aiplatform.Endpoint.create(
            display_name=config.endpoint_name,
            # Enable Response logging for setting up Model Monitoring. This will log incoming requests to the
            # endpoint in the defined BQ destination table.
            enable_request_response_logging=True,
            request_response_logging_sampling_rate=1.0,  # Default: 0.0
            request_response_logging_bq_destination_table=f"bq://{project_id}.vertex_ai.endpoint_logs",
        )
        return endpoint


class ExistingOrCreateEndpointStrategy(EndpointHandlingStrategy):
    def get_or_create_endpoint(
        self, config: DeploymentConfig, project_id
    ) -> aiplatform.Endpoint:
        logger.info(f"Searching for existing endpoint '{config.endpoint_name}'.")
        endpoints = aiplatform.Endpoint.list(
            filter=f'display_name="{config.endpoint_name}"'
        )
        if endpoints:
            logger.info(f"Found existing endpoint: {endpoints[0].resource_name}")
            return endpoints[0]
        logger.info(
            f"Endpoint not found. Creating new endpoint '{config.endpoint_name}'."
        )
        endpoint = aiplatform.Endpoint.create(
            display_name=config.endpoint_name,
            # Enable Response logging for setting up Model Monitoring. This will log incoming requests to the
            # endpoint in the defined BQ destination table.
            enable_request_response_logging=True,
            request_response_logging_sampling_rate=1.0,  # Default: 0.0
            request_response_logging_bq_destination_table=f"bq://{project_id}.vertex_ai.endpoint_logs",
        )
        return endpoint


# --- Traffic Split Calculation Strategies ---


class TrafficSplitStrategy(ABC):
    @abstractmethod
    def calculate_traffic_split(
        self, config: DeploymentConfig, endpoint: aiplatform.Endpoint
    ) -> Dict[str, int]:
        pass


class ExplicitTrafficSplitStrategy(TrafficSplitStrategy):
    def calculate_traffic_split(
        self, config: DeploymentConfig, endpoint: aiplatform.Endpoint
    ) -> Dict[str, int]:
        logger.info(f"Using explicit traffic split: {config.traffic_split}")
        # Basic validation
        if sum(config.traffic_split.values()) > 100:
            raise ValueError("Sum of traffic split percentages cannot exceed 100.")
        return config.traffic_split


class PrimaryDeploymentTrafficSplitStrategy(TrafficSplitStrategy):
    def calculate_traffic_split(
        self, config: DeploymentConfig, endpoint: aiplatform.Endpoint
    ) -> Dict[str, int]:
        logger.info("Primary deployment: Assigning 100% traffic to new model ('0').")
        traffic_split = {key: 0 for key in (endpoint.traffic_split or {})}
        traffic_split["0"] = 100
        return traffic_split


class ChallengerTrafficSplitStrategy(TrafficSplitStrategy):
    def calculate_traffic_split(
        self, config: DeploymentConfig, endpoint: aiplatform.Endpoint
    ) -> Dict[str, int]:
        logger.info("Challenger deployment: Adding new model ('0') with 0% traffic.")
        traffic_split = endpoint.traffic_split.copy() if endpoint.traffic_split else {}
        traffic_split["0"] = 0
        return traffic_split


class InitialDeploymentTrafficSplitStrategy(TrafficSplitStrategy):
    def calculate_traffic_split(
        self, config: DeploymentConfig, endpoint: aiplatform.Endpoint
    ) -> Dict[str, int]:
        logger.info(
            "Initial deployment on endpoint: Assigning 100% traffic to new model ('0')."
        )
        return {"0": 100}


# --- Registry Update Strategies ---


class RegistryUpdateStrategy(ABC):
    @abstractmethod
    def update_registry(
        self,
        client: ExpandedModelRegistryClient,
        config: DeploymentConfig,
        endpoint: aiplatform.Endpoint,
        model_to_deploy: aiplatform.Model,
        deployed_models_details: Dict[str, Any],
    ):
        pass


class ExplicitTrafficRegistryUpdateStrategy(RegistryUpdateStrategy):
    def update_registry(
        self, client, config, endpoint, model_to_deploy, deployed_models_details
    ):
        logger.info("Updating registry for explicit traffic split.")
        for deployed_model_id, percentage in config.traffic_split.items():
            if deployed_model_id == "0":
                resource_name = model_to_deploy.resource_name
                version_id = model_to_deploy.version_id
            else:
                resource_name = deployed_models_details[deployed_model_id][
                    "vertex_ai_model_resource_name"
                ]
                version_id = deployed_models_details[deployed_model_id][
                    "vertex_ai_model_version_id"
                ]
            publish_status = "champion" if percentage == 100 else "challenger"

            client.update_status(
                vertex_ai_model_resource_name=resource_name,
                vertex_ai_model_version_id=int(version_id),
                deployment_environment=config.model_reference_config.deployment_environment,
                deployed_model_endpoint=endpoint.resource_name,
                publish_status=publish_status,
                deployment_status={
                    "status": "active",
                    "deployment_type": "shadow" if config.shadow_mode else "canary",
                    "traffic_split_percentage": percentage,
                    "validation_result": "pending",
                    "inference_type": "online-inference",
                    "reason": "explicit traffic split deployment",
                },
            )


class PrimaryDeploymentRegistryUpdateStrategy(RegistryUpdateStrategy):
    def update_registry(
        self, client, config, endpoint, model_to_deploy, deployed_models_details
    ):
        logger.info("Updating registry for primary deployment (new champion).")
        client.publish_primary(
            vertex_ai_model_resource_name=model_to_deploy.resource_name,
            vertex_ai_model_version_id=int(model_to_deploy.version_id),
            deployment_environment=config.model_reference_config.deployment_environment,
            published_model_endpoint=endpoint.resource_name,
            published_model_publish_status="champion",
            published_model_deployment_status={
                "status": "active",
                "deployment_type": "shadow" if config.shadow_mode else "canary",
                "traffic_split_percentage": 100,
                "validation_result": "complete",
                "inference_type": "online-inference",
                "reason": "new champion deployed",
            },
            demoted_model_publish_status="archived",
            demoted_model_deployment_status={
                "status": "demoted",
                "reason": "new champion deployed",
            },
        )


class ChallengerRegistryUpdateStrategy(RegistryUpdateStrategy):
    def update_registry(
        self, client, config, endpoint, model_to_deploy, deployed_models_details
    ):
        logger.info("Updating registry for new challenger model.")
        client.update_status(
            vertex_ai_model_resource_name=model_to_deploy.resource_name,
            vertex_ai_model_version_id=int(model_to_deploy.version_id),
            deployment_environment=config.model_reference_config.deployment_environment,
            deployed_model_endpoint=endpoint.resource_name,
            publish_status="challenger",
            deployment_status={
                "status": "active",
                "deployment_type": "shadow" if config.shadow_mode else "canary",
                "traffic_split_percentage": 0,
                "validation_result": "pending",
                "inference_type": "online-inference",
                "reason": "new challenger deployed",
            },
        )


# --- Deployment Action Strategies ---


class DeploymentActionStrategy(ABC):
    """Abstract base class for strategies determining how to deploy/update a model on an endpoint."""

    @abstractmethod
    def execute_deployment_action(
        self,
        endpoint: aiplatform.Endpoint,
        model_to_deploy: aiplatform.Model,
        traffic_split_dict: Dict[str, int],
        deployed_model_display_name: str,
        machine_type: Optional[str] = None,
        min_replica_count: Optional[int] = None,
        max_replica_count: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Executes the specific deployment or update action on the given endpoint.

        Args:
            endpoint: The AI Platform Endpoint object.
            model_to_deploy: The AI Platform Model object to deploy (may not be used by update strategies).
            traffic_split_dict: The calculated traffic split dictionary.
            deployed_model_display_name: The display name of the deployed model.
            machine_type: (Optional) The type of machine to use for the deployed model.
            min_replica_count: (Optional) The minimum number of replicas to deploy.
            max_replica_count: (Optional) The maximum number of replicas to deploy.
            kwargs: Additional keyword arguments that can be passed to endpoint.deploy or endpoint.update.
        """
        pass


class UpdateExistingModelDeploymentStrategy(DeploymentActionStrategy):
    """
    Strategy for updating an already deployed model's configuration on an endpoint,
    typically by re-assigning its traffic split and/or machine configuration.
    """

    def __init__(self, deployed_model_id: str):
        self._deployed_model_id = deployed_model_id

    def execute_deployment_action(
        self,
        endpoint: aiplatform.Endpoint,
        model_to_deploy: aiplatform.Model,
        traffic_split_dict: Dict[str, int],
        deployed_model_display_name: str,
        machine_type: Optional[str] = None,
        min_replica_count: Optional[int] = None,
        max_replica_count: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        logger.info(
            f"Updating Endpoint: model with ID '{self._deployed_model_id}' already deployed. "
        )

        # AI Platform's endpoint.update expects traffic_split keys to refer to deployed_model_ids.
        # If '0' is present in traffic_split_dict (our placeholder for the target model),
        # we need to map it to the actual deployed_model_id.
        actual_traffic_split = {}
        for k, v in traffic_split_dict.items():
            if k == "0":
                actual_traffic_split[self._deployed_model_id] = v
            else:
                actual_traffic_split[k] = v

        # Occurs if the model is the only model and is already deployed to this endpoint
        if actual_traffic_split == {self._deployed_model_id: 0}:
            raise ValueError(
                "This model is already deployed to this endpoint with 100% traffic."
            )

        update_args = {"traffic_split": actual_traffic_split}
        logger.info(f"Traffic split for update: {actual_traffic_split}")

        # Merge any additional kwargs
        update_args.update(kwargs)

        logger.info(f"Calling endpoint.update with args: {update_args}")
        endpoint.update(**update_args)
        logger.info(f"Endpoint '{endpoint.display_name}' updated successfully.")


class DeployNewModelToEndpointStrategy(DeploymentActionStrategy):
    """
    Strategy for deploying a new model to an endpoint.
    """

    def execute_deployment_action(
        self,
        endpoint: aiplatform.Endpoint,
        model_to_deploy: aiplatform.Model,
        traffic_split_dict: Dict[str, int],
        deployed_model_display_name: str,
        machine_type: Optional[str] = None,
        min_replica_count: Optional[int] = None,
        max_replica_count: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        logger.info(
            f"Deploying new model '{deployed_model_display_name}' "
            f"to Endpoint '{endpoint.display_name}' with traffic split {traffic_split_dict}."
        )

        deploy_args = {
            "model": model_to_deploy,
            "deployed_model_display_name": deployed_model_display_name,
            "traffic_split": traffic_split_dict,
        }

        if machine_type:
            deploy_args["machine_type"] = machine_type
        if min_replica_count is not None:
            deploy_args["min_replica_count"] = min_replica_count
        if max_replica_count is not None:
            deploy_args["max_replica_count"] = max_replica_count

        # Merge any additional kwargs
        deploy_args.update(kwargs)

        # Log args, excluding the bulky model object for cleaner output
        log_args = {k: v for k, v in deploy_args.items() if k != "model"}
        logger.info(f"Calling endpoint.deploy with args: {log_args}")
        endpoint.deploy(**deploy_args)
        logger.info(
            f"Model '{model_to_deploy.display_name}' deployed successfully to endpoint '{endpoint.display_name}'."
        )
