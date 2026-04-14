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

"""Strategies for the model.registry module.
These are mostly used externally within the data scientist's code."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

from google.cloud import aiplatform


logger = logging.getLogger(__name__)


class ModelRetrievalStrategy(ABC):
    """Abstract base class for all model retrieval strategies."""

    def __init__(self, model_reference_config: Dict[str, Any], emr_client: Any):
        self._config = model_reference_config
        self._emr_client = emr_client

    @abstractmethod
    def retrieve_model_uri(self) -> str:
        """
        Retrieves the model URI based on the strategy's logic.
        Must return a string representing the model's URI.
        """
        pass

    @abstractmethod
    def retrieve_emr_model_object(
        self,
    ) -> Dict[str, Any]:  # Changed return type to Dict[str, Any]
        """
        Retrieves the full model record from the EMR based on the strategy's logic.
        Returns the entire model response object as a dictionary.
        """
        pass

    @abstractmethod
    def retrieve_vertex_ai_resource_name_and_version(
        self,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Retrieves the Vertex AI model resource name and version ID.
        Returns (resource_name, version_id). If not applicable, returns (None, None).
        """
        pass

    def _get_vertex_ai_model_uri(
        self, vertex_ai_model_resource_name: str, vertex_ai_model_version_id: str
    ) -> str:
        """Helper to construct Vertex AI model URI."""
        vertex_ai_model = aiplatform.Model(
            model_name=vertex_ai_model_resource_name, version=vertex_ai_model_version_id
        )
        return vertex_ai_model.uri


class LatestModelStrategy(ModelRetrievalStrategy):
    """Strategy to retrieve the latest deployed model from EMR."""

    def __init__(self, model_reference_config: Dict[str, Any], emr_client: Any):
        super().__init__(model_reference_config, emr_client)
        self._cached_emr_response: Optional[Dict[str, Any]] = (
            None  # Cache for EMR response
        )

    def _fetch_from_emr(self) -> Dict[str, Any]:
        """Helper to fetch model details from EMR for 'latest' strategy, with caching."""
        if self._cached_emr_response:
            return self._cached_emr_response

        model_name = self._config.get("model_name")
        deployment_environment = self._config.get("deployment_environment")

        if not all([model_name, deployment_environment]):
            raise ValueError(
                f"Missing 'model_name' or 'deployment_environment' in config for 'latest' strategy. Received: {self._config}"
            )

        if self._emr_client is None:
            logger.info(f"Lite Mode active: Retrieving latest model for '{model_name}' directly from Vertex AI.")
            models = aiplatform.Model.list(filter=f'display_name="{model_name}"')
            if not models:
                raise ValueError(f"No model found with display name '{model_name}'")
            
            parent_model = models[0]
            model_registry = aiplatform.models.ModelRegistry(model=parent_model.resource_name)
            versions = model_registry.list_versions()
            if not versions:
                raise ValueError(f"No versions found for model '{model_name}'")
                
            versions.sort(key=lambda v: v.version_create_time, reverse=True)
            latest_version = versions[0]
            
            response_dict = {
                "vertex_ai_model_resource_name": parent_model.resource_name,
                "vertex_ai_model_version_id": latest_version.version_id
            }
        else:
            logger.info(
                f"Retrieving latest model for '{model_name}' in '{deployment_environment}'."
            )
            response = self._emr_client.retrieve_latest(
                model_name=model_name, deployment_environment=deployment_environment
            )
            response_dict = response.json()

        self._cached_emr_response = response_dict  # Cache the full response

        vertex_ai_model_resource_name = response_dict.get(
            "vertex_ai_model_resource_name"
        )
        vertex_ai_model_version_id = response_dict.get("vertex_ai_model_version_id")
        logger.info(
            f"Latest model found: {vertex_ai_model_resource_name}@{vertex_ai_model_version_id}"
        )
        return response_dict

    def retrieve_model_uri(self) -> str:
        response_dict = self._fetch_from_emr()
        vertex_ai_model_resource_name = response_dict.get(
            "vertex_ai_model_resource_name"
        )
        vertex_ai_model_version_id = response_dict.get("vertex_ai_model_version_id")
        if not vertex_ai_model_resource_name:
            raise ValueError("EMR response missing 'vertex_ai_model_resource_name'.")
        return self._get_vertex_ai_model_uri(
            vertex_ai_model_resource_name, vertex_ai_model_version_id
        )

    def retrieve_emr_model_object(self) -> Dict[str, Any]:
        """
        Retrieves the full model record from the EMR based on the 'latest' strategy.
        """
        return self._fetch_from_emr()

    def retrieve_vertex_ai_resource_name_and_version(
        self,
    ) -> Tuple[Optional[str], Optional[str]]:
        response_dict = self._fetch_from_emr()
        return response_dict.get("vertex_ai_model_resource_name"), response_dict.get(
            "vertex_ai_model_version_id"
        )


class PrimaryModelStrategy(ModelRetrievalStrategy):
    """Strategy to retrieve the primary deployed model from EMR."""

    def __init__(self, model_reference_config: Dict[str, Any], emr_client: Any):
        super().__init__(model_reference_config, emr_client)
        self._cached_emr_response: Optional[Dict[str, Any]] = (
            None  # Cache for EMR response
        )

    def _fetch_from_emr(self) -> Dict[str, Any]:
        """Helper to fetch model details from EMR for 'primary' strategy, with caching."""
        if self._cached_emr_response:
            return self._cached_emr_response

        model_name = self._config.get("model_name")
        deployment_environment = self._config.get("deployment_environment")

        if not all([model_name, deployment_environment]):
            raise ValueError(
                f"Missing 'model_name' or 'deployment_environment' in config for 'primary' strategy. Received: {self._config}"
            )

        if self._emr_client is None:
            logger.info(f"Lite Mode active: Retrieving primary (default) model for '{model_name}' directly from Vertex AI.")
            models = aiplatform.Model.list(filter=f'display_name="{model_name}"')
            if not models:
                raise ValueError(f"No model found with display name '{model_name}'")
            
            parent_model = models[0]
            model_registry = aiplatform.models.ModelRegistry(model=parent_model.resource_name)
            try:
                default_version = model_registry.get_model(version="default")
            except Exception as e:
                raise ValueError(f"No 'default' version found for model '{model_name}': {e}")
                
            response_dict = {
                "vertex_ai_model_resource_name": parent_model.resource_name,
                "vertex_ai_model_version_id": default_version.version_id
            }
        else:
            logger.info(
                f"Retrieving primary model for '{model_name}' in '{deployment_environment}'."
            )
            response = self._emr_client.retrieve_primary(
                model_name=model_name, deployment_environment=deployment_environment
            )
            response_dict = response.json()

        self._cached_emr_response = response_dict  # Cache the full response

        vertex_ai_model_resource_name = response_dict.get(
            "vertex_ai_model_resource_name"
        )
        vertex_ai_model_version_id = response_dict.get("vertex_ai_model_version_id")
        logger.info(
            f"Primary model found: {vertex_ai_model_resource_name}@{vertex_ai_model_version_id}"
        )
        return response_dict

    def retrieve_model_uri(self) -> str:
        response_dict = self._fetch_from_emr()
        vertex_ai_model_resource_name = response_dict.get(
            "vertex_ai_model_resource_name"
        )
        vertex_ai_model_version_id = response_dict.get("vertex_ai_model_version_id")
        if not vertex_ai_model_resource_name:
            raise ValueError("EMR response missing 'vertex_ai_model_resource_name'.")
        return self._get_vertex_ai_model_uri(
            vertex_ai_model_resource_name, vertex_ai_model_version_id
        )

    def retrieve_emr_model_object(self) -> Dict[str, Any]:
        """
        Retrieves the full model record from the EMR based on the 'primary' strategy.
        """
        return self._fetch_from_emr()

    def retrieve_vertex_ai_resource_name_and_version(
        self,
    ) -> Tuple[Optional[str], Optional[str]]:
        response_dict = self._fetch_from_emr()
        return response_dict.get("vertex_ai_model_resource_name"), response_dict.get(
            "vertex_ai_model_version_id"
        )


class SemanticVersionModelStrategy(ModelRetrievalStrategy):
    """Strategy to retrieve a model by semantic version from EMR."""

    def __init__(self, model_reference_config: Dict[str, Any], emr_client: Any):
        super().__init__(model_reference_config, emr_client)
        self._cached_emr_response: Optional[Dict[str, Any]] = (
            None  # Cache for EMR response
        )

    def _fetch_from_emr(self) -> Dict[str, Any]:
        """Helper to fetch model details from EMR for 'semantic version' strategy, with caching."""
        if self._cached_emr_response:
            return self._cached_emr_response

        model_name = self._config.get("model_name")
        model_inference_reference = self._config.get("model_inference_reference")

        if not all([model_name, model_inference_reference]):
            raise ValueError(
                f"Missing 'model_name' or 'model_inference_reference' in config for semantic version strategy. Received: {self._config}"
            )

        if self._emr_client is None:
            logger.info(f"Lite Mode active: Retrieving model by semantic version '{model_inference_reference}' for '{model_name}' directly from Vertex AI.")
            models = aiplatform.Model.list(filter=f'display_name="{model_name}"')
            if not models:
                raise ValueError(f"No model found with display name '{model_name}'")
            
            parent_model = models[0]
            formatted_alias = f"v{model_inference_reference.replace('.', '-')}"
            
            model_registry = aiplatform.models.ModelRegistry(model=parent_model.resource_name)
            try:
                target_version = model_registry.get_model(version=formatted_alias)
            except Exception as e:
                raise ValueError(f"No version found with alias '{formatted_alias}' for model '{model_name}': {e}")
                
            response_dict = {
                "vertex_ai_model_resource_name": parent_model.resource_name,
                "vertex_ai_model_version_id": target_version.version_id
            }
        else:
            logger.info(
                f"Retrieving model by semantic version '{model_inference_reference}' for '{model_name}'."
            )
            response = self._emr_client.retrieve_semantic_version(
                model_name=model_name, model_semantic_version=model_inference_reference
            )
            response_dict = response.json()

        self._cached_emr_response = response_dict  # Cache the full response

        vertex_ai_model_resource_name = response_dict.get(
            "vertex_ai_model_resource_name"
        )
        vertex_ai_model_version_id = response_dict.get("vertex_ai_model_version_id")
        logger.info(
            f"Model for semantic version found: {vertex_ai_model_resource_name}@{vertex_ai_model_version_id}"
        )
        return response_dict

    def retrieve_model_uri(self) -> str:
        response_dict = self._fetch_from_emr()
        vertex_ai_model_resource_name = response_dict.get(
            "vertex_ai_model_resource_name"
        )
        vertex_ai_model_version_id = response_dict.get("vertex_ai_model_version_id")
        if not vertex_ai_model_resource_name:
            raise ValueError("EMR response missing 'vertex_ai_model_resource_name'.")
        return self._get_vertex_ai_model_uri(
            vertex_ai_model_resource_name, vertex_ai_model_version_id
        )

    def retrieve_emr_model_object(self) -> Dict[str, Any]:
        """
        Retrieves the full model record from the EMR based on the 'semantic version' strategy.
        """
        return self._fetch_from_emr()

    def retrieve_vertex_ai_resource_name_and_version(
        self,
    ) -> Tuple[Optional[str], Optional[str]]:
        response_dict = self._fetch_from_emr()
        return response_dict.get("vertex_ai_model_resource_name"), response_dict.get(
            "vertex_ai_model_version_id"
        )


class GCSUriStrategy(ModelRetrievalStrategy):
    """Strategy to use a direct GCS URI for the model."""

    def retrieve_model_uri(self) -> str:
        model_uri = self._config.get("model_inference_reference")
        if not model_uri:
            raise ValueError(
                f"model_inference_reference is missing for GCS URI strategy. Received: {self._config}"
            )
        logger.info(f"Using direct GCS URI: {model_uri}")
        return model_uri

    def retrieve_emr_model_object(self) -> Dict[str, Any]:
        """
        GCS URI strategy does not retrieve a full model record from EMR.
        Returns an empty dictionary to indicate no EMR object is available.
        """
        logger.debug(
            "GCSUriStrategy does not retrieve a full model record from EMR; returning empty dict."
        )
        return {}

    def retrieve_vertex_ai_resource_name_and_version(
        self,
    ) -> Tuple[Optional[str], Optional[str]]:
        # A GCS URI directly points to artifacts, not necessarily a registered Vertex AI Model
        # Returning (None, None) explicitly indicates that this strategy doesn't yield VA Model details.
        logger.warning(
            "GCSUriStrategy cannot provide Vertex AI model resource name or version ID directly."
        )
        return None, None


class DirectVertexAIModelStrategy(ModelRetrievalStrategy):
    """
    Strategy to use an explicit Vertex AI model resource name and optionally a version ID
    provided directly in the config, typically as a fallback or direct reference.
    """

    def retrieve_model_uri(self) -> str:
        vertex_ai_model_resource_name, vertex_ai_model_version_id = (
            self.retrieve_vertex_ai_resource_name_and_version()
        )
        if not vertex_ai_model_resource_name:  # Should not happen given retrieve_vertex_ai_resource_name_and_version's logic
            raise ValueError(
                f"DirectVertexAIModelStrategy failed to get model resource name. Config: {self._config}"
            )
        return self._get_vertex_ai_model_uri(
            vertex_ai_model_resource_name, vertex_ai_model_version_id
        )

    def retrieve_emr_model_object(self) -> Dict[str, Any]:
        """
        Direct Vertex AI model strategy does not retrieve a full model record from EMR.
        Returns an empty dictionary to indicate no EMR object is available.
        """
        logger.debug(
            "DirectVertexAIModelStrategy does not retrieve a full model record from EMR; returning empty dict."
        )
        return {}

    def retrieve_vertex_ai_resource_name_and_version(
        self,
    ) -> Tuple[Optional[str], Optional[str]]:
        vertex_ai_model_resource_name = self._config.get(
            "vertex_ai_model_resource_name"
        )
        vertex_ai_model_version_id = self._config.get("vertex_ai_model_version_id")

        if not vertex_ai_model_resource_name:
            raise ValueError(
                f"No explicit 'vertex_ai_model_resource_name' provided in config for direct Vertex AI strategy. Received: {self._config}"
            )

        logger.info(
            f"Using direct Vertex AI model resource: {vertex_ai_model_resource_name} (version: {vertex_ai_model_version_id or 'default'})"
        )
        return vertex_ai_model_resource_name, vertex_ai_model_version_id
