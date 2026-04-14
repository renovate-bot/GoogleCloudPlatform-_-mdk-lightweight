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

"""Defines the factory for all model registry providers."""

import re
from typing import Optional, Type
from functools import lru_cache

import mdk.config
from mdk.model.registry.models import RegistryAppConfig
from mdk.model.registry.providers.base import ModelRegistryProvider
from mdk.model.registry.strategies import (
    ModelRetrievalStrategy,
    LatestModelStrategy,
    PrimaryModelStrategy,
    SemanticVersionModelStrategy,
    GCSUriStrategy,
    DirectVertexAIModelStrategy,
)
from mdk.model.registry.clients.expanded_model_registry import (
    ExpandedModelRegistryClient,
)


# Cache for dynamically loaded provider classes
@lru_cache(maxsize=None)
def _get_provider_class_from_string(
    provider_module_path: str,
) -> Type[ModelRegistryProvider]:
    """Dynamically imports and returns a provider class."""
    module_name, class_name = provider_module_path.rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


class ProviderFactory:
    """
    Implements the Factory Design Pattern to create model registry providers.
    This makes it easy to add new providers (e.g., MLflow, S3) in the future
    without changing the client code.
    """

    # Store as string for lazy import
    _providers = {
        "vertex": "mdk.model.registry.providers.vertex.VertexAIModelRegistryProvider"
    }

    @classmethod
    def get_provider(
        cls,
        provider_name: str,
        project_id: str,
        region: str,
        expanded_model_registry_endpoint: str,
        app_config_for_upload: Optional[RegistryAppConfig] = None,
        access_token: Optional[str] = None,
    ) -> ModelRegistryProvider:
        """
        Returns an instance of the requested model registry provider.

        Args:
            provider_name: The name of the provider (e.g., "vertex").
            project_id: The Google Cloud project ID.
            region: The Google Cloud region.
            expanded_model_registry_endpoint: The endpoint URL for the Expanded Model Registry service.
            app_config_for_upload: Optional. The full RegistryAppConfig object, specifically needed for
                                   model upload operations that require detailed metadata.
                                   This is required if the provider needs RegistryAppConfig for its operations
                                   and is being instantiated for upload.
            access_token:  Optional. An auth token. If not provided, one will be generated.

        Returns:
            An instance of a concrete ModelRegistryProvider.
        """
        provider_module_path = cls._providers.get(provider_name)
        if not provider_module_path:
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available providers: {list(cls._providers.keys())}"
            )

        provider_cls = _get_provider_class_from_string(provider_module_path)

        # Instantiate the provider with the core GCP parameters and optional RegistryAppConfig
        return provider_cls(
            project_id=project_id,
            region=region,
            expanded_model_registry_endpoint=expanded_model_registry_endpoint,
            app_config_for_upload=app_config_for_upload,
            access_token=access_token,
        )

    class ModelRetrievalStrategyFactory:
        """
        Factory to create the appropriate ModelRetrievalStrategy based on model_reference_config.
        Nested within ProviderFactory to keep all factory logic organized.
        """

        @staticmethod
        def create_strategy(
            model_reference_config: mdk.config.ModelReferenceConfig,
            emr_client: ExpandedModelRegistryClient,
        ) -> ModelRetrievalStrategy:
            """
            Creates and returns the appropriate model retrieval strategy.

            Args:
                model_reference_config: A validated ModelReferenceConfig Pydantic model.
                emr_client: An instance of the ExpandedModelRegistryClient.

            Returns:
                An instance of a concrete ModelRetrievalStrategy.

            Raises:
                ValueError: If no matching strategy is found or required parameters are missing.
            """
            model_inference_reference = model_reference_config.model_inference_reference
            vertex_ai_model_resource_name = (
                model_reference_config.vertex_ai_model_resource_name
            )

            config_dict = model_reference_config.model_dump()

            if model_inference_reference == "latest":
                return LatestModelStrategy(config_dict, emr_client)
            elif model_inference_reference == "primary":
                return PrimaryModelStrategy(config_dict, emr_client)
            elif model_inference_reference and re.fullmatch(
                r"^\d+\.\d+\.\d+$", model_inference_reference
            ):
                return SemanticVersionModelStrategy(config_dict, emr_client)
            elif model_inference_reference and model_inference_reference.startswith(
                "gs://"
            ):
                return GCSUriStrategy(config_dict, emr_client)
            elif vertex_ai_model_resource_name:
                # This covers the case where model_inference_reference is None/empty,
                # but a direct Vertex AI resource name is provided.
                return DirectVertexAIModelStrategy(config_dict, emr_client)
            elif not model_inference_reference and not vertex_ai_model_resource_name:
                raise ValueError(
                    "Either 'model_inference_reference' or 'vertex_ai_model_resource_name' "
                    "must be specified in the model reference configuration."
                )
            else:
                # This 'else' should theoretically be unreachable if Pydantic validation and prior conditions are thorough,
                # but it's a good safeguard for unexpected states.
                raise ValueError(
                    f"Unable to determine model retrieval strategy for model_inference_reference='{model_inference_reference}' "
                    f"and vertex_ai_model_resource_name='{vertex_ai_model_resource_name}'. "
                    "Please check your model reference configuration."
                )
