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

"""Defines the factory for all model deployment providers."""

from typing import Optional

from mdk.model.deployment.models import DeploymentAppConfig
from mdk.model.deployment.providers.base import DeploymentProvider
from mdk.model.deployment.providers.vertex import VertexAIDeploymentProvider


class ProviderFactory:
    """
    Implements the Factory Design Pattern for creating various deployment services.
    This centralized approach allows for easy addition of new providers (like EKS or Azure)
    by simply updating the _providers dictionary without changing client code.
    """

    _providers = {"vertex": VertexAIDeploymentProvider}

    @classmethod
    def get_provider(
        cls,
        provider_name: str,
        config: DeploymentAppConfig,
        access_token: Optional[str] = None,
    ) -> DeploymentProvider:
        """
        Returns an instance of the requested model deployment provider.
        """
        provider = cls._providers.get(provider_name)
        if not provider:
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available providers: {list(cls._providers.keys())}"
            )
        if provider_name == "gke":
            raise NotImplementedError("GKE deployment provider is not yet implemented.")

        return provider(config=config, access_token=access_token)
