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

"""Defines the factory for all model monitoring providers."""

from typing import Optional

from mdk.model.monitoring.models import MonitoringAppConfig
from mdk.model.monitoring.providers.base import MonitoringProvider
from mdk.model.monitoring.providers.vertex import VertexAIMonitoringProvider


class ProviderFactory:
    """Implements the Factory Design Pattern to create model monitoring
    providers.

    This makes it easy to add new providers in the future without changing the
    client code.
    """

    _providers = {"vertex": VertexAIMonitoringProvider}

    @classmethod
    def get_provider(
        cls,
        provider_name: str,
        config: MonitoringAppConfig,
        access_token: Optional[str] = None,
    ) -> MonitoringProvider:
        """
        Returns an instance of the requested model registry provider.
        """
        provider = cls._providers.get(provider_name)
        if not provider:
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available providers: {list(cls._providers.keys())}"
            )
        return provider(config=config, access_token=access_token)
