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

"""Defines the abstract base class (interface) for all model registry providers."""

from abc import ABC, abstractmethod
from typing import Any


class ModelRegistryProvider(ABC):
    """
    Abstract Base Class defining the interface for a model registry provider.
    This serves as the 'Strategy' interface in the Strategy Design Pattern.
    """

    @abstractmethod
    def upload_to_expanded_registry(self) -> Any:
        """Uploads a model to the expanded model registry."""
        pass

    @abstractmethod
    def upload(self) -> str:
        """Uploads a model to the specific model registry (e.g. Vertex AI Model Registry)."""
        pass

    @abstractmethod
    def delete_version(self, model_resource: Any):
        """
        Deletes a specific model version, used if two-phase commit fails
        when writing to expanded model registry.
        """
        pass
