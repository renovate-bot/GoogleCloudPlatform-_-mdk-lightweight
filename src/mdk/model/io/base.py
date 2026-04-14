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

"""Defines the abstract base class (interface) for all serializer strategies."""

from abc import ABC, abstractmethod
from typing import Any


class SerializerStrategy(ABC):
    """
    Abstract Base Class for a serialization strategy. This defines the
    contract that all concrete serializers (e.g., for PyTorch, XGBoost)
    must follow.
    """

    @abstractmethod
    def save(self, model: Any, filename: str) -> None:
        """Saves a model object to a file."""
        pass

    @abstractmethod
    def load(self, filename: str, **kwargs: Any) -> Any:
        """Loads a model object from a file."""
        pass
