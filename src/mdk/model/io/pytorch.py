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

"""Implements the serialization strategy for PyTorch models."""

from typing import Any, Optional, Type
from mdk.model.io.base import SerializerStrategy


class PyTorchSerializer(SerializerStrategy):
    """Concrete serializer for PyTorch models."""

    def save(self, model: Any, filename: Optional[str] = None) -> None:
        """Saves a PyTorch model to a .pt file."""
        try:
            import torch
        except ImportError:
            raise ImportError(
                "torch is not installed. Please install it to save PyTorch models."
            )

        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Model must be a torch.nn.Module, not {type(model)}")

        torch.save(model, filename)

    def load(self, filename: str, model_class: Optional[Type] = None) -> Any:
        """Loads a PyTorch model from a file."""
        try:
            import torch
        except ImportError:
            raise ImportError(
                "torch is not installed. Please install it to load PyTorch models."
            )

        return torch.load(filename)
