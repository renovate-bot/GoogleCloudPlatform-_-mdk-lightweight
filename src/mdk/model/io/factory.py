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

"""Implements the Factory pattern to select the correct serialization strategy."""

import logging
from typing import Any, Type
from mdk.model.io.base import SerializerStrategy
from mdk.model.io.pickle import PickleSerializer
from mdk.model.io.pytorch import PyTorchSerializer
from mdk.model.io.xgboost import XGBoostSerializer

logger = logging.getLogger(__name__)


class SerializerFactory:
    """
    Implements the Factory Design Pattern to create model serialization providers.
    """

    _load_by_extension = {
        "ubj": XGBoostSerializer,
        "bst": XGBoostSerializer,
        "pt": PyTorchSerializer,
        "pth": PyTorchSerializer,
        "pkl": PickleSerializer,
    }

    @classmethod
    def get_serializer_for_save(
        cls, model: Any, filename: str = None
    ) -> SerializerStrategy:
        """Selects the appropriate serializer based on the model's type."""
        if filename and filename.endswith(".pkl"):
            return PickleSerializer()

        try:
            import xgboost

            if isinstance(model, (xgboost.Booster, xgboost.XGBClassifier)):
                return XGBoostSerializer()
        except ImportError:
            pass

        try:
            import torch

            if isinstance(model, torch.nn.Module):
                return PyTorchSerializer()
        except ImportError:
            pass

        logger.info(
            f"No specific serializer found for model type {type(model)}. "
            "Defaulting to pickle."
        )
        return PickleSerializer()

    @classmethod
    def get_serializer_for_load(
        cls, filename: str, model_class: Type = None
    ) -> SerializerStrategy:
        """Selects the appropriate serializer based on file extension or model class."""
        # If a specific class is given, it takes precedence
        if model_class:
            try:
                import xgboost

                if issubclass(model_class, (xgboost.Booster, xgboost.XGBClassifier)):
                    return XGBoostSerializer()
            except ImportError:
                pass
            try:
                import torch

                if issubclass(model_class, torch.nn.Module):
                    return PyTorchSerializer()
            except ImportError:
                pass

        # Otherwise, infer from file extension
        extension = filename.split(".")[-1].lower()
        serializer_class = cls._load_by_extension.get(extension)
        if serializer_class:
            return serializer_class()

        raise NotImplementedError(
            f"Could not determine a serializer for filename '{filename}' with extension "
            f"'{extension}' and model_class '{model_class}'."
        )
