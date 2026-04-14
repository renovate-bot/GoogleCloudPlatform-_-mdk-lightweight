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

"""Implements the SerializerStrategy for XGBoost models."""

from typing import Any
from mdk.model.io.base import SerializerStrategy


class XGBoostSerializer(SerializerStrategy):
    """A serializer for XGBoost models (.ubj)."""

    def save(self, model: Any, filename: str) -> None:
        """Saves an XGBoost model."""
        model.save_model(filename)

    def load(self, filename: str, **kwargs: Any) -> Any:
        """
        Loads an XGBoost model.

        Handles the special case where the user wants a specific class like
        XGBClassifier instead of the default Booster.
        """
        # Don't do the import unless someone is using XGBoost.
        import xgboost

        model_class = kwargs.get("model_class")

        if model_class and issubclass(model_class, xgboost.sklearn.XGBClassifier):
            # Instantiate the class and then load the model into it
            model = model_class()
            model.load_model(filename)
            return model

        # Default to loading a Booster object
        model = xgboost.Booster()
        model.load_model(filename)
        return model
