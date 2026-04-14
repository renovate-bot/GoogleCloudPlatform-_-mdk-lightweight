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

"""Internal facade for the serialization module."""

from typing import Any, Type
from mdk.model.io.factory import SerializerFactory


def save(model: Any, filename: str = None) -> str:
    """
    Selects the correct serializer and saves a model to a file.
    """
    if filename is None:
        # Generate a default filename if not provided
        if "xgboost" in str(type(model)):
            filename = "model.ubj"
        elif "torch" in str(type(model)):
            filename = "model.pt"
        else:
            filename = "model.pkl"

    serializer = SerializerFactory.get_serializer_for_save(model, filename)
    serializer.save(model, filename)
    return filename


def load(filename: str, model_class: Type = None) -> Any:
    """
    Selects the correct serializer and loads a model from a file.
    """
    serializer = SerializerFactory.get_serializer_for_load(filename, model_class)
    return serializer.load(filename, model_class=model_class)
