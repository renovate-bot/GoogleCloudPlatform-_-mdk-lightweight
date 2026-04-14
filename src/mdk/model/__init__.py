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

"""Public facade for the MDK model module."""

import logging
from typing import Any, Type, Optional

# 1. Import the internal implementations under private names.
from mdk.model.io import load as load_model, save as save_model

# Configure a logger for the library.
logger = logging.getLogger(__name__)


# 2. Define the high-level public functions.
def save(model: Any, filename: Optional[str] = None) -> str:
    """
    Writes a model object to a file, automatically selecting the best format.

    Supported formats are inferred from the model type:
    - XGBoost models -> .ubj
    - PyTorch models -> .pt
    - All others -> .pkl (Pickle)

    Args:
        model (Any): The model object to save.
        filename (str, optional): The path to save the file to. If not
            provided, a default (e.g., 'model.ubj') will be used.

    Returns:
        str: The filename to which the model was written.
    """
    return save_model(model, filename)


def load(filename: str, model_class: Optional[Type] = None) -> Any:
    """
    Reads a model object from a file, inferring the format from the extension.

    Supported extensions: .ubj (XGBoost), .pt/.pth (PyTorch), .pkl (Pickle).

    Args:
        filename (str): The path of the file to load.
        model_class (Type, optional): Explicitly specify the model class to load,
            especially useful for XGBoost (e.g., xgboost.XGBClassifier).

    Returns:
        Any: The loaded model object.
    """
    return load_model(filename, model_class)


# 3. Define what symbols are exposed when 'from mdk.model import *' is used.
__all__ = [
    "load",
    "save",
]

# Set the version of your package.
__version__ = "0.1.0"
