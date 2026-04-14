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

"""Implements the SerializerStrategy for fallback Pickle serialization."""

import pickle
from typing import Any
from mdk.model.io.base import SerializerStrategy


class PickleSerializer(SerializerStrategy):
    """The default serializer using Python's pickle module (.pkl)."""

    def save(self, model: Any, filename: str) -> None:
        """Saves an object using pickle."""
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    def load(self, filename: str, **kwargs: Any) -> Any:
        """Loads an object using pickle."""
        with open(filename, "rb") as f:
            return pickle.load(f)
