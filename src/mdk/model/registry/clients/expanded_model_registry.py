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

"""Client for interacting with the expanded model registry."""

import logging
import requests
from typing import Optional, Dict, Any

from mdk.util.auth import generate_gcp_jwt

logger = logging.getLogger(__name__)


class ExpandedModelRegistryClient:
    """A client for interacting with the Expanded Model Registry service."""

    def __init__(self, base_url: str, access_token: Optional[str] = None):
        """
        Initializes the client.

        Args:
            base_url: The base URL of the Expanded Model Registry endpoint.
            access_token: Optional. An auth token. If not provided, one will be generated.
        """
        if not base_url:
            raise ValueError(
                "base_url for ExpandedModelRegistryClient cannot be empty."
            )
        self.base_url = base_url.rstrip("/")
        self._access_token = access_token

    def _get_auth_header(self) -> Dict[str, str]:
        """Generates the Authorization header for requests."""
        if not self._access_token:
            logger.debug("Generating new GCP JWT for Expanded Model Registry.")
            self._access_token = generate_gcp_jwt(audience=self.base_url)

        return {"Authorization": f"Bearer {self._access_token}"}

    def _post(self, route: str, payload: Dict[str, Any]) -> requests.Response:
        """Helper method to perform a POST request."""
        url = f"{self.base_url}/{route}"
        headers = {"Content-Type": "application/json", **self._get_auth_header()}
        logger.info(f"Sending POST to {url} with payload: {payload}")
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            logger.info(f"Request to {url} successful. Status: {response.status_code}")
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"API request to {url} failed: {e}", exc_info=True)
            raise

    def _get(self, route: str) -> requests.Response:
        """Helper method to perform a GET request."""
        url = f"{self.base_url}/{route}"
        headers = {"Content-Type": "application/json", **self._get_auth_header()}
        logger.info(f"Sending GET to {url}")
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            logger.info(f"Request to {url} successful. Status: {response.status_code}")
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"API request to {url} failed: {e}", exc_info=True)
            raise

    def create_model(self, **kwargs) -> requests.Response:
        """Creates a new model record in the expanded model registry."""
        return self._post("models", kwargs)

    def publish_primary(self, **kwargs) -> requests.Response:
        """Publishes a new primary (champion) model."""
        return self._post("deployments/publish_primary", kwargs)

    def update_status(self, **kwargs) -> requests.Response:
        """Updates the deployment and publish status of a model."""
        return self._post("deployments/update_status", kwargs)

    def rollback_primary(self, **kwargs) -> requests.Response:
        """Rolls back to a previous primary model."""
        if not (
            kwargs.get("vertex_ai_model_resource_name") or kwargs.get("model_name")
        ):
            raise ValueError(
                "Either 'vertex_ai_model_resource_name' or 'model_name' must be provided for rollback."
            )
        return self._post("deployments/rollback_primary", kwargs)

    def retrieve_model_by_vertex_version(self, **kwargs) -> requests.Response:
        """Retrieves a model by vertex_ai_model_resource_name and vertex_ai_model_version_id."""
        vertex_ai_model_resource_name = kwargs.get("vertex_ai_model_resource_name")
        vertex_ai_model_version_id = kwargs.get("vertex_ai_model_version_id")
        if not (vertex_ai_model_resource_name and vertex_ai_model_version_id):
            raise ValueError(
                "Both 'vertex_ai_model_resource_name' and 'vertex_ai_model_version_id' must be provided for retrieving by vertex version."
            )
        return self._get(
            f"models/{vertex_ai_model_resource_name}/vertex_ai_model_version_id/{vertex_ai_model_version_id}"
        )

    def retrieve_primary(self, **kwargs) -> requests.Response:
        """Retrieves a primary model."""
        deployment_environment = kwargs.get("deployment_environment")
        model_name = kwargs.get("model_name")
        if not (deployment_environment or model_name):
            raise ValueError(
                "Both 'deployment_environment' or 'model_name' must be provided for retrieving primary."
            )
        return self._get(f"models/primary/{model_name}/{deployment_environment}")

    def retrieve_latest(self, **kwargs) -> requests.Response:
        """Retrieves a the latest trained model."""
        deployment_environment = kwargs.get("deployment_environment")
        model_name = kwargs.get("model_name")
        if not (deployment_environment or model_name):
            raise ValueError(
                "Both 'deployment_environment' or 'model_name' must be provided for retrieving primary."
            )
        return self._get(f"models/latest/{model_name}/{deployment_environment}")

    def retrieve_semantic_version(self, **kwargs) -> requests.Response:
        """Retrieves a model by model_semantic_version."""
        model_semantic_version = kwargs.get("model_semantic_version")
        model_name = kwargs.get("model_name")
        if not (model_semantic_version or model_name):
            raise ValueError(
                "Both 'model_semantic_version' or 'model_name' must be provided for retrieving by model_semantic_version."
            )
        return self._get(
            f"models/{model_name}/model_semantic_version/{model_semantic_version}"
        )
