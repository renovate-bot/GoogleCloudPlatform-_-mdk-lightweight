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

import os
import json
import logging
import sys
import requests
from typing import Dict, Any, Optional

from google.cloud import aiplatform

try:
    from mdk.model.registry import (
        ExpandedModelRegistryClient,
        get_vertex_ai_model_resource_name_and_version_for_inference,
    )
    from mdk.model.deployment import strategies
except ImportError as e:
    logging.error(f"Failed to import mdk components: {e}")
    logging.error(
        "Please ensure the 'mdk' package is available in the Python environment and its structure is correct."
    )
    logging.error(f"Current PYTHONPATH: {sys.path}")
    sys.exit(1)

# Configure logging for the script
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _get_models_from_endpoint(endpoint: aiplatform.Endpoint) -> Dict[str, Any]:
    """Helper to get details of models currently on the endpoint."""
    models_dict = {}
    for model in endpoint.list_models():
        models_dict[model.id] = {
            "vertex_ai_model_resource_name": model.model,
            "vertex_ai_model_version_id": model.model_version_id,
        }
    return models_dict


def run_deployment_operations():
    """
    Reads parsed deployment configurations from environment variable and executes
    the corresponding model registry API calls using mdk.model.registry.ExpandedModelRegistryClient.
    """
    all_parsed_configs_json = os.environ.get("ALL_PARSED_CONFIGS_JSON")

    if not all_parsed_configs_json:
        logger.warning(
            "No parsed configurations found in ALL_PARSED_CONFIGS_JSON. Exiting."
        )
        # Exit successfully if no configs were found, as the GHA job handles this.
        sys.exit(0)

    try:
        all_parsed_configs = json.loads(all_parsed_configs_json)
    except json.JSONDecodeError as e:
        logger.error(
            f"Failed to parse ALL_PARSED_CONFIGS_JSON: {e}. Raw JSON: {all_parsed_configs_json[:500]}..."
        )  # Log beginning of malformed JSON
        sys.exit(1)

    if not isinstance(all_parsed_configs, list):
        logger.error("Expected ALL_PARSED_CONFIGS_JSON to be a list of config objects.")
        sys.exit(1)

    if not all_parsed_configs:
        logger.info(
            "ALL_PARSED_CONFIGS_JSON was an empty list. No operations to run. Exiting."
        )
        sys.exit(0)

    logger.info(f"Received {len(all_parsed_configs)} config files to process.")

    # Iterate through each config file's parsed content
    for i, config_obj in enumerate(all_parsed_configs):
        logger.info(f"\n--- Processing Config File #{i + 1} ---")

        api_settings = config_obj.get("api_settings", {})
        expanded_model_registry_endpoint = api_settings.get(
            "expanded_model_registry_endpoint"
        )
        gcp_project_id = api_settings.get("project_id")
        gcp_region = api_settings.get("region")
        global_deployment_environment = api_settings.get("deployment_environment")
        operations = config_obj.get("operations", [])

        if not all(
            [
                expanded_model_registry_endpoint,
                gcp_project_id,
                gcp_region,
                global_deployment_environment,
            ]
        ):
            logger.error(
                f"Config object {i + 1} missing one or more essential 'api_settings' fields (expanded_model_registry_endpoint, project_id, region, deployment_environment). Skipping this config."
            )
            sys.exit(1)  # Fail if a config is malformed and missing critical info

        if not operations:
            logger.info(
                f"No 'operations' defined in config #{i + 1} for API Base URL: {expanded_model_registry_endpoint}. Skipping this config."
            )
            continue  # Move to the next config file if this one has no operations

        access_token = os.environ.get(
            "GCP_ID_TOKEN"
        )  # Assumes token is always available
        registry_client = ExpandedModelRegistryClient(
            base_url=expanded_model_registry_endpoint, access_token=access_token
        )

        logger.info(
            f"Processing {len(operations)} operations for API Base URL: {expanded_model_registry_endpoint}"
        )

        # Iterate through each operation within the current config file
        for j, operation in enumerate(operations):
            operation_name = operation.get(
                "name", f"Unnamed Operation #{j + 1} in config #{i + 1}"
            )
            operation_type = operation.get("type")
            target = operation.get("target", {})
            champion_status = operation.get("champion_status", {})
            demoted_status = operation.get("demoted_status", {})
            updated_status = operation.get("updated_status", {})

            logger.info(
                f"  [{j + 1}/{len(operations)}] Attempting operation: '{operation_name}' (Type: {operation_type})"
            )

            # Construct ModelReferenceConfig data from the operation's 'target' section
            # and the global deployment_environment
            model_reference_config_data: Dict[str, Any] = {
                "model_name": target.get("model_name"),
                "deployment_environment": global_deployment_environment,
                "model_inference_reference": target.get("model_inference_reference"),
                "vertex_ai_model_resource_name": target.get(
                    "vertex_ai_model_resource_name"
                ),
                "vertex_ai_model_version_id": target.get("vertex_ai_model_version_id"),
            }
            # Filter out None values to avoid issues with Pydantic validation if fields are optional
            model_reference_config_data = {
                k: v for k, v in model_reference_config_data.items() if v is not None
            }

            resolved_vertex_ai_model_resource_name: Optional[str] = None
            resolved_vertex_ai_model_version_id: Optional[str] = None
            resolved_model_name_for_display: Optional[str] = target.get(
                "model_name"
            )  # Fallback display name

            try:
                # Resolve the actual Vertex AI model resource name and version ID using the utility
                (
                    resolved_vertex_ai_model_resource_name,
                    resolved_vertex_ai_model_version_id,
                ) = get_vertex_ai_model_resource_name_and_version_for_inference(
                    model_reference_config_data=model_reference_config_data,
                    gcp_project_id=gcp_project_id,
                    gcp_region=gcp_region,
                    expanded_model_registry_endpoint=expanded_model_registry_endpoint,
                    access_token=access_token,
                )
                if resolved_vertex_ai_model_resource_name:
                    logger.info(
                        f"  Resolved target model to Vertex AI resource: '{resolved_vertex_ai_model_resource_name}' version: '{resolved_vertex_ai_model_version_id}'"
                    )
                else:
                    logger.info(
                        "  Target model reference did not resolve to a specific Vertex AI resource (might be an EMR-managed reference like 'latest' or 'primary')."
                    )

            except Exception as e:
                logger.error(
                    f"  Failed to resolve model reference for operation '{operation_name}': {e}",
                    exc_info=True,
                )
                sys.exit(1)

            try:
                response = None
                if operation_type == "publish_primary":
                    if not resolved_vertex_ai_model_resource_name:
                        raise ValueError(
                            f"For 'publish_primary' operation, model reference must resolve to a specific Vertex AI model. Could not resolve from target: {target}"
                        )

                    response = registry_client.publish_primary(
                        vertex_ai_model_resource_name=resolved_vertex_ai_model_resource_name,
                        vertex_ai_model_version_id=resolved_vertex_ai_model_version_id,
                        deployment_environment=global_deployment_environment,
                        published_model_deployment_status=champion_status.get(
                            "deployment_status"
                        ),
                        published_model_publish_status=champion_status.get(
                            "publish_status", "champion"
                        ),
                        demoted_model_deployment_status=demoted_status.get(
                            "deployment_status"
                        ),
                        demoted_model_publish_status=demoted_status.get(
                            "publish_status", "archived"
                        ),
                    )
                elif operation_type == "rollback_primary":
                    # EMR's rollback_primary can take either specific Vertex AI details or just the logical model_name
                    # We pass resolved Vertex AI details if available, otherwise rely on the logical model_name for EMR lookup.
                    payload: Dict[str, Any] = {
                        "deployment_environment": global_deployment_environment,
                        "published_model_deployment_status": champion_status.get(
                            "deployment_status"
                        ),
                        "published_model_publish_status": champion_status.get(
                            "publish_status", "champion"
                        ),
                        "demoted_model_deployment_status": demoted_status.get(
                            "deployment_status"
                        ),
                        "demoted_model_publish_status": demoted_status.get(
                            "publish_status", "challenger"
                        ),
                    }
                    if (
                        resolved_vertex_ai_model_resource_name
                        and target.get("model_inference_reference") != "latest"
                    ):
                        payload["vertex_ai_model_resource_name"] = (
                            resolved_vertex_ai_model_resource_name
                        )
                        payload["vertex_ai_model_version_id"] = (
                            resolved_vertex_ai_model_version_id
                        )
                    elif (
                        target.get("model_name")
                        and target.get("model_inference_reference") == "latest"
                    ):
                        # If no specific Vertex AI model, EMR client will use logical model_name to find previous primary
                        payload["model_name"] = target.get("model_name")
                    else:
                        raise ValueError(
                            "For 'rollback_primary' operation, either a specific Vertex AI model or a logical 'model_name' must be provided in target."
                        )

                    response = registry_client.rollback_primary(**payload)

                elif operation_type == "update_deployment_status":
                    if not resolved_vertex_ai_model_resource_name:
                        raise ValueError(
                            f"For 'update_deployment_status' operation, model reference must resolve to a specific Vertex AI model. Could not resolve from target: {target}"
                        )

                    response = registry_client.update_status(
                        vertex_ai_model_resource_name=resolved_vertex_ai_model_resource_name,
                        vertex_ai_model_version_id=resolved_vertex_ai_model_version_id,
                        deployment_environment=global_deployment_environment,
                        deployment_status=updated_status.get("deployment_status"),
                        publish_status=updated_status.get("publish_status"),
                        challenger_model_ids=updated_status.get("challenger_model_ids"),
                    )
                else:
                    logger.warning(
                        f"  Unknown operation type '{operation_type}' for operation '{operation_name}'. Skipping."
                    )
                    continue  # Skip to the next operation

                # Check response status from the requests library
                if response:
                    response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
                    logger.info(
                        f"  Operation '{operation_name}' successful. Status Code: {response.status_code}"
                    )
                    # Log response body for debugging if needed, but be mindful of sensitive data
                    logger.debug(
                        f"  Response Body: {json.dumps(response.json(), indent=2)}"
                    )
                else:
                    logger.warning(
                        f"  Operation '{operation_name}' completed without generating a response (might be an internal script logic issue)."
                    )

            except requests.exceptions.RequestException as e:
                logger.error(
                    f"  Network/API Request Error for operation '{operation_name}': {e}"
                )
                if hasattr(e, "response") and e.response is not None:
                    logger.error(f"  API Error Status Code: {e.response.status_code}")
                    logger.error(f"  API Error Details: {e.response.text}")
                sys.exit(1)  # Fail the workflow if any operation's API call fails
            except (
                ValueError
            ) as e:  # Catch validation errors from our Python client functions
                logger.error(
                    f"  Validation Error for operation '{operation_name}': {e}"
                )
                sys.exit(1)
            except Exception as e:
                logger.error(
                    f"  An unexpected error occurred during operation '{operation_name}': {e}",
                    exc_info=True,
                )
                sys.exit(1)

            # Update Vertex AI endpoints only for online-inference types that promote a primary model
            champion_deployment_status = champion_status.get("deployment_status", {})
            if champion_deployment_status.get(
                "inference_type"
            ) == "online-inference" and (
                operation_type == "publish_primary"
                or operation_type == "rollback_primary"
            ):
                # Ensure Vertex AI SDK is initialized
                aiplatform.init(project=gcp_project_id, location=gcp_region)

                response_dict = response.json()
                primary_model_details = None
                for model_details in response_dict:
                    if model_details.get("is_primary_deployment"):
                        primary_model_details = model_details
                        break

                if not primary_model_details:
                    logger.error(
                        f"No primary model details found in EMR response after '{operation_type}'. Cannot update Vertex AI endpoint."
                    )
                    sys.exit(1)

                primary_model_vertex_resource_name = primary_model_details.get(
                    "vertex_ai_model_resource_name"
                )
                primary_model_vertex_version_id = primary_model_details.get(
                    "vertex_ai_model_version_id"
                )
                deployment_endpoint_id = primary_model_details.get(
                    "deployment_endpoint_id"
                )
                model_display_name = primary_model_details.get(
                    "model_name", resolved_model_name_for_display
                )

                if (
                    not primary_model_vertex_resource_name
                    or not primary_model_vertex_version_id
                ):
                    logger.error(
                        "EMR response for primary model is missing 'vertex_ai_model_resource_name' or 'vertex_ai_model_version_id'. Cannot update Vertex AI endpoint."
                    )
                    sys.exit(1)
                elif not deployment_endpoint_id:
                    logger.error(
                        f"No deployment_endpoint_id registered for model {primary_model_vertex_resource_name}@{primary_model_vertex_version_id}. Update EMR record to include the deployment_endpoint_id."
                    )
                    sys.exit(1)

                try:
                    endpoint = aiplatform.Endpoint(deployment_endpoint_id)
                except Exception as e:
                    logger.error(
                        f"There was an error loading endpoint with id: {deployment_endpoint_id}. Please be sure this endpoint exists. {e}"
                    )
                    sys.exit(1)

                current_traffic_split = endpoint.traffic_split or {}
                traffic_split_dict = {key: 0 for key in current_traffic_split}
                traffic_split_dict["0"] = 100  # New primary model gets 100% traffic

                model_to_deploy = aiplatform.Model(
                    model_name=primary_model_vertex_resource_name,
                    version=primary_model_vertex_version_id,
                )
                logger.info(
                    f"Setting {primary_model_vertex_resource_name}@{primary_model_vertex_version_id} as the primary model on endpoint with id {deployment_endpoint_id}"
                )

                deployed_models_details = _get_models_from_endpoint(endpoint=endpoint)
                logger.info(
                    f"Updating or deploying model to endpoint '{endpoint.display_name}' with traffic: {traffic_split_dict}"
                )
                deployment_strategy: strategies.DeploymentActionStrategy = (
                    strategies.DeployNewModelToEndpointStrategy()
                )

                # Check if the exact model version is already deployed
                found_deployed_model_id = None
                for deployed_model_id, details in deployed_models_details.items():
                    if details[
                        "vertex_ai_model_resource_name"
                    ] == primary_model_vertex_resource_name and str(
                        details["vertex_ai_model_version_id"]
                    ) == str(primary_model_vertex_version_id):
                        found_deployed_model_id = deployed_model_id
                        break

                if found_deployed_model_id:
                    deployment_strategy = (
                        strategies.UpdateExistingModelDeploymentStrategy(
                            deployed_model_id=found_deployed_model_id
                        )
                    )

                # Execute the chosen strategy. machine_type, min/max_replica_count are not in YAML, so pass as None.
                deployment_strategy.execute_deployment_action(
                    endpoint=endpoint,
                    model_to_deploy=model_to_deploy,
                    traffic_split_dict=traffic_split_dict,
                    deployed_model_display_name=model_display_name,
                    # machine_type=None, # These are not configured in the provided YAML
                    # min_replica_count=None,
                    # max_replica_count=None,
                )
                logger.info(
                    f"Successfully updated or deployed model to Vertex AI endpoint: {endpoint.resource_name}"
                )

    logger.info(
        "\nAll deployment operations across all config files completed successfully."
    )


if __name__ == "__main__":
    run_deployment_operations()
