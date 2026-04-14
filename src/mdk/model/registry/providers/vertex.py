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

"""Implements the ModelRegistryProvider for Google Cloud Vertex AI."""

import logging
from typing import Dict, Any, Optional, Tuple
from google.cloud import aiplatform
import os

import mdk.config
from mdk.model.registry.models import RegistryAppConfig
from mdk.model.registry.providers.base import ModelRegistryProvider
from mdk.model.registry.clients.expanded_model_registry import (
    ExpandedModelRegistryClient,
)


logger = logging.getLogger(__name__)


class VertexAIModelRegistryProvider(ModelRegistryProvider):
    """
    A concrete implementation of ModelRegistryProvider for Google Cloud Vertex AI.
    This provider manages uploading models to and deleting specific versions from
    the Vertex AI Model Registry, and also retrieves model URIs for inference.
    """

    def __init__(
        self,
        project_id: str,
        region: str,
        expanded_model_registry_endpoint: Optional[str] = None,
        app_config_for_upload: Optional[RegistryAppConfig] = None,
        access_token: Optional[str] = None,
    ):
        """
        Initializes the VertexAIModelRegistryProvider with the necessary GCP configuration.

        Args:
            project_id: The Google Cloud project ID.
            region: The Google Cloud region.
            expanded_model_registry_endpoint: The endpoint URL for the Expanded Model Registry service.
            app_config_for_upload: Optional. The full RegistryAppConfig object, specifically needed for
                                   model upload operations that require detailed metadata.
            access_token:  Optional. An auth token. If not provided, one will be generated.
        """
        self.project_id = project_id
        self.region = region
        self.expanded_model_registry_endpoint = expanded_model_registry_endpoint
        self.access_token = access_token
        is_lite = os.environ.get("MDK_LITE_MODE") == "True"

        if not self.expanded_model_registry_endpoint or is_lite:
            logger.info(
                "Disabling Expanded Model Registry Client due to Lite Mode or empty endpoint."
            )
            self.registry_client = None
        else:
            self.registry_client = ExpandedModelRegistryClient(
                base_url=self.expanded_model_registry_endpoint,
                access_token=self.access_token,
            )
        # Initialize Vertex AI SDK
        logger.info(
            f"Initializing Vertex AI for project '{self.project_id}' in '{self.region}'"
        )
        aiplatform.init(project=self.project_id, location=self.region)

        # Store app_config only if provided, for upload-specific operations
        self._app_config_for_upload = app_config_for_upload
        if self._app_config_for_upload and not isinstance(
            self._app_config_for_upload, RegistryAppConfig
        ):
            raise TypeError(
                "app_config_for_upload must be an instance of RegistryAppConfig if provided."
            )

    def _get_model_retrieval_strategy(
        self, model_reference_config_data: Dict[str, Any]
    ):
        """Helper to get the appropriate model retrieval strategy."""
        # Import locally to avoid circular dependency.
        from mdk.model.registry.providers.factory import ProviderFactory

        model_reference_config = mdk.config.ModelReferenceConfig.model_validate(
            model_reference_config_data
        )
        return ProviderFactory.ModelRetrievalStrategyFactory.create_strategy(
            model_reference_config=model_reference_config,
            emr_client=self.registry_client,
        )

    def get_model_uri_for_inference(
        self, model_reference_config_data: Dict[str, Any]
    ) -> str:
        """
        Retrieves the model URI for inference based on the provided configuration.
        """
        strategy = self._get_model_retrieval_strategy(model_reference_config_data)
        model_uri = strategy.retrieve_model_uri()
        logger.info(f"Model URI for inference resolved to: {model_uri}")
        return model_uri

    def get_emr_model_object(
        self, model_reference_config_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Retrieves the full model record from the expanded model registry based on the provided configuration.
        """
        strategy = self._get_model_retrieval_strategy(model_reference_config_data)
        model_response_dict = strategy.retrieve_emr_model_object()
        logger.info(f"Model response dict for inference: {model_response_dict}")
        return model_response_dict

    def get_vertex_ai_model_object(
        self, model_reference_config_data: Dict[str, Any]
    ) -> Optional[aiplatform.Model]:
        """
        Retrieves the Vertex AI model object (aiplatform.Model) based on the provided configuration.

        Args:
            model_reference_config_data: A dictionary containing model reference configuration details.

        Returns:
            An instance of aiplatform.Model if the strategy resolves to a registered Vertex AI model,
            otherwise None (e.g., for direct GCS URIs).
        """
        strategy = self._get_model_retrieval_strategy(model_reference_config_data)
        vertex_ai_model_resource_name, vertex_ai_model_version_id = (
            strategy.retrieve_vertex_ai_resource_name_and_version()
        )

        if vertex_ai_model_resource_name:
            logger.info(
                f"Found Vertex AI model details: resource='{vertex_ai_model_resource_name}', version='{vertex_ai_model_version_id}'"
            )
            return aiplatform.Model(
                model_name=vertex_ai_model_resource_name,
                version=vertex_ai_model_version_id,
            )
        else:
            logger.info(
                "Model reference configuration did not resolve to a registered Vertex AI model object."
            )
            return None

    def get_vertex_ai_model_resource_name_and_version(
        self, model_reference_config_data: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Retrieves the Vertex AI model resource name and version ID based on the provided configuration.

        Args:
            model_reference_config_data: A dictionary containing model reference configuration details.

        Returns:
            A tuple (resource_name, version_id) if the strategy resolves to a registered Vertex AI model.
            Otherwise, (None, None) (e.g., for direct GCS URIs).
        """
        strategy = self._get_model_retrieval_strategy(model_reference_config_data)
        resource_name, version_id = (
            strategy.retrieve_vertex_ai_resource_name_and_version()
        )
        if resource_name:
            logger.info(
                f"Found Vertex AI model details: resource='{resource_name}', version='{version_id}'"
            )
        else:
            logger.info(
                "Model reference configuration did not resolve to a registered Vertex AI model details."
            )
        return resource_name, version_id

    def upload_to_expanded_registry(
        self,
        vertex_model: aiplatform.Model,
        performance_metrics: dict | None,
        pipeline_job_id: str | None,
    ):
        """Handles the logic for uploading metadata to the expanded model registry."""
        if not self.registry_client:
            logger.info("Skipping Expanded Model Registry upload.")
            return None
        if not self._app_config_for_upload:
            raise RuntimeError(
                "RegistryAppConfig not provided to VertexAIModelRegistryProvider for upload operation."
            )

        logger.info("Preparing payload for the expanded model registry...")
        ext_cfg = self._app_config_for_upload.metadata.expanded_mr
        parameters = {
            "model_name": self._app_config_for_upload.metadata.vertex_mr.model_name,
            "model_gcp_project_id": self._app_config_for_upload.gcp.project_id,
            "vertex_ai_model_resource_name": vertex_model.resource_name,
            "vertex_ai_model_version_id": vertex_model.version_id,
            "gcs_artifact_path": vertex_model.uri,
            "vertex_ai_pipeline_job_run_id": pipeline_job_id,
            "performance_metrics_summary": performance_metrics,
            **ext_cfg.model_dump(exclude_none=True),
        }
        logger.info("Creating model in Expanded Model Registry.")
        return self.registry_client.create_model(**parameters)

    def upload(
        self,
        artifact_folder_uri: str,
        performance_metrics: dict | None,
        pipeline_job_id: str | None,
    ) -> str:
        """
        Uploads a model to the Vertex AI Model Registry.

        Args:
            artifact_folder_uri: The URI of the model to upload.
            performance_metrics: Performance metrics associated with the trained model.
            pipeline_job_id: Vertex AI pipeline job ID associated with the trained model.

        Returns:
            The resource_name of the newly uploaded model version.
        """
        if not self._app_config_for_upload:
            raise RuntimeError(
                "RegistryAppConfig not provided to VertexAIModelRegistryProvider for upload operation."
            )

        cfg = self._app_config_for_upload.metadata.vertex_mr

        # Determine if the model is new or a new version of an existing model
        models = aiplatform.Model.list(filter=f'display_name="{cfg.model_name}"')
        parent_model_resource_name = models[0].resource_name if models else None
        model_id_for_new_model = (
            None
            if parent_model_resource_name
            else cfg.model_name.lower().replace("_", "-")
        )
        logger.info(
            f"Uploading model '{cfg.model_name}' to Vertex AI Model Registry with artifact uri '{artifact_folder_uri}'..."
        )

        try:
            uploaded_model = aiplatform.Model.upload(
                display_name=cfg.model_name,
                artifact_uri=artifact_folder_uri,
                serving_container_image_uri=cfg.serving_container_image_uri,
                serving_container_ports=cfg.serving_container_ports,
                parent_model=parent_model_resource_name,
                model_id=model_id_for_new_model,
                version_aliases=cfg.model_version_aliases,
                version_description=cfg.model_version_description,
                labels=cfg.model_labels,
                is_default_version=True,
            )
            display_name = uploaded_model.display_name
            version_id = uploaded_model.version_id
            model_resource_name = uploaded_model.resource_name
            logger.info(
                f"Successfully uploaded model '{display_name}' "
                f"(resource name: {model_resource_name}, version: {version_id}) to Vertex AI."
            )

        except Exception as e:
            logger.error(
                f"Failed to upload model '{cfg.model_name}'. See logs for details. {e}",
                exc_info=True,
            )
            raise

        try:
            response = self.upload_to_expanded_registry(
                vertex_model=uploaded_model,
                performance_metrics=performance_metrics,
                pipeline_job_id=pipeline_job_id,
            )
            if response:
                json_response = response.json()
                logger.info(
                    f"Successfully wrote model to expanded model registry. Service responded with '{json_response}'."
                )
            else:
                json_response = {}
                logger.info("Skipped writing to Expanded Model Registry.")

        except Exception as e:
            # To make sure the expanded model registry stays in sync, delete the Vertex AI Model.
            logger.error(
                f"An error occurred when uploading the model to expanded model registry: {e}"
            )
            if uploaded_model:
                try:
                    logger.info(
                        f"Attempting to delete uploaded model '{model_resource_name}' "
                    )
                    self.delete_version(uploaded_model)
                except Exception as rollback_e:
                    logger.critical(
                        f"CRITICAL: Rollback failed: {rollback_e}", exc_info=True
                    )
            raise

        # Update Vertex AI model resource aliases with the generated model_semantic_version
        model_semantic_version = json_response.get("model_semantic_version", None)
        if model_semantic_version:
            model_registry = aiplatform.models.ModelRegistry(model=model_resource_name)

            # Note: Vertex AI requires model aliases to follow this format: [a-z][a-zA-Z0-9-]{0,126}[a-z0-9]
            formatted_model_semantic_version = (
                f"v{model_semantic_version.replace('.', '-')}"
            )

            model_registry.add_version_aliases(
                new_aliases=[formatted_model_semantic_version], version=version_id
            )

        return model_resource_name

    def delete_version(self, model_resource: aiplatform.Model):
        """
        Deletes a specific model version from the Vertex AI Model Registry.
        This is typically used for rollback if a two-phase commit fails.

        This method now handles cases where the target version is the 'default'
        version, reassigning the 'default' alias to the previous version if available,
        or deleting the entire model if it's the only version.

        Args:
            model_resource: An aiplatform.Model object representing the specific
                            model version to be deleted.

        Raises:
            TypeError: If `model_resource` is not an `aiplatform.Model` object.
            Exception: If the deletion operation fails.
        """
        if not isinstance(model_resource, aiplatform.Model):
            raise TypeError("model_resource must be an aiplatform.Model object")

        version_id = model_resource.version_id
        resource_name = model_resource.resource_name

        logger.warning(
            f"Attempting to delete model version '{version_id}' "
            f"from model: '{resource_name}'"
        )

        try:
            model_registry = aiplatform.models.ModelRegistry(model=resource_name)

            # Get the details of the target version to check its aliases
            target_version_details = model_registry.get_model(version=version_id)

            # Check if the target version is currently the default version
            if "default" in target_version_details.version_aliases:
                logger.info(
                    f"Version '{version_id}' of model '{resource_name}' is the default version. Attempting to reassign 'default' alias."
                )

                # Get all versions to find a suitable previous one
                all_versions = model_registry.list_versions()

                # Filter out the target version and sort the remaining by creation time (latest first)
                other_versions = [v for v in all_versions if v.version_id != version_id]
                other_versions.sort(
                    key=lambda v: v.version_create_time, reverse=True
                )  # Latest created non-target version first

                if other_versions:
                    # Found at least one other version; reassign 'default' to the most recent non-target version
                    previous_version_details = other_versions[0]
                    logger.info(
                        f"Found previous version '{previous_version_details.version_id}' to assign 'default' alias to."
                    )

                    # Add 'default' alias to the previous version
                    new_previous_aliases = list(
                        previous_version_details.version_aliases
                    )
                    if "default" not in new_previous_aliases:
                        new_previous_aliases.append("default")

                    model_registry.add_version_aliases(
                        new_aliases=new_previous_aliases,
                        version=previous_version_details.version_id,
                    )
                    logger.info(
                        f"Reassigned 'default' alias to version '{previous_version_details.version_id}'."
                    )

                    # Remove 'default' alias from the target version (keeping any other aliases)
                    current_target_aliases = [
                        alias
                        for alias in target_version_details.version_aliases
                        if alias != "default"
                    ]
                    # An alias is always required, if an alias doesn't exist, set a dummy one for now
                    if not current_target_aliases:
                        current_target_aliases = ["v0"]
                    model_registry.remove_version_aliases(
                        version=version_id, target_aliases=current_target_aliases
                    )
                    logger.info(f"Removed 'default' alias from version '{version_id}'.")

                    # Now that 'default' is reassigned, proceed with deleting the target version
                    model_registry.delete_version(version=version_id)
                    logger.info(
                        f"Successfully rolled back and deleted version '{version_id}' of model '{resource_name}'."
                    )

                else:
                    # The target version is the ONLY version of the model and it's the default.
                    # As per Vertex AI recommendation, delete the entire model resource.
                    logger.warning(
                        f"Version '{version_id}' is the ONLY version of model '{resource_name}' "
                        f"AND it's the default. Deleting the entire model resource as per Vertex AI recommendation."
                    )
                    model_resource.delete()  # Deletes the entire model resource, including all versions.
                    logger.info(
                        f"Successfully deleted the entire model '{resource_name}' as it only had one version."
                    )
                    return  # Exit the function as the model is entirely gone
            else:
                # The target version is NOT the default, so we can delete it directly.
                logger.info(
                    f"Version '{version_id}' is not the default. Proceeding with direct deletion."
                )
                model_registry.delete_version(version=version_id)
                logger.info(
                    f"Successfully deleted non-default version '{version_id}' of model '{resource_name}'."
                )

        except Exception as e:
            logger.error(
                f"Failed to delete model version '{version_id}' of model '{resource_name}' during rollback. "
                f"See logs for details. {e}",
                exc_info=True,
            )
            raise
