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

"""Strategies for the model.monitoring module."""

import logging
from abc import ABC, abstractmethod
from typing import Optional

from vertexai.resources.preview import ml_monitoring
from google.api_core import exceptions

import mdk.config
from mdk.model.monitoring.models import MonitoringAppConfig

logger = logging.getLogger(__name__)


class MonitorRetrievalStrategy(ABC):
    """Abstract base class for strategies to retrieve a Vertex AI ModelMonitor."""

    @abstractmethod
    def get_or_create_monitor(
        self,
        config: MonitoringAppConfig,
        monitoring_schema: ml_monitoring.spec.ModelMonitoringSchema,
        access_token: Optional[str],
    ) -> ml_monitoring.ModelMonitor:
        """Retrieves or creates a ModelMonitor based on the implemented strategy."""
        pass


class FindOrCreateMonitorStrategy(MonitorRetrievalStrategy):
    """
    Concrete strategy to find an existing monitor, or create a new one if not found.
    The search order is:
    1. By explicit resource name.
    2. By matching model resource name and version ID.
    3. If none found, create a new one.
    """

    def get_or_create_monitor(
        self,
        config: MonitoringAppConfig,
        monitoring_schema: ml_monitoring.spec.ModelMonitoringSchema,
        access_token: Optional[str] = None,
    ) -> ml_monitoring.ModelMonitor:
        from mdk.model.registry import (
            get_vertex_ai_model_resource_name_and_version_for_inference,
        )

        # 1. Attempt to get by explicit resource name
        if config.monitoring.model_monitor_resource_name:
            try:
                logger.info(
                    f"Attempting to retrieve monitor by resource name: {config.monitoring.model_monitor_resource_name}"
                )
                return ml_monitoring.ModelMonitor(
                    model_monitor_name=config.monitoring.model_monitor_resource_name
                )
            except exceptions.NotFound:
                logger.warning(
                    f"Monitor with name '{config.monitoring.model_monitor_resource_name}' not found. "
                    "Searching by model details instead."
                )

        # 2. Retrieve the model using the model_reference_config
        model_ref_fields = set(mdk.config.ModelReferenceConfig.model_fields.keys())
        model_reference_config_data = (
            config.monitoring.model_reference_config.model_dump(
                include=model_ref_fields
            )
        )
        try:
            vertex_ai_model_resource_name, vertex_ai_model_version_id = (
                get_vertex_ai_model_resource_name_and_version_for_inference(
                    model_reference_config_data=model_reference_config_data,
                    gcp_project_id=config.gcp.project_id,
                    gcp_region=config.gcp.region,
                    expanded_model_registry_endpoint=config.gcp.expanded_model_registry_endpoint,
                    access_token=access_token,
                )
            )
            logger.info(
                f"Successfully retrieved Vertex AI model with name: '{vertex_ai_model_resource_name}', "
                f"Version: '{vertex_ai_model_version_id}'"
            )
        except Exception as e:
            log_message = (
                f"An unexpected error occurred during Vertex AI Model loading for "
                f"model_reference_config: '{config.monitoring.model_reference_config}'. "
                f"Original error: {e}"
            )
            logger.error(log_message, exc_info=True)
            raise RuntimeError(
                f"Deployment preparation failed due to an unexpected error while loading the model. "
                f"See logs for details. Error: {e}"
            ) from e

        # 3. Search for a monitor matching the model and version
        logger.info(
            f"Searching for an existing monitor for model '{vertex_ai_model_resource_name}' "
            f"version '{vertex_ai_model_version_id}'"
        )
        monitors = ml_monitoring.ModelMonitor.list()
        for monitor in monitors:
            target = monitor.gca_resource.model_monitoring_target
            if (
                target
                and target.vertex_model
                and target.vertex_model.model == vertex_ai_model_resource_name
                and str(target.vertex_model.model_version_id)
                == str(vertex_ai_model_version_id)
            ):
                logger.info(f"Found existing model monitor: {monitor.resource_name}")
                return monitor

        # 3. If none found, create a new one
        logger.info("No existing model monitor found. Creating a new one.")
        return ml_monitoring.ModelMonitor.create(
            project=config.gcp.project_id,
            location=config.gcp.region,
            display_name=config.monitoring.model_monitor_display_name,
            model_name=vertex_ai_model_resource_name,
            model_version_id=str(vertex_ai_model_version_id),
            model_monitoring_schema=monitoring_schema,
        )
