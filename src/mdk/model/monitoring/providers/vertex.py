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

"""Implements the VertexAIMonitoringProvider for Google Cloud Vertex AI."""

import datetime
import logging
import time
from typing import Tuple, Optional

import vertexai
from vertexai.resources.preview import ml_monitoring
import mdk.util.auto_retraining

from mdk.model.monitoring.providers.base import MonitoringProvider
from mdk.model.monitoring.models import MonitoringAppConfig, DriftThresholdsConfig
from mdk.model.monitoring.strategies import FindOrCreateMonitorStrategy

logger = logging.getLogger(__name__)


class VertexAIMonitoringProvider(MonitoringProvider):
    """Manages setting up and running model monitoring jobs on Google Cloud
    Vertex AI.
    """

    def __init__(self, config: MonitoringAppConfig, access_token: Optional[str] = None):
        self.config = config
        self.access_token = access_token
        self.monitor_retrieval_strategy = FindOrCreateMonitorStrategy()

        logger.info(
            f"Initializing Vertex AI for project '{config.gcp.project_id}' in '{config.gcp.region}'"
        )
        vertexai.init(project=config.gcp.project_id, location=config.gcp.region)

    def set_up_monitoring(
        self,
        baseline_dataset_uri: str,
        notification_channels: list[str] = [],
    ) -> Tuple[Optional[str], Optional[str]]:
        """Orchestrates the model monitoring setup process.

        Args:
            baseline_dataset_uri (str): The baseline dataset to compare drifts to.
                This is typically your training dataset.
            notification_channels (list[str]): Channels to which errors should be
                sent (e.g. for Webex notification)

        Returns:
            tuple: The elements of the returned tuple are, respectively:
                - The URL or resource name for the created monitoring job.
                - The URL or resource name for the created monitoring schedule.
        """
        monitoring_schema = self._build_monitoring_schema()

        model_monitor = self.monitor_retrieval_strategy.get_or_create_monitor(
            config=self.config,
            monitoring_schema=monitoring_schema,
            access_token=self.access_token,
        )

        baseline_input = None
        if baseline_dataset_uri:
            baseline_input = self._build_monitoring_input(baseline_dataset_uri)
            
        target_dataset_uri = self.config.monitoring.target_dataset_uri
        target_endpoint = self.config.monitoring.target_endpoint

        # Determine inference type based on the target input spec
        inference_type = "real-time" if target_endpoint else "batch"
        if inference_type == "real-time":
            target_input = self._build_monitoring_input(
                target_endpoint,
                self.config.monitoring.target_dataset_query,  # Optional
            )
        else:
            target_input = self._build_monitoring_input(
                target_dataset_uri,
                self.config.monitoring.target_dataset_query,  # Optional
            )
        logger.info(f"Detected inference type: '{inference_type}'")

        notification_spec = self._build_notification_spec(notification_channels)
        output_spec = self._build_output_spec()

        feature_drift_spec = self._build_data_drift_spec(
            self.config.monitoring.feature_drift
        )

        prediction_drift_spec = None
        if self.config.monitoring.use_prediction_output_drift:
            prediction_drift_spec = self._build_data_drift_spec(
                self.config.monitoring.prediction_drift
            )

        monitoring_job_kwargs = {
            "inference_type": inference_type,
            "model_monitor": model_monitor,
            "target_dataset": target_input,
            "notification_spec": notification_spec,
            "output_spec": output_spec,
            "feature_drift_spec": feature_drift_spec,
            "prediction_output_drift_spec": prediction_drift_spec,
        }

        if baseline_input:
            monitoring_job_kwargs["baseline_dataset"] = baseline_input

        monitoring_job, monitoring_schedule = self._run_or_schedule_monitoring(
            **monitoring_job_kwargs
        )

        # Set up retraining trigger if enabled and if a job or schedule was created.
        logger.info(
            f"Setup Retraining: '{self.config.monitoring.retraining.set_up_retraining}'"
        )

        if self.config.monitoring.retraining.set_up_retraining:
            if monitoring_job or monitoring_schedule:
                self._set_up_retraining_trigger(
                    model_monitor, monitoring_job, monitoring_schedule
                )
            else:
                logger.error(
                    "Retraining trigger setup requested, but no monitoring job or schedule was created. Skipping."
                )

        # Return the most relevant URL
        if not monitoring_job and not monitoring_schedule:
            message = "No monitoring job or schedule was created."
            logger.error(message)
            return message
        monitoring_job_uri = (
            self._get_job_console_url(monitoring_job.resource_name)
            if monitoring_job
            else None
        )
        monitoring_schedule_uri = (
            self._get_schedule_console_url(monitoring_schedule.name)
            if monitoring_schedule
            else None
        )
        return monitoring_job_uri, monitoring_schedule_uri

    def _build_monitoring_schema(self) -> ml_monitoring.spec.ModelMonitoringSchema:
        """Builds the monitoring schema spec from the configuration."""
        cfg = self.config.monitoring
        feature_fields = [
            ml_monitoring.spec.FieldSchema(name=k, data_type=v)
            for k, v in cfg.feature_fields_schema_map.items()
        ]

        ground_truth_fields = None
        if cfg.ground_truth_fields_schema_map:
            ground_truth_fields = [
                ml_monitoring.spec.FieldSchema(name=k, data_type=v)
                for k, v in cfg.ground_truth_fields_schema_map.items()
            ]

        prediction_fields = None
        if cfg.prediction_fields_schema_map:
            prediction_fields = [
                ml_monitoring.spec.FieldSchema(name=k, data_type=v)
                for k, v in cfg.prediction_fields_schema_map.items()
            ]

        return ml_monitoring.spec.ModelMonitoringSchema(
            feature_fields=feature_fields,
            ground_truth_fields=ground_truth_fields,
            prediction_fields=prediction_fields,
        )

    def _build_monitoring_input(
        self, dataset_uri: str, query: Optional[str] = None
    ) -> ml_monitoring.spec.MonitoringInput:
        """Builds a monitoring input spec based on the URI scheme (gs, bq, or endpoint)."""
        if dataset_uri and dataset_uri.startswith("bq://"):
            return ml_monitoring.spec.MonitoringInput(table_uri=dataset_uri)
        elif dataset_uri and dataset_uri.startswith("gs://"):
            # Assuming CSV format for GCS for simplicity, could be configured
            return ml_monitoring.spec.MonitoringInput(
                gcs_uri=dataset_uri, data_format="csv"
            )
        elif (
            dataset_uri and "/endpoints/" in dataset_uri
        ):  # Heuristic for endpoint resource name
            if not hasattr(self.config.monitoring, "window"):
                logger.warning(
                    "Real-time monitoring specified but no 'window' found in config. Defaulting to '24h'."
                )
                window = "24h"
            else:
                window = self.config.monitoring.window
            logger.info(
                f"Configuring real-time monitoring for endpoint '{dataset_uri}' with window '{window}'"
            )
            return ml_monitoring.spec.MonitoringInput(
                endpoints=[dataset_uri], window=window
            )
        elif query:
            return ml_monitoring.spec.MonitoringInput(query=query)
        else:
            raise ValueError(
                f"Unsupported dataset URI, query, or resource name scheme: {dataset_uri}"
            )

    def _build_notification_spec(
        self,
        notification_channels: list[str],
    ) -> ml_monitoring.spec.notification.NotificationSpec | None:
        """Builds the notification spec.

        Args:
            notification_channels (list[str]) Channels to which errors should be
                sent (typically will be for Webex notification)

        Returns:
            ml_monitoring.spec.notification.NotificationSpec: Notificaction spec
        """
        cfg = self.config
        if not cfg.monitoring.user_emails and not notification_channels:
            logger.warning(
                "No notification channels or user emails specified. Alerts will not be sent."
            )
            return None  # Monitoring job allows this to be optional

        return ml_monitoring.spec.notification.NotificationSpec(
            user_emails=cfg.monitoring.user_emails,
            notification_channels=notification_channels,
            enable_cloud_logging=True,
        )

    def _build_output_spec(self) -> ml_monitoring.spec.OutputSpec:
        """Builds the output spec if a GCS logs URI is provided."""
        if self.config.monitoring.gcs_logs_uri:
            return ml_monitoring.spec.OutputSpec(
                gcs_base_dir=self.config.monitoring.gcs_logs_uri
            )
        return None

    def _build_data_drift_spec(
        self,
        drift_config: DriftThresholdsConfig,
    ) -> ml_monitoring.spec.DataDriftSpec:
        """Builds a data drift spec from the drift configuration model."""
        return ml_monitoring.spec.DataDriftSpec(
            categorical_metric_type=drift_config.categorical_metric_type,
            numeric_metric_type=drift_config.numeric_metric_type,
            default_categorical_alert_threshold=drift_config.default_categorical_alert_threshold,
            default_numeric_alert_threshold=drift_config.default_numeric_alert_threshold,
            feature_alert_thresholds=drift_config.feature_alert_thresholds,
        )

    def _run_or_schedule_monitoring(self, inference_type: str, **kwargs):
        """
        Creates a scheduled job for real-time inference or runs an on-demand job
        for batch inference. Also creates a schedule for batch if cron is set.
        """
        job_display_name = (
            f"{self.config.monitoring.model_monitor_job_display_name}-"
            f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        )

        tabular_objective = ml_monitoring.spec.TabularObjective(
            feature_drift_spec=kwargs.get("feature_drift_spec"),
            prediction_output_drift_spec=kwargs.get("prediction_output_drift_spec"),
        )

        common_args = {
            "display_name": job_display_name,
            "baseline_dataset": kwargs["baseline_dataset"],
            "target_dataset": kwargs["target_dataset"],
            "tabular_objective_spec": tabular_objective,
            "notification_spec": kwargs["notification_spec"],
            "output_spec": kwargs["output_spec"],
        }

        monitoring_job = None
        monitoring_schedule = None

        # For real-time, we ONLY create a schedule.
        if inference_type == "real-time":
            if not self.config.monitoring.cron_schedule:
                raise ValueError(
                    "A 'cron_schedule' must be provided for real-time monitoring."
                )

            logger.info(
                f"Creating a scheduled monitoring job for real-time endpoint with cron: '{self.config.monitoring.cron_schedule}'"
            )
            monitoring_schedule = kwargs["model_monitor"].create_schedule(
                cron=self.config.monitoring.cron_schedule, **common_args
            )
            logger.info(type(monitoring_schedule))
            logger.info(
                f"Model Monitoring schedule '{monitoring_schedule.display_name}' created."
            )

        # For batch, we run an on-demand job, and optionally create a schedule.
        elif inference_type == "batch":
            if self.config.monitoring.cron_schedule:
                logger.info(
                    f"Creating a scheduled monitoring job with cron: '{self.config.monitoring.cron_schedule}'"
                )
                monitoring_schedule = kwargs["model_monitor"].create_schedule(
                    cron=self.config.monitoring.cron_schedule, **common_args
                )
                logger.info(type(monitoring_schedule))

                logger.info(
                    f"Model Monitoring schedule '{monitoring_schedule.display_name}' created."
                )

            logger.info(
                f"Running on-demand monitoring job: '{job_display_name}' for batch inference."
            )
            monitoring_job = kwargs["model_monitor"].run(**common_args)
            logger.info("Waiting 10 seconds to allow the monitoring job to initialize.")
            time.sleep(10)
            logger.info(f"Model Monitoring job {monitoring_job.resource_name} started.")

        return monitoring_job, monitoring_schedule

    def _set_up_retraining_trigger(
        self,
        monitor: ml_monitoring.ModelMonitor,
        job: Optional[ml_monitoring.ModelMonitoringJob],
        schedule,
    ):
        """Sets up the retraining trigger artifacts using the MDK."""
        logger.info("Setting up retraining trigger artifacts.")
        gcp_cfg = self.config.gcp
        retraining_cfg = self.config.monitoring.retraining

        # Extract the final component (ID) from the resource names
        monitor_id = monitor.name.split("/")[-1]
        job_id = job.name.split("/")[-1] if job else None
        schedule_id = schedule.name.split("/")[-1] if schedule else None

        mdk.util.auto_retraining.set_up_retraining_via_model_monitoring(
            monitor_id=monitor_id,
            job_id=job_id,
            schedule_id=schedule_id,
            gcs_bucket_root=gcp_cfg.data_bucket,
            pipeline_root=gcp_cfg.pipeline_staging_dir,
            pipeline_runner_sa=gcp_cfg.pipeline_service_account,
            training_pipeline_name=retraining_cfg.training_pipeline_name,
            inference_pipeline_name=retraining_cfg.inference_pipeline_name,
            experiment_name=gcp_cfg.experiment_name,
            region=gcp_cfg.region,
            ar_repo=gcp_cfg.artifact_registry_repo,
            environment=gcp_cfg.deployment_environment,
            app_root=retraining_cfg.app_root,
        )

    def _get_job_console_url(self, job_resource_name: str) -> str:
        """Constructs the URL to the monitoring job in the Google Cloud Console."""
        # projects/PROJECT/locations/REGION/modelMonitors/MONITOR_ID/modelMonitoringJobs/JOB_ID
        parts = job_resource_name.split("/")
        if len(parts) != 8:
            return job_resource_name  # Return raw name if format is unexpected

        return (
            f"https://console.cloud.google.com/vertex-ai/model-monitoring/locations/"
            f"{parts[3]}/model-monitors/{parts[5]}/model-monitoring-jobs/{parts[7]}?project={parts[1]}"
        )

    def _get_schedule_console_url(self, schedule_resource_name: str) -> str:
        """Constructs the URL to the monitoring schedule in the Google Cloud Console."""
        # projects/PROJECT/locations/REGION/modelMonitors/MONITOR_ID/schedules/SCHEDULE_ID
        logger.info(f"schedule name: {schedule_resource_name}")
        parts = schedule_resource_name.split("/")
        if len(parts) != 6:
            return schedule_resource_name  # Return raw name if format is unexpected

        return (
            f"https://console.cloud.google.com/vertex-ai/model-monitoring/locations/"
            f"{parts[3]}/schedules/{parts[5]}/overview?project={parts[1]}"
        )
