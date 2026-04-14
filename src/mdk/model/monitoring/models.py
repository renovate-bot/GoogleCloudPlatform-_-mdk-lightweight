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

"""Pydantic models for the model.monitoring module."""

from typing import Literal, Optional

from pydantic import (
    BaseModel,
    Field,
    confloat,
    constr,
    field_validator,
    model_validator,
)
import yaml

import mdk.config

# Supported data types for Vertex AI schema fields
SchemaDataType = Literal["float", "integer", "boolean", "string", "categorical"]

# Supported metric types for drift detection
CategoricalMetricType = Literal["l_infinity", "jensen_shannon_divergence"]
NumericMetricType = Literal["jensen_shannon_divergence"]


class DriftThresholdsConfig(BaseModel):
    """Configuration for data drift alert thresholds and metrics."""

    default_categorical_alert_threshold: confloat(ge=0.0, le=1.0) = 0.1
    default_numeric_alert_threshold: confloat(ge=0.0, le=1.0) = 0.1
    categorical_metric_type: CategoricalMetricType = "l_infinity"
    numeric_metric_type: NumericMetricType = "jensen_shannon_divergence"
    feature_alert_thresholds: Optional[dict[str, confloat(ge=0.0, le=1.0)]] = Field(
        None, description="Per-feature alert thresholds to override defaults."
    )


class RetrainingConfig(BaseModel):
    """Configuration for setting up automatic retraining triggers."""

    set_up_retraining: bool = False
    training_pipeline_name: Optional[str] = Field(
        None, description="Name of the training pipeline to trigger."
    )
    inference_pipeline_name: Optional[str] = Field(
        None, description="Name of the inference pipeline."
    )
    app_root: str = "/app"

    @model_validator(mode="after")
    def check_required_fields_if_enabled(self):
        if self.set_up_retraining and not self.training_pipeline_name:
            raise ValueError(
                "`training_pipeline_name` is required when `set_up_retraining` is True."
            )
        return self


class ModelMonitoringConfig(BaseModel):
    """Main configuration for the Vertex AI Model Monitoring job."""

    target_dataset_uri: Optional[constr(pattern=r"^(gs|bq)://.*")] = Field(
        None, description="GCS or BigQuery URI for the target dataset (for batch)."
    )

    baseline_dataset_uri: Optional[constr(pattern=r"^(gs|bq)://.*")] = Field(
        None, description="GCS or BigQuery URI for the baseline dataset."
    )

    target_endpoint: Optional[str] = Field(
        None, description="Endpoint resource name for the target (for real-time)."
    )

    target_dataset_query: Optional[str] = Field(
        None,
        description="Standard SQL for BigQuery to be used instead of the target_dataset_uri.",
    )

    model_reference_config: mdk.config.ModelReferenceConfig = Field(
        default_factory=mdk.config.ModelReferenceConfig,
        description="Specifies how to retrieve a given model.",
    )

    feature_fields_schema_map: dict[str, SchemaDataType] = Field(
        ..., description="Schema map for input features."
    )

    model_monitor_job_display_name: str = "model-monitoring-job"
    model_monitor_display_name: str = "model-monitor"
    model_monitor_resource_name: Optional[str] = Field(
        None, description="Resource name of an existing monitor to use."
    )

    ground_truth_fields_schema_map: Optional[dict[str, SchemaDataType]] = None
    prediction_fields_schema_map: Optional[dict[str, SchemaDataType]] = None

    gcs_logs_uri: Optional[constr(pattern=r"^gs://.*")] = None
    use_prediction_output_drift: bool = False
    cron_schedule: Optional[str] = Field(
        None, description="The schedule to use for monitoring."
    )
    window: Optional[str] = Field(
        "24h",
        description="Time window for collecting endpoint logs for real-time monitoring (e.g., '24h').",
    )
    user_emails: list[str] = Field(
        [], description="list of user emails for notifications."
    )

    feature_drift: DriftThresholdsConfig = Field(default_factory=DriftThresholdsConfig)
    prediction_drift: DriftThresholdsConfig = Field(
        default_factory=DriftThresholdsConfig
    )

    retraining: RetrainingConfig = Field(default_factory=RetrainingConfig)

    @model_validator(mode="after")
    def validate_monitoring_config(self) -> "ModelMonitoringConfig":
        """Validate target inputs."""
        # If target_dataset_uri, target_endpoint, and target_dataset_query are None, raise an error.
        if (
            not self.target_dataset_uri
            and not self.target_endpoint
            and not self.target_dataset_query
        ):
            raise ValueError(
                'One of ["target_dataset_uri", "target_endpoint", "target_dataset_query"] must be specified.'
            )
        return self

    @field_validator("feature_fields_schema_map", mode="before")
    @classmethod
    def feature_fields_schema_map_validator(
        cls, d: Optional[dict[str, str]]
    ) -> dict[str, str] | None:
        return cls.to_lower(d)

    @field_validator("ground_truth_fields_schema_map", mode="before")
    @classmethod
    def ground_truth_schema_map_validator(
        cls, d: Optional[dict[str, str]]
    ) -> dict[str, str] | None:
        return cls.to_lower(d)

    @field_validator("prediction_fields_schema_map", mode="before")
    @classmethod
    def prediction_schema_map_validator(
        cls, d: Optional[dict[str, str]]
    ) -> dict[str, str] | None:
        return cls.to_lower(d)

    @classmethod
    def to_lower(
        cls,
        d: dict[str, str] | None,
    ) -> dict[str, str] | None:
        """This maps the values in a dict to lower case.  It allows the user to
        supply values like "STRING" or "String" and have it automatically
        converted to to "string", which is useful because Vertex AI expects
        lower casing.

        If d is None, returns None.

        Args:
            d (Optional dict[str, str]): The dict whose values need to mapped to
                lower case.
        Returns:
            dict[str, str] | None: The input dict, with values mapped to lower
                case.  If d is None, returns None.
        """
        if not d:
            return d
        d = {k: v.lower() for k, v in d.items()}
        return d


class MonitoringAppConfig(BaseModel):
    """A single, unified configuration object for the application."""

    gcp: mdk.config.GCPConfig
    monitoring: ModelMonitoringConfig
    environment: str

    @classmethod
    def from_yaml_files(
        cls, gcp_config_path: str, general_config_path: str
    ) -> "MonitoringAppConfig":
        """Loads and validates configuration from YAML files."""
        with open(gcp_config_path, "r") as f:
            gcp_data = yaml.safe_load(f)
        environment = gcp_data["deployment_environment"]

        general_config = mdk.config.readAndMergeYAMLConfig(
            config_filename=general_config_path, environment=environment
        )

        general_data = general_config["general"]
        monitoring_data = general_config["model_monitoring"]

        # Handle the nested Pydantic Models
        model_ref_fields = set(mdk.config.ModelReferenceConfig.model_fields.keys())
        model_reference_data = {}

        for key, value in general_data.items():
            if key in model_ref_fields:
                model_reference_data[key] = value

        # Get deployment_environment from the GCP config
        model_reference_data["deployment_environment"] = environment
        # Combine model_reference_config with the deployment config data
        monitoring_data["model_reference_config"] = model_reference_data

        return cls(
            gcp=mdk.config.GCPConfig(**gcp_data),
            monitoring=ModelMonitoringConfig(**monitoring_data),
            environment=environment,
        )
