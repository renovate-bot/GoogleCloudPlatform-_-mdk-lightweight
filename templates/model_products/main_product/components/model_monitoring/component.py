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

import datetime
import inspect
import logging
import os

from kfp import dsl
import google.cloud.aiplatform

from mdk.model.monitoring import MonitoringAppConfig
from mdk.model.monitoring.providers.factory import ProviderFactory
from mdk.model.registry import get_emr_model_object
import mdk.util.framework
import mdk.config
import mdk.custom_job

logger = logging.getLogger(__name__)


@dsl.component(
    target_image=mdk.util.framework.getTargetImage(__file__, "model_monitoring")
)
def model_monitoring(
    gcp_config_filename: str,
    pipeline_config_filename: str,
    general_config_filename: str,
    model_monitoring_job: dsl.Output[dsl.Artifact],
    model_monitoring_schedule: dsl.Output[dsl.Artifact],
    target_endpoint: dsl.Input[dsl.Artifact] = None,
):
    ############################################################################
    # INITIALIZATION:
    # --------------------------------------------------------------------------
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # Read our config files.
    gcp_config = mdk.config.GCPConfig.from_yaml_file(gcp_config_filename)
    pipeline_config = mdk.config.readYAMLConfig(pipeline_config_filename)
    general_config = mdk.config.readAndMergeYAMLConfig(
        config_filename=general_config_filename,
        environment=gcp_config.deployment_environment,
    )
    model_monitoring_config = general_config["model_monitoring"]

    if not general_config["model_monitoring"].get("target_endpoint"):
        if target_endpoint:
            logger.info(
                f"Using KFP endpoint '{target_endpoint.uri}' as target_endpoint."
            )
            general_config["model_monitoring"]["target_endpoint"] = target_endpoint.uri
        elif general_config["model_monitoring"].get("target_dataset_uri"):
            logger.info(
                f"Target dataset '{general_config['model_monitoring'].get('target_dataset_uri')}' found. "
                "Running batch inference monitoring."
            )

    app_config = MonitoringAppConfig.from_yaml_files(
        gcp_config_path=gcp_config_filename,
        general_config_path=general_config_filename,
    )

    ############################################################################
    # MANAGE INPUTS (see input arguments in function signature, above)
    # --------------------------------------------------------------------------

    # Set our project ID.
    google.cloud.aiplatform.init(project=gcp_config.project_id, location=gcp_config.region)  # fmt: skip
    os.environ["GOOGLE_CLOUD_PROJECT"] = gcp_config.project_id

    # Optional: handle custom job execution if specified.
    staging_bucket = f"{gcp_config.pipeline_staging_dir}/custom_job/staging"
    base_output_dir = f"{gcp_config.pipeline_staging_dir}/custom_job/{timestamp}"
    custom_job_executed = mdk.custom_job.handle_custom_job_if_configured(
        gcp_config=gcp_config.model_dump(),
        model_config=model_monitoring_config,  # named model_monitoring_config instead
        pipeline_config=pipeline_config,
        component_name=inspect.currentframe().f_code.co_name,
        staging_bucket=staging_bucket,
        base_output_dir=base_output_dir,
    )
    if custom_job_executed:
        model_monitoring_job.uri = f"{base_output_dir}/model_monitoring_job"
    else:
        # The rest of this component is run by default if "container_specs" are not specified
        ############################################################################
        # MANAGE INPUTS (see input arguments in function signature, above)
        # --------------------------------------------------------------------------

        # Get our webex notification channels.
        cloud_resources_config = mdk.config.readCloudResourcesConfig(
            gcp_config.data_bucket
        )
        notification_channels = _get_notification_channels(cloud_resources_config)
        logger.info(f"Using notification channels: {notification_channels}")

        # This will be set if we are running locally.
        access_token = os.environ.get("ID_TOKEN")

        # Get the training_data_uri from the EMR:

        # Get the model URI.
        model_ref_fields = set(mdk.config.ModelReferenceConfig.model_fields.keys())
        model_reference_config_data = {
            key: general_config["general"].get(key) for key in model_ref_fields
        }
        # Add in deployment_environment, this field is required by ModelReferenceConfig
        model_reference_config_data["deployment_environment"] = (
            gcp_config.deployment_environment
        )
        try:
            model_response_dict = get_emr_model_object(
                model_reference_config_data=model_reference_config_data,
                gcp_project_id=gcp_config.project_id,
                gcp_region=gcp_config.region,
                expanded_model_registry_endpoint=gcp_config.expanded_model_registry_endpoint,
                access_token=access_token,
            )
            logger.info(f"Successfully retrieved model object: {model_response_dict}")
            training_data_uri = model_response_dict.get("training_data_uri")
        except Exception as e:
            logger.warning(f"Error retrieving model object from EMR: {e}")
            training_data_uri = None

        if not training_data_uri:
            logger.info("training_data_uri not found in EMR response. Reading baseline_dataset_uri from config.")
            training_data_uri = app_config.monitoring.baseline_dataset_uri
            
        if not training_data_uri:
            raise ValueError(
                "baseline_dataset_uri is missing. In Lite Mode, it must be provided in the 'model_monitoring' section of the config."
            )

        # Create our Vertex AI model monitoring provider, and set up monitoring.
        monitoring_provider = ProviderFactory.get_provider(
            "vertex", app_config, access_token
        )
        job_url, schedule_url = monitoring_provider.set_up_monitoring(
            training_data_uri, notification_channels
        )

        ############################################################################
        # MANAGE OUTPUTS (see output arguments in function signature, above)
        # --------------------------------------------------------------------------

        # Set our component output.
        model_monitoring_job.uri = job_url
        model_monitoring_schedule.uri = schedule_url


def _get_notification_channels(cloud_resources_config: dict) -> list[str]:
    """Read the project factory config file to get the notification channels,
    without raising an exception.

    Args:
        cloud_resources_config (dict): Parsed version of config file written
            to GCS upon project creation by project factory.

    Returns:
        list[str]: List of notification channels as reflected in project factory
            config.
    """
    # The original YAML looks something like this:
    #
    # "notification_channels":
    #   "retrain_alerts":
    #     "display_name": "Retrain Alerts"
    #     "id": "projects/[...]/notificationChannels/1234567890123456789"
    #     "name": "projects/[...]/notificationChannels/1234567890123456789"
    #   "webex_alerts":
    #     "display_name": "Webex Alerts"
    #     "id": "projects/[...]/notificationChannels/1234567890123456789"
    #     "name": "projects/[...]/notificationChannels/1234567890123456789"

    channels = []
    for alerts_type in ("retrain_alerts", "webex_alerts"):
        try:
            notification_channels = cloud_resources_config["notification_channels"]
            alerts = notification_channels[alerts_type]
            channel = alerts["id"]

        # If this file breaks somehow, we don't want to break the pipeline, so
        #   we print an error message but continue onward.
        except Exception as e:
            logger.error(
                f"ERROR getting notification channel from project resources"
                f" YAML on GCP: {e}"
            )
            channel = None

        if channel:
            channels.append(channel)

    return channels
