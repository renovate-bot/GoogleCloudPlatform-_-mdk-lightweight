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

from mdk.model.registry import upload_model, RegistryAppConfig
import mdk.util.framework
import mdk.util.storage
import mdk.config
import mdk.custom_job

import google.cloud.aiplatform
from kfp import dsl
import json
import datetime
import inspect
import logging
import os

logger = logging.getLogger(__name__)

PM_FILENAME = "upload_performance_metrics.json"


@dsl.component(
    target_image=mdk.util.framework.getTargetImage(__file__, "upload_to_model_registry")
)
def upload_to_model_registry(
    gcp_config_filename: str,
    pipeline_config_filename: str,
    general_config_filename: str,
    pipeline_job_name: str,
    trained_model: dsl.Input[dsl.Model],
    train_dataset: dsl.Input[dsl.Dataset],
    scalar_metrics: dsl.Input[dsl.Metrics],
    uploaded_model: dsl.Output[dsl.Model],
):
    """
    This component uploads a model to the Vertex AI Model Registry and the
    expanded model registry using the refactored orchestration logic.
    """
    ############################################################################
    # INITIALIZATION:
    # --------------------------------------------------------------------------
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # Read our config files.
    gcp_config = mdk.config.GCPConfig.from_yaml_file(gcp_config_filename).model_dump()
    pipeline_config = mdk.config.readYAMLConfig(pipeline_config_filename)
    general_config = mdk.config.readAndMergeYAMLConfig(
        config_filename=general_config_filename,
        environment=gcp_config["deployment_environment"],
    )
    # Set Vertex AI Experiment from gcp config
    general_config["model_registry"]["vertex_ai_experiment_name"] = gcp_config[
        "experiment_name"
    ]
    model_registry_config = general_config["model_registry"]
    if not model_registry_config.get("training_data_uri"):
        model_registry_config["training_data_uri"] = train_dataset.uri

    app_config = RegistryAppConfig.from_yaml_files(
        gcp_config_path=gcp_config_filename,
        general_config_path=general_config_filename,
    )

    # Set our project ID.
    google.cloud.aiplatform.init(project=gcp_config.get("project_id"))  # fmt: skip
    os.environ["GOOGLE_CLOUD_PROJECT"] = gcp_config.get("project_id")

    # Bail out if we detect we're running locally.
    if not uploaded_model.uri.startswith("gs://"):
        logger.info("*** LOCAL RUN DETECTED -> Skipping model upload. ***")
        uploaded_model.uri = None
        return

    # Optional: handle custom job execution if specified.
    staging_bucket = f"{gcp_config.get('pipeline_staging_dir')}/custom_job/staging"
    base_output_dir = f"{gcp_config.get('pipeline_staging_dir')}/custom_job/{timestamp}"
    custom_job_executed = mdk.custom_job.handle_custom_job_if_configured(
        gcp_config=gcp_config,
        model_config=model_registry_config,
        pipeline_config=pipeline_config,
        component_name=inspect.currentframe().f_code.co_name,
        staging_bucket=staging_bucket,
        base_output_dir=base_output_dir,
        input_artifact_map={
            "trained_model": trained_model.uri,
            "scalar_metrics": scalar_metrics.uri,
        },
    )
    if custom_job_executed:
        # kfp_classification_report.uri = f"{base_output_dir}/kfp_classification_report"
        uploaded_model.uri = f"{base_output_dir}/uploaded_model"
    else:
        # The rest of this component is run by default if "container_specs" are not specified
        ############################################################################
        # MANAGE INPUTS (see input arguments in function signature, above)
        # --------------------------------------------------------------------------
        # Vertex AI Model Registry expects the parent folder URI.
        gcs_parent_folder = mdk.util.storage.get_parent_path_intelligent(
            trained_model.uri
        )
        # Download and load the performance metrics from the input artifact.
        if scalar_metrics:
            mdk.util.storage.download(scalar_metrics.uri, PM_FILENAME)
            with open(PM_FILENAME, "r") as fin:
                performance_metrics_summary = json.loads(fin.read())
        else:
            performance_metrics_summary = None

        ############################################################################
        # UPLOAD TO MODEL REGISTRY
        # --------------------------------------------------------------------------
        logger.info("Starting model upload and registration process...")
        uploaded_model_resource_name = upload_model(
            config=app_config,
            artifact_folder_uri=gcs_parent_folder,
            performance_metrics_summary=performance_metrics_summary,
            vertex_ai_pipeline_job_run_id=pipeline_job_name,
            access_token=os.environ.get("ID_TOKEN"),  # Gets set if running locally
        )
        logger.info(
            f"Successfully uploaded model. Resource Name: {uploaded_model_resource_name}"
        )

        ############################################################################
        # MANAGE OUTPUTS (see output arguments in function signature, above)
        # --------------------------------------------------------------------------
        uploaded_model.uri = uploaded_model_resource_name
