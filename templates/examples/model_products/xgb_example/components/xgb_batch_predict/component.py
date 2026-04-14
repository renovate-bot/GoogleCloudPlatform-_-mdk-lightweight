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

import sys

sys.path.append("examples/src")
import xgb_example
import mdk.util.framework
import mdk.config
import google.cloud.aiplatform
import mdk.custom_job
from kfp import dsl
import datetime
import inspect
import os


# Kubeflow documentation on inputs and outputs:
#
# https://www.kubeflow.org/docs/components/pipelines/user-guides/data-handling/artifacts/


@dsl.component(
    target_image=mdk.util.framework.getTargetImage(__file__, "xgb_batch_predict")
)
def xgb_batch_predict(
    gcp_config_filename: str,
    pipeline_config_filename: str,
    general_config_filename: str,
    batch_predictions_table: dsl.Output[dsl.Dataset],
    run_monitoring: dsl.OutputPath(str),
):
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
    inference_config = general_config["inference"]

    # Set our project ID.
    google.cloud.aiplatform.init(project=gcp_config.get("project_id"))  # fmt: skip
    os.environ["GOOGLE_CLOUD_PROJECT"] = gcp_config.get("project_id")

    run_monitoring_flag = inference_config.get("run_monitoring", False)

    # Optional: handle custom job execution if specified.
    staging_bucket = f"{gcp_config.get('pipeline_staging_dir')}/custom_job/staging"
    base_output_dir = f"{gcp_config.get('pipeline_staging_dir')}/custom_job/{timestamp}"
    custom_job_executed = mdk.custom_job.handle_custom_job_if_configured(
        gcp_config=gcp_config,
        model_config=inference_config,
        pipeline_config=pipeline_config,
        component_name=inspect.currentframe().f_code.co_name,
        staging_bucket=staging_bucket,
        base_output_dir=base_output_dir,
    )
    if custom_job_executed:
        run_monitoring.uri = f"{base_output_dir}/run_monitoring"
        batch_predictions_table.uri = f"{base_output_dir}/batch_predictions_table"
    else:
        # The rest of this component is run by default if "container_specs" are not specified
        ############################################################################
        # CALCULATION
        # --------------------------------------------------------------------------

        # Generate predictions using XGBoost model.
        outputs = xgb_example.batch_predict.batch_prediction(
            general_config_filename=general_config_filename,
            gcp_config_filename=gcp_config_filename,
            environment=gcp_config["deployment_environment"],
            access_token=os.environ.get("ID_TOKEN"),  # Gets set if running locally
        )

        ############################################################################
        # MANAGE OUTPUTS (see output arguments in function signature, above)
        # --------------------------------------------------------------------------

        batch_predictions_table.uri = outputs
        with open(run_monitoring, "w") as f:
            f.write(str(run_monitoring_flag).lower())
