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

from mdk.model.deployment import deploy_model, DeploymentAppConfig
import mdk.util.framework
import mdk.config
import mdk.custom_job

import google.cloud.aiplatform
from kfp import dsl
import datetime
import inspect
import logging
import os

logger = logging.getLogger(__name__)


@dsl.component(
    target_image=mdk.util.framework.getTargetImage(__file__, "model_deployment")
)
def model_deployment(
    gcp_config_filename: str,
    pipeline_config_filename: str,
    general_config_filename: str,
    deployed_endpoint: dsl.Output[dsl.Artifact],
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
    model_deployment_config = general_config["deployment"]

    app_config = DeploymentAppConfig.from_yaml_files(
        gcp_config_path=gcp_config_filename,
        general_config_path=general_config_filename,
    )
    # Read the monitoring flag from the deployment configuration to decide
    # whether to setup model monitoring.
    run_monitoring_flag = model_deployment_config.get("run_monitoring", False)

    # Set our project ID.
    google.cloud.aiplatform.init(project=gcp_config.get("project_id"))  # fmt: skip
    os.environ["GOOGLE_CLOUD_PROJECT"] = gcp_config.get("project_id")

    # Optional: handle custom job execution if specified.
    staging_bucket = f"{gcp_config.get('pipeline_staging_dir')}/custom_job/staging"
    base_output_dir = f"{gcp_config.get('pipeline_staging_dir')}/custom_job/{timestamp}"
    custom_job_executed = mdk.custom_job.handle_custom_job_if_configured(
        gcp_config=gcp_config,
        model_config=model_deployment_config,
        pipeline_config=pipeline_config,
        component_name=inspect.currentframe().f_code.co_name,
        staging_bucket=staging_bucket,
        base_output_dir=base_output_dir,
    )
    if custom_job_executed:
        deployed_endpoint.uri = f"{base_output_dir}/deployed_endpoint"
    else:
        # The rest of this component is run by default if "container_specs" are not specified
        ############################################################################
        # MANAGE INPUTS (see input arguments in function signature, above)
        # --------------------------------------------------------------------------

        vertex_ai_endpoint_resource_name = deploy_model(
            config=app_config,
            access_token=os.environ.get("ID_TOKEN"),  # Gets set if running locally)
        )

        ############################################################################
        # MANAGE OUTPUTS (see output arguments in function signature, above)
        # --------------------------------------------------------------------------

        # Set our component output.
        deployed_endpoint.uri = vertex_ai_endpoint_resource_name

    with open(run_monitoring, "w") as f:
        f.write(str(run_monitoring_flag).lower())
