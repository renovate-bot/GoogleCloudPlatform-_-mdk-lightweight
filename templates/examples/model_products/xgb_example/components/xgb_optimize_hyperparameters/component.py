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
import mdk.config
import mdk.custom_job
import mdk.util.framework
import mdk.util.storage
import google.cloud.aiplatform
from kfp import dsl
import json
import logging
import os
import datetime
import inspect

logger = logging.getLogger(__name__)

HYPERPARAMETERS_JSON = "hyperparameters.json"


# Kubeflow documentation on inputs and outputs:
#
# https://www.kubeflow.org/docs/components/pipelines/user-guides/data-handling/artifacts/


@dsl.component(target_image=mdk.util.framework.getTargetImage(__file__, "xgb_optimize_hyperparameters"))  # fmt: skip
def xgb_optimize_hyperparameters(
    gcp_config_filename: str,
    pipeline_config_filename: str,
    general_config_filename: str,
    val_dataset: dsl.Input[dsl.Dataset],
    hyperparameters: dsl.Output[dsl.Artifact],
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
    training_config = general_config["training"]

    # Set our project ID.
    google.cloud.aiplatform.init(project=gcp_config.get("project_id"))  # fmt: skip
    os.environ["GOOGLE_CLOUD_PROJECT"] = gcp_config.get("project_id")

    # Optional: handle custom job execution if specified.
    staging_bucket = f"{gcp_config.get('pipeline_staging_dir')}/custom_job/staging"
    base_output_dir = f"{gcp_config.get('pipeline_staging_dir')}/custom_job/{timestamp}"
    custom_job_executed = mdk.custom_job.handle_custom_job_if_configured(
        gcp_config=gcp_config,
        model_config=training_config,
        pipeline_config=pipeline_config,
        component_name=inspect.currentframe().f_code.co_name,
        staging_bucket=staging_bucket,
        base_output_dir=base_output_dir,
        input_artifact_map={
            "val_dataset": val_dataset.uri,
        },
    )
    if custom_job_executed:
        hyperparameters.uri = f"{base_output_dir}/hyperparameters"
    else:
        # The rest of this component is run by default if "container_specs" are not specified
        ############################################################################
        # MANAGE INPUTS (see input arguments in function signature, above)
        # --------------------------------------------------------------------------

        val_uri = val_dataset.uri

        ############################################################################
        # CALCULATION
        # --------------------------------------------------------------------------

        # Find optimal hyperparameters.
        hyperparameters_dict = (
            xgb_example.optimize_hyperparameters.optimize_hyperparameters(
                general_config_filename, val_uri, gcp_config["deployment_environment"]
            )
        )

        ############################################################################
        # MANAGE OUTPUTS (see output arguments in function signature, above)
        # --------------------------------------------------------------------------

        # Save our model and upload it to GCS.
        with open(HYPERPARAMETERS_JSON, "w") as fout:
            fout.write(json.dumps(hyperparameters_dict))

        mdk.util.storage.upload(HYPERPARAMETERS_JSON, hyperparameters.uri, mkdir=True)
