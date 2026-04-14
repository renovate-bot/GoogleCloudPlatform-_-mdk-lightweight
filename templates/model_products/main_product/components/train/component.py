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

import model_workflow
import mdk.util.framework
import mdk.util.storage
import mdk.config
import mdk.custom_job
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


@dsl.component(target_image=mdk.util.framework.getTargetImage(__file__, "train"))
def train(
    gcp_config_filename: str,
    pipeline_config_filename: str,
    general_config_filename: str,
    train_dataset: dsl.Input[dsl.Dataset],
    test_dataset: dsl.Input[dsl.Dataset],
    hyperparameters: dsl.Input[dsl.Artifact],
    trained_model: dsl.Output[dsl.Model],
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
            "train_dataset": train_dataset.uri,
            "test_dataset": test_dataset.uri,
            "hyperparameters": hyperparameters.uri,
        },
    )
    if custom_job_executed:
        trained_model.uri = f"{base_output_dir}/trained_model"
    else:
        # The rest of this component is run by default if "container_specs" are not specified
        ############################################################################
        # MANAGE INPUTS (see input arguments in function signature, above)
        # --------------------------------------------------------------------------

        train_dataset_uri = train_dataset.uri
        test_dataset_uri = test_dataset.uri

        # Download our train and validation datasets from GCS:
        mdk.util.storage.download(hyperparameters.uri, HYPERPARAMETERS_JSON)
        with open(HYPERPARAMETERS_JSON, "r") as fin:
            hyperparameters_dict = json.loads(fin.read())

        ############################################################################
        # CALCULATION
        # --------------------------------------------------------------------------

        # Train our model.
        model_filename = model_workflow.train.train(
            general_config_filename,
            train_dataset_uri,
            test_dataset_uri,
            hyperparameters_dict,
            gcp_config["deployment_environment"],
        )

        ############################################################################
        # MANAGE OUTPUTS (see output arguments in function signature, above)
        # --------------------------------------------------------------------------

        # Upload our model to GCS.
        # Model Registry will want the file to have a specific filename, so we'll
        #   use the default URL that Kubeflow has generated for us as the name of a
        #   parent folder, and we'll put the model file in that folder.
        trained_model.uri = f"{trained_model.uri}/{model_filename}"
        mdk.util.storage.upload(model_filename, trained_model.uri, mkdir=True)
