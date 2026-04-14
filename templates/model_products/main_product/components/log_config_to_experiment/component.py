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

import mdk.config
import mdk.custom_job
import mdk.util.framework
import google.cloud.aiplatform
from kfp import dsl
import os
import datetime
import inspect
import logging
from typing import Dict, Any


# Kubeflow documentation on inputs and outputs:
#
# https://www.kubeflow.org/docs/components/pipelines/user-guides/data-handling/artifacts/

logger = logging.getLogger(__name__)


def flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "_"
) -> Dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


@dsl.component(
    target_image=mdk.util.framework.getTargetImage(__file__, "log_config_to_experiment")
)
def log_config_to_experiment(
    gcp_config_filename: str,
    pipeline_config_filename: str,
    general_config_filename: str,
    config_metrics: dsl.Output[dsl.Metrics],
):
    """
    Loads the YAML configuration files and writes their contents to the Metrics
    output for UI visibility.
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
    training_config = general_config["training"]

    # Set our project ID.
    google.cloud.aiplatform.init(project=gcp_config["project_id"])  # fmt: skip
    os.environ["GOOGLE_CLOUD_PROJECT"] = gcp_config["project_id"]

    # Optional: handle custom job execution if specified.
    staging_bucket = f"{gcp_config['pipeline_staging_dir']}/custom_job/staging"
    base_output_dir = f"{gcp_config['pipeline_staging_dir']}/custom_job/{timestamp}"
    custom_job_executed = mdk.custom_job.handle_custom_job_if_configured(
        gcp_config=gcp_config,
        model_config=training_config,
        pipeline_config=pipeline_config,
        component_name=inspect.currentframe().f_code.co_name,
        staging_bucket=staging_bucket,
        base_output_dir=base_output_dir,
    )
    if custom_job_executed:
        config_metrics.uri = f"{base_output_dir}/config_metrics"
    else:
        # The rest of this component is run by default if "container_specs" are not specified
        ############################################################################
        # MANAGE INPUTS (see input arguments in function signature, above)
        # --------------------------------------------------------------------------

        all_params: dict = {}
        all_params.update(gcp_config)
        all_params.update(general_config)
        all_params = flatten_dict(all_params)

        ############################################################################
        # MANAGE OUTPUTS (see output arguments in function signature, above)
        # --------------------------------------------------------------------------

        for key, value in all_params.items():
            config_metrics.log_metric(f"param_{key}", value)

        # Return the URI to VAI Experiments.
        config_metrics.uri = (
            f"https://console.cloud.google.com/vertex-ai/experiments/locations"
            f"/{gcp_config['region']}/experiments/{gcp_config['experiment_name']}"
        )

        logger.info(
            f"Successfully logged {len(all_params)} configuration values as metrics."
        )
