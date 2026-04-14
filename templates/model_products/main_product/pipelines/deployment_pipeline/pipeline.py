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

"""This module provides a pipeline for Kubefulow / Vertex AI Pipelines."""

import mdk.util.framework
import kfp.dsl
import logging

logger = logging.getLogger(__name__)


@kfp.dsl.pipeline(name="deployment_pipeline")
def pipeline(
    environment: str,
):
    """This is pipeline for Kubefulow / Vertex AI Pipelines, comprised of
    a series of Kubeflow components.

    Args:
        environment (str): One of: ("train", "stage" or "prod"). This will be
            used to infer the filename of the GCP config file the project uses
            to store the project ID, etc.
    """
    logger.info("Creating tasks...")

    # Infer our config filenames.
    # Get the relative path object.
    here = mdk.util.framework.get_relative_path(__file__)

    # Instantiate the path helper
    paths = mdk.util.framework.PipelinePaths(base=here)

    # Use the helper methods to get the final strings
    gcp_config_filename = paths.get_gcp_config(environment)
    general_config_filename = paths.get_general_config()
    pc_filename = paths.get_pipeline_config()

    # Load our component tasks from file:
    log_config_to_experiment = mdk.util.framework.loadComponentSpec(pc_filename, "log_config_to_experiment")  # fmt: skip
    model_deployment = mdk.util.framework.loadComponentSpec(pc_filename, "model_deployment")  # fmt: skip
    model_monitor = mdk.util.framework.loadComponentSpec(pc_filename, "model_monitoring")  # fmt: skip

    # Execute our component tasks:

    # Kubeflow documentation on inputs and outputs:
    # (Note that since this is containerized, we use the traditional artifact syntax.)
    #
    # https://www.kubeflow.org/docs/components/pipelines/user-guides/data-handling/artifacts/

    log_config_to_experiment_task = log_config_to_experiment(
        gcp_config_filename=gcp_config_filename,
        pipeline_config_filename=pc_filename,
        general_config_filename=general_config_filename,
    )

    deploy_task = model_deployment(
        gcp_config_filename=gcp_config_filename,
        pipeline_config_filename=pc_filename,
        general_config_filename=general_config_filename,
    )

    with kfp.dsl.Condition(
        deploy_task.outputs["run_monitoring"] == True,  # noqa: E712
        name="run-model-monitoring",
    ):
        model_monitor_task = model_monitor(
            gcp_config_filename=gcp_config_filename,
            pipeline_config_filename=pc_filename,
            general_config_filename=general_config_filename,
            target_endpoint=deploy_task.outputs["deployed_endpoint"],
        ).after(deploy_task)

    # Apply the resources to the pipeline tasks if set in the pipeline config
    mdk.util.framework.apply_resource_settings_to_task(
        task_object=log_config_to_experiment_task,
        pipeline_config_filename=pc_filename,
        component_name=log_config_to_experiment_task.name,
    )
    mdk.util.framework.apply_resource_settings_to_task(
        task_object=deploy_task,
        pipeline_config_filename=pc_filename,
        component_name=deploy_task.name,
    )
    mdk.util.framework.apply_resource_settings_to_task(
        task_object=model_monitor_task,
        pipeline_config_filename=pc_filename,
        component_name=model_monitor_task.name,
    )
