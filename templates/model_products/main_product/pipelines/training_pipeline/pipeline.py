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
import kfp
import logging

logger = logging.getLogger(__name__)


@kfp.dsl.pipeline(name="training_pipeline")
def pipeline(
    environment: str,
):
    """This is pipeline for Kubeflow / Vertex AI Pipelines, comprised of
    a series of Kubeflow components.

    Args:
        environment (str): One of: ("train", "stage" or "prod").  This will be
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
    preprocess = mdk.util.framework.loadComponentSpec(pc_filename, "preprocess")
    hyperopt = mdk.util.framework.loadComponentSpec(pc_filename, "optimize_hyperparameters")  # fmt: skip
    train = mdk.util.framework.loadComponentSpec(pc_filename, "train")
    evaluate = mdk.util.framework.loadComponentSpec(pc_filename, "evaluate")
    upload_to_model_registry = mdk.util.framework.loadComponentSpec(pc_filename, "upload_to_model_registry")  # fmt: skip

    # Execute our component tasks:

    # Kubeflow documentation on inputs and outputs:
    #
    # https://www.kubeflow.org/docs/components/pipelines/user-guides/data-handling/artifacts/
    #
    # (Note that since this is containerized, we use the traditional artifact syntax.)

    log_config_to_experiment_task = log_config_to_experiment(
        gcp_config_filename=gcp_config_filename,
        pipeline_config_filename=pc_filename,
        general_config_filename=general_config_filename,
    )

    preprocess_task = preprocess(
        gcp_config_filename=gcp_config_filename,
        pipeline_config_filename=pc_filename,
        general_config_filename=general_config_filename,
    )

    hyperopt_task = hyperopt(
        gcp_config_filename=gcp_config_filename,
        pipeline_config_filename=pc_filename,
        general_config_filename=general_config_filename,
        val_dataset=preprocess_task.outputs["val_dataset"],
    ).after(preprocess_task)

    train_task = train(
        gcp_config_filename=gcp_config_filename,
        pipeline_config_filename=pc_filename,
        general_config_filename=general_config_filename,
        train_dataset=preprocess_task.outputs["train_dataset"],
        test_dataset=preprocess_task.outputs["test_dataset"],
        hyperparameters=hyperopt_task.outputs["hyperparameters"],
    ).after(hyperopt_task)

    evaluate_task = evaluate(
        gcp_config_filename=gcp_config_filename,
        pipeline_config_filename=pc_filename,
        general_config_filename=general_config_filename,
        trained_model=train_task.outputs["trained_model"],
        test_dataset=preprocess_task.outputs["test_dataset"],
    ).after(train_task)

    upload_to_model_registry_task = upload_to_model_registry(
        gcp_config_filename=gcp_config_filename,
        pipeline_config_filename=pc_filename,
        general_config_filename=general_config_filename,
        pipeline_job_name=kfp.dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
        trained_model=train_task.outputs["trained_model"],
        train_dataset=preprocess_task.outputs["train_dataset"],
        scalar_metrics=evaluate_task.outputs["scalar_metrics"],
    ).after(evaluate_task)

    # Apply the resources to the pipeline tasks if set in the pipeline config
    mdk.util.framework.apply_resource_settings_to_task(
        task_object=log_config_to_experiment_task,
        pipeline_config_filename=pc_filename,
        component_name=log_config_to_experiment_task.name,
    )
    mdk.util.framework.apply_resource_settings_to_task(
        task_object=preprocess_task,
        pipeline_config_filename=pc_filename,
        component_name=preprocess_task.name,
    )
    mdk.util.framework.apply_resource_settings_to_task(
        task_object=hyperopt_task,
        pipeline_config_filename=pc_filename,
        component_name=hyperopt_task.name,
    )
    mdk.util.framework.apply_resource_settings_to_task(
        task_object=train_task,
        pipeline_config_filename=pc_filename,
        component_name=train_task.name,
    )
    mdk.util.framework.apply_resource_settings_to_task(
        task_object=evaluate_task,
        pipeline_config_filename=pc_filename,
        component_name=evaluate_task.name,
    )
    mdk.util.framework.apply_resource_settings_to_task(
        task_object=upload_to_model_registry_task,
        pipeline_config_filename=pc_filename,
        component_name=upload_to_model_registry_task.name,
    )
