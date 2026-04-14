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

"""This serves as a template for new projects, so that data scientists can fill
in the below methods for the purpose of getting up and running with MLOps
infrastructure.

This provides a convenience function to prepare, optimize, fit and save a model
locally, without invoking a pipeline.
"""

import model_workflow
import mdk.config
import mdk.util.framework
import google.cloud.aiplatform
import logging
import pickle
import pprint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run(
    gcp_config_filename: str,
    general_config_filename: str,
):
    """This provides a mechanism to prepare, optimize, fit and save a model
    locally, without invoking a pipeline.

    Args:
        gcp_config_filename (str): Filename of config file with GCP-related
            configuration info such as the GCP project ID.
        general_config_filename (str): Filename of config file with model-
            related configuration info.
    """
    logger.info("Evaluating model")

    # Call aiplatform.init()
    gcp_config = mdk.config.GCPConfig.from_yaml_file(gcp_config_filename).model_dump()
    project_id = gcp_config.get("project_id")
    google.cloud.aiplatform.init(project=project_id)
    environment = gcp_config["deployment_environment"]
    general_config = mdk.config.readAndMergeYAMLConfig(
        config_filename=general_config_filename, environment=environment
    )

    # Get our data:
    train_dataset_uri = "tmp_train"
    val_dataset_uri = "tmp_val"
    test_dataset_uri = "tmp_test"
    uris = model_workflow.prepare.prepare(
        general_config_filename,
        project_id,
        train_dataset_uri,
        val_dataset_uri,
        test_dataset_uri,
        environment,
    )

    # Optimize hyperparamters:
    hyperparameters = model_workflow.optimize_hyperparameters.optimize_hyperparameters(
        general_config_filename, uris.val_uri, environment
    )

    # Train our model:
    model = model_workflow.train.train(
        general_config_filename,
        uris.train_uri,
        uris.test_uri,
        hyperparameters,
        environment,
    )

    # Evaluate our model:
    scalars = model_workflow.evaluate.evaluate(
        general_config_filename, model, uris.test_uri, environment
    )
    logger.info(f"Scalar metrics:\n{pprint.pformat(scalars)}")

    # Write our model to file.
    training_config = general_config["training"]
    with open(training_config["model_filename"], "wb") as fout:
        pickle.dump(model, fout)


if __name__ == "__main__":
    # Infer the names of our config files.
    pipeline_mapping = mdk.util.framework.get_pipeline_mapping()
    pipeline_module_file = pipeline_mapping["training_pipeline"]
    parent_dir = pipeline_module_file.parent.parent
    gcp_config_filename = parent_dir / "state/train.yml"
    general_config_filename = parent_dir / "config/config.yml"

    # Run the steps of our modeling process.
    run(
        gcp_config_filename=gcp_config_filename,
        general_config_filename=general_config_filename,
    )
