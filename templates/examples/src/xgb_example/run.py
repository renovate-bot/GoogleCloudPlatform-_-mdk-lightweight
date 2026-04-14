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

"""This provides a convenience function to prepare, optimize, fit and save an
XGBoost model locally, without invoking a pipeline.
"""

import sys

sys.path.append("examples/src")
import xgb_example
import mdk.config
import mdk.util.framework
import google.cloud.aiplatform
import logging
import pickle
import pprint

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="{levelname}:{asctime}:{name}:{lineno}:{message}",
    level=logging.DEBUG,
    style="{",
)


def run(
    gcp_config_filename: str,
    general_config_filename: str,
):
    """This provides a mechanism to prepare, optimize, fit and save an
    XGBoost model locally, without invoking a pipeline

    Args:
        gcp_config_filename (str): Filename of config file with GCP-related
            configuration info such as the GCP project ID.
        general_config_filename (str): Filename of config file with model-
            related configuration info.
    """
    # Call aiplatform.init()
    gcp_config = mdk.config.GCPConfig.from_yaml_file(gcp_config_filename).model_dump()
    project_id = gcp_config.get("project_id")
    google.cloud.aiplatform.init(project=project_id)
    environment = gcp_config["deployment_environment"]
    general_config = mdk.config.readAndMergeYAMLConfig(
        config_filename=general_config_filename, environment=environment
    )

    # Get our data:
    data_uris = xgb_example.prepare.prepare(
        general_config_filename, project_id, environment
    )

    # Optimize hyperparamters.
    hyperparameters = xgb_example.optimize_hyperparameters.optimize_hyperparameters(
        general_config_filename, data_uris.val_uri, environment
    )

    # Train our model:
    model = xgb_example.train.train(
        general_config_filename,
        data_uris.train_uri,
        data_uris.test_uri,
        hyperparameters,
        environment,
    )

    # Evaluate our model:
    eval = xgb_example.evaluate.evaluate(
        general_config_filename, model, data_uris.test_uri, environment
    )

    logger.info(f"Scalar metrics:\n{pprint.pformat(eval.scalars)}")

    # Write our model to file.
    with open(general_config["model_filename"], "wb") as fout:
        pickle.dump(model, fout)


if __name__ == "__main__":
    # Infer the names of our config files.
    pipeline_mapping = mdk.util.framework.get_pipeline_mapping()
    pipeline_module_file = pipeline_mapping["xgb_training_pipeline"]
    parent_dir = pipeline_module_file.parent.parent
    gcp_config_filename = parent_dir / "state/train.yml"
    general_config_filename = parent_dir / "config/config.yml"

    # Run the steps of our modeling process.
    run(
        gcp_config_filename=gcp_config_filename,
        general_config_filename=general_config_filename,
    )
