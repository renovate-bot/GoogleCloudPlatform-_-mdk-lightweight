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

This module is for implementing hyperparameter optimization.
"""

import mdk.data
import mdk.config
import mdk.util.storage
import logging

logger = logging.getLogger(__name__)


def optimize_hyperparameters(
    general_config_filename: str,
    val_dataset_uri: str,
    environment: str,
) -> dict:
    """Example model that trains a dataset.

    Args:
        general_config_filename (str): Filename of config file with model-
            related configuration info.
        val_dataset_uri (str): URI of the validation dataset.
        environment: The environment to use (e.g., 'prod', 'stage', 'train').

    Returns:
        dict: A dict giving a mapping of hyperparameter names to hyperparameter
            values (so that the downstream train method can use those
            hyperparameter values).
    """
    logger.info("Optimizing hyperparameters")

    # Load our config file.
    general_config = mdk.config.readAndMergeYAMLConfig(
        config_filename=general_config_filename, environment=environment
    )
    training_config = general_config["training"]  # noqa: F841

    ############################################################################
    ############################################################################

    # Load our dataset.

    # Example of how to get a BigQuery dataset:

    # client = google.cloud.bigquery.Client()
    # df_val = mdk.data.getDataframeFromBigQuery(client, val_dataset_uri)

    # Example of how to get datasets in the form of a local file (such as CSV,
    #   parquet, or pickle):

    # val_csv_filename = "val.csv"
    # mdk.util.storage.download(val_dataset_uri, val_csv_filename)
    # df_val = pd.read_csv(val_csv_filename, index_col=0)

    # Implement your hyperparameter optimization logic here:

    hyperparameters: dict = {}

    # (For an example of how to implement this for an XGBoost model, please see
    #   the following file: src/examples/xgb_example/optimize_hyperparameters.py)

    ############################################################################
    ############################################################################

    logger.info("Hyperparameter optimization complete.")

    return hyperparameters
