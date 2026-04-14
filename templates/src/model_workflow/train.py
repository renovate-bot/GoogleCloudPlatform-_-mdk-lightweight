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

This module is for implementing model training logic.
"""

import mdk.data
import mdk.config
from mdk.model import save as save_model
import mdk.util.storage
import logging

logger = logging.getLogger(__name__)


def train(
    general_config_filename: str,
    train_dataset_uri: str,
    test_dataset_uri: str,
    hyperparameters: dict | None,
    environment: str,
) -> str:
    """Example model that trains a dataset.

    Please note: The train dataset is for in-sample fitting.  The test dataset
    is only provided for displaying out-of-sample performance only, so that it
    does not affect the fit.

    Args:
        general_config_filename (str): Filename of config file with model-
            related configuration info.
        train_dataset_uri (str): URI for the training dataset.  (This data is what is
            used to perform the fit.)
        test_dataset_uri (str): URI for the test dataset.  (The test data is not used to
            train; the test data is only used for inforomational purposes to
            print performance info to the console.)
        hyperparameters (dict): A dict that contains a mapping from
            hyperparameter names to hyperparameter values.
        environment: The environment to use (e.g., 'prod', 'stage', 'train').

    Returns:
        str: Path to serialized trained XGBoost classifier model.
    """
    logger.info("Training model")

    # Load our config file.
    general_config = mdk.config.readAndMergeYAMLConfig(
        config_filename=general_config_filename, environment=environment
    )
    training_config = general_config["training"]  # noqa: F841
    model_filename = training_config["model_filename"]

    ############################################################################
    ############################################################################

    # Load our datasets.

    # Example of how to get BigQuery datasets:

    # client = google.cloud.bigquery.Client()
    # df_train = mdk.data.getDataframeFromBigQuery(client, train_dataset_uri)
    # df_test = mdk.data.getDataframeFromBigQuery(client, test_dataset_uri)

    # Example of how to get datasets in the form of a local file (such as CSV,
    #   parquet, or pickle):

    # train_csv_filename = "train.csv"
    # test_csv_filename = "test.csv"
    # mdk.util.storage.download(train_dataset_uri, train_csv_filename)
    # mdk.util.storage.download(test_dataset_uri, test_csv_filename)
    # df_train = pd.read_csv(train_csv_filename, index_col=0)
    # df_test = pd.read_csv(test_csv_filename, index_col=0)

    # Implement your model training logic here:

    model = None

    # (For an example of how to implement this for an XGBoost model, please see
    #   the following file: src/examples/xgb_example/train.py)

    ############################################################################
    ############################################################################

    # Save our model.
    model_path = save_model(model, model_filename)
    return model_path
