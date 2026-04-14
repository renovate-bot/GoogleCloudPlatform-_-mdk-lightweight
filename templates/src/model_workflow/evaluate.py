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
infrastructure

This module is for implementing model evaluation metrics (against the test
dataset).
"""

import mdk.config
import mdk.data
from mdk.model import load as load_model
import mdk.util.storage
import logging

logger = logging.getLogger(__name__)


def evaluate(
    general_config_filename: str,
    model_filename: str,
    test_dataset_uri: str,
    environment: str,
) -> dict:
    """This function evaluates a model against a test dataset.

    Args:
        general_config_filename (str): Filename of config file with model-
            related configuration info.
        model_filename (str): Filename of the trained model that is
            to be evaluated.
        test_dataset_uri (str): URI of the test dataset against which metrics are to be
            calculated.
        environment: The environment to use (e.g., 'prod', 'stage', 'train').

    Returns:
        dict: A dict giving scalar metric values, indexed by their name.  The
            metric names are free-form.
    """
    logger.info("Running model evaluation...")

    # Load our config file.
    general_config = mdk.config.readAndMergeYAMLConfig(
        config_filename=general_config_filename, environment=environment
    )
    training_config = general_config["training"]  # noqa: F841

    # Load our model.
    model = load_model(model_filename)  # noqa: F841

    ############################################################################
    ############################################################################

    # Example of how to get a BigQuery dataset:

    # client = google.cloud.bigquery.Client()
    # df_test = mdk.data.getDataframeFromBigQuery(client, test_dataset_uri)

    # Example of how to get a dataset in the form of a local file (such as CSV,
    #   parquet, or pickle):

    # test_csv_filename = "test.csv"
    # mdk.util.storage.download(test_dataset_uri, test_csv_filename)
    # df_test = pd.read_csv(test_csv_filename, index_col=0)

    # Implement your evaluation logic here:

    scalars: dict = {}

    # (For an example of how to implement this for an XGBoost model, please see
    #   the following file: src/examples/xgb_example/evaluate.py)

    ############################################################################
    ############################################################################

    return scalars
