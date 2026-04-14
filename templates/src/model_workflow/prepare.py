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

This module is for implementing data prep, including splitting data into train,
validation and test datasets.
"""

import mdk.data
import mdk.config
import collections
import logging

logger = logging.getLogger(__name__)


PrepareOutput = collections.namedtuple(
    "PrepareOutput", ["train_uri", "val_uri", "test_uri"]
)


def prepare(
    general_config_filename: str,
    project_id: str,
    train_dataset_suggested_uri: str,
    val_dataset_suggested_uri: str,
    test_dataset_suggested_uri: str,
    environment: str,
) -> PrepareOutput:
    """Prepares sample data.

    Args:
        general_config_filename (str): Filename of config file with model-
            related configuration info.
        project_id: GCP project ID
        train_dataset_suggested_uri (str): Suggested URI to use for temporary dataset
            file storage, for the train dataset.  (This will be unused if the
            train dataset is a BQ table.)
        val_dataset_suggested_uri (str): Suggested URI to use for temporary dataset
            file storage, for the val dataset.  (This will be unused if the
            val dataset is a BQ table.)
        test_dataset_suggested_uri (str): Suggested URI to use for temporary dataset
            file storage, for the test dataset.  (This will be unused if the
            test dataset is a BQ table.)
        environment: The environment to use (e.g., 'prod', 'stage', 'train').

    Returns:
        PrepareOutput: The train_uri, val_uri and test_uri members give the
            URIs for the train, validation and test datasets, respectively.
    """
    logger.info("Preparing data...")

    # Load and validate our config files.
    general_config = mdk.config.readAndMergeYAMLConfig(
        config_filename=general_config_filename, environment=environment
    )
    training_config = general_config["training"]  # noqa: F841

    # Create handlers to help pass datasets as URIs.
    train_dataset_handler = mdk.data.DatasetHandler(train_dataset_suggested_uri)
    val_dataset_handler = mdk.data.DatasetHandler(val_dataset_suggested_uri)
    test_dataset_handler = mdk.data.DatasetHandler(test_dataset_suggested_uri)

    ############################################################################
    ############################################################################

    # Implement your data preparation logic here:

    # (For an example of how to implement this for an XGBoost model, please see
    #   the following file: src/examples/xgb_example/prepare.py)

    # After you have created your datasets, you can send it to the framework
    #   as follows:

    # Example of how to provide datasets in the form of a BigQuery tables:

    # bq_dataset = training_config["dataset"]
    # train_dataset_handler.set_bigquery_table(project_id, bq_dataset, train_table)
    # val_dataset_handler.set_bigquery_table(project_id, bq_dataset, val_table)
    # test_dataset_handler.set_bigquery_table(project_id, bq_dataset, test_table)

    # Example of how to provide datasets in the form of a flat file (such as
    #   CSV, parquet, or pickle):

    # train_dataset_handler.set_local_file(train_csv_filename)
    # val_dataset_handler.set_local_file(val_csv_filename)
    # test_dataset_handler.set_local_file(test_csv_filename)

    ############################################################################
    ############################################################################

    uris = PrepareOutput(
        train_dataset_handler.uri,
        val_dataset_handler.uri,
        test_dataset_handler.uri,
    )

    return uris
