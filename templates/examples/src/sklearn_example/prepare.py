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

This module is for implementing data preprocessing.
"""

import mdk.data
import mdk.config
import pandas_gbq
import sklearn.model_selection
import collections
import logging

logger = logging.getLogger(__name__)

RANDOM_STATE = 0

PrepareOutput = collections.namedtuple(
    "PrepareOutput", ["train_uri", "val_uri", "test_uri"]
)


def prepare(
    general_config_filename: str,
    project_id: str,
    environment: str,
) -> PrepareOutput:
    """Prepares sample data, querying from Vertex AI Feature Store.

    Args:
        general_config_filename (str): Filename of config file with model-
            related configuration info.
        project_id: GCP project ID
        environment: The environment to use (e.g., 'prod', 'stage', 'train').

    Returns:
        PrepareOutput: The train_uri, val_uri and test_uri members give the
            URIs of the BigQuery tables for the train, validation and test data,
            respectively.
    """
    logger.info("Preparing data...")

    # Load and validate our config files.
    general_config = mdk.config.readAndMergeYAMLConfig(
        config_filename=general_config_filename, environment=environment
    )
    training_config = general_config["training"]
    fs_region = training_config["feature_store"]["region"]
    fs_feature_group_id = training_config["feature_store"]["feature_group_id"]
    fs_read_instances_uri = training_config["feature_store"]["read_instances_uri"]

    # Load parameters for the data splitting process.
    test_size = training_config.get("test_size", 0.2)
    val_size = training_config.get("val_size", 0.2)
    target_col = training_config["target_column"]

    logger.info("Fetching data from feature store...")
    df = mdk.data.getDataFrameFromFeatureStore(
        fs_feature_group_id, fs_read_instances_uri, project_id, fs_region
    )
    # --- Data Preprocessing and Splitting ---

    logger.info("Preprocessing and splitting data...")

    # Drop metadata columns that are not needed for training.
    df = df.drop(columns=["entity_id", "feature_timestamp"], errors="ignore")

    # Map string-based class labels to integer values for model training.
    class_names = sorted(set(df[target_col].unique()))
    mapping = {name: i for i, name in enumerate(class_names)}
    df[target_col] = df[target_col].map(mapping)

    # First split: Separate the test set from the training and validation sets.
    train_plus_val_df, test_df = sklearn.model_selection.train_test_split(
        df, test_size=test_size, random_state=RANDOM_STATE, stratify=df[target_col]
    )

    # Calculate the relative size of the validation set compared to the combined
    # training+validation set.
    train_plus_val = 1.0 - test_size
    relative_val_size = val_size / train_plus_val

    # Second split: Separate the validation set from the training set.
    train_df, val_df = sklearn.model_selection.train_test_split(
        train_plus_val_df,
        test_size=relative_val_size,
        random_state=RANDOM_STATE,
        stratify=train_plus_val_df[target_col],
    )

    # --- Save Processed Data to BigQuery ---

    dataset = training_config["dataset"]
    train_table_id = f"{project_id}.{dataset}.train_skl"
    test_table_id = f"{project_id}.{dataset}.test_skl"
    val_table_id = f"{project_id}.{dataset}.val_skl"

    datasets_to_upload = [
        (train_df, train_table_id),
        (test_df, test_table_id),
        (val_df, val_table_id),
    ]

    for df_to_upload, destination_table in datasets_to_upload:
        logger.info(f"Uploading to: {destination_table}...")
        pandas_gbq.to_gbq(
            dataframe=df_to_upload,
            destination_table=destination_table,
            project_id=project_id,
            if_exists="replace",
        )

    logger.info("Successfully saved all DataFrames to BigQuery.")

    uris = PrepareOutput(
        f"bq://{train_table_id}",
        f"bq://{test_table_id}",
        f"bq://{val_table_id}",
    )

    return uris
