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
"""This module provides sample functionality to train an SKLearn model."""

import mdk.config
import mdk.data
import sklearn.tree
import google.cloud.bigquery
import logging
from mdk.model import save as save_model

logger = logging.getLogger(__name__)


def train(
    general_config_filename: str,
    train_uri: str,
    test_uri: str,
    hyperparameters: dict | None,
    environment: str,
) -> str:
    """Example model that trains a dataset.

    Args:
        general_config_filename (str): Filename of config file with model-
            related configuration info.
        train_uri (str): BigQuerye URI of the training data.  (This data is what
            is used to perform the fit.)
        test_uri (str): BigQuery URI of the test data.  (The test data is not
            used to train; the test data performance is only used for
            inforomational purposes to print info to the console, and is then
            discarded.)
        hyperparameters (dict): A dict that contains a mapping from
            hyperparameter names to hyperparameter values.  Specifically, it
            contains values for "max_depth", "learning_rate", and "subsample".
        environment: The environment to use (e.g., 'prod', 'stage', 'train').

    Returns:
        str: Filename for a trained SKLearn model.
    """
    # Load our config file.
    general_config = mdk.config.readAndMergeYAMLConfig(
        config_filename=general_config_filename, environment=environment
    )
    training_config = general_config["training"]

    # Load our dataset.
    client = google.cloud.bigquery.Client()
    df_train = mdk.data.getDataframeFromBigQuery(client, train_uri)
    df_test = mdk.data.getDataframeFromBigQuery(client, test_uri)

    logger.info("First row of training dataframe:")
    logger.info(df_train.head(1))

    # SKLearn Decison Tree Model Parameters.
    dtree_params = {
        "criterion": "gini",  # Default: Gini impurity
        "max_depth": None,  # Default: No limit on depth
        "min_samples_split": 2,  # Default: 2 minimum samples to split
        "min_samples_leaf": 1,  # Default: 1 minimum sample per leaf
        "max_features": None,  # Default: Consider all features
        "max_leaf_nodes": None,  # Default: No limit on number of leaves
        "min_impurity_decrease": 0.0,  # Default: No minimum impurity decrease required
        "ccp_alpha": 0.0,  # Default: No pruning by complexity parameter
        "class_weight": None,  # Default: All classes have weight 1
        "splitter": "best",
    }

    # If we have been passed hyperparameters, we incorporate those here:
    if hyperparameters:
        logger.info(f"Using provided hyperparameters: {hyperparameters}")
        dtree_params.update(hyperparameters)

    # Prepare our independent and dependent variables.
    target_column = training_config["target_column"]
    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]
    X_test = df_test.drop(columns=[target_column])
    y_test = df_test[target_column]

    model = sklearn.tree.DecisionTreeClassifier(**dtree_params)
    model.fit(X_train, y_train)

    # Print some performance stats.
    score_train = model.score(X_train, y_train)
    logger.info(f"In-sample (train) score: {score_train}")
    score_test = model.score(X_test, y_test)
    logger.info(f"Out-of-sample (test) score: {score_test}")

    logger.info("Model training complete.")

    # Save our model to file.
    model_filename = training_config["model_filename"]
    output_filename = save_model(model, model_filename)

    return output_filename
