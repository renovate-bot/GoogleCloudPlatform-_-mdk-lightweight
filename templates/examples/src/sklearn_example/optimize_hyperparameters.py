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
"""This module provides sample functionality to perform a hyperparameter
optimization search for an xgboost model.
"""

import mdk.config
import mdk.data
import sklearn.model_selection
import sklearn.tree
import google.cloud.bigquery
import logging

logger = logging.getLogger(__name__)


def optimize_hyperparameters(
    general_config_filename: str,
    val_uri: str,
    environment: str,
) -> dict:
    """Example model that trains a dataset.

    Args:
        general_config_filename (str): Filename of config file with model-
            related configuration info.
        val_uri (str): BigQuerye URI of the validation data.
        environment: The environment to use (e.g., 'prod', 'stage', 'train').

    Returns:
        dict: A dict giving a mapping of hyperparameter names to hyperparameter
            values (so that the downstream train method can use those
            hyperparameter values).  Specifically, it contains values for
            "max_depth", "learning_rate", and "subsample".
    """

    # Load our config file.
    general_config = mdk.config.readAndMergeYAMLConfig(
        config_filename=general_config_filename, environment=environment
    )
    training_config = general_config["training"]

    # Load our dataset.
    client = google.cloud.bigquery.Client()
    df_val = mdk.data.getDataframeFromBigQuery(client, val_uri)

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

    # Prepare our independent and dependent variables.
    target_column = training_config["target_column"]
    X_val = df_val.drop(columns=[target_column])
    y_val = df_val[target_column]

    param_grid = {
        # To make this run more quickly, we skip to the "right answer."  To
        #   actually do the hyperparameter optimization, comment this...
        #
        "max_depth": [4,],
        "min_samples_split": [2,],
        #
        # ... and uncomment this:
        #
        # "criterion": ["gini", "entropy"],
        # "max_depth": [4, 6, 8, 10, None],
        # "min_samples_split": [2, 5, 10],
        # "min_samples_leaf": [1, 2, 4],
    }  # fmt: skip

    dt_model = sklearn.tree.DecisionTreeClassifier(**dtree_params)

    grid_search = sklearn.model_selection.GridSearchCV(
        dt_model, param_grid, cv=5, scoring="accuracy", verbose=2
    )

    logger.info("Finding optimal hyperparameters...")
    grid_search.fit(X_val, y_val)

    logger.info(f"Best hyperparameters: {grid_search.best_params_}")
    logger.info(f"Best score: {grid_search.best_score_}")

    ## TO DO: Add functionality to log parameters to vertex ai experiment

    logger.info("Hyperparameter optimization complete.")

    return grid_search.best_params_
