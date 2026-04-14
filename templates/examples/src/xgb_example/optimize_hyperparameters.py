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

import mdk.data
import mdk.config
import sklearn.model_selection
import xgboost as xgb
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
        val_uri (str): BigQuery URI of the validation data.
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

    # XGBoost Model Parameters (these are examples, tune them for your data)
    xgb_params = {
        "objective": "multi:softprob",  # For multiclass classification
        "eval_metric": "mlogloss",  # Evaluation metric
        # "eta": 0.05,  # Learning rate - we will optimize this
        # "max_depth": 6,  # Maximum depth of a tree - we will optimize this
        # "subsample": 0.8,  # Subsample ratio of the training instance - we will optimize this
        "colsample_bytree": 0.8,  # Subsample ratio of columns when constructing each tree
        "random_state": 42,
        "booster": "gbtree",
        # "boosting_rounds": 200, # evidently not used
        "nthread": -1,  # Use all available threads
        "gamma": 0.1,
        "lambda": 1,  # L2 regularization
        "alpha": 0.5,  # L1 regularization
    }

    # Infer how many classes we have from the data.
    target_column = training_config["target_column"]
    xgb_params["num_class"] = len(df_val[target_column].unique())

    # Convert our data to the expected structure.

    X_val = df_val.drop(columns=[target_column])
    y_val = df_val[target_column]

    param_grid = {
        # To make this run more quickly, we skip to the "right answer."  To
        #   actually do the hyperparameter optimization, comment this...
        #
        "max_depth": [4,],
        "learning_rate": [0.1,],  # eta
        "subsample": [0.6,],
        #
        # ... and uncomment this:
        #
        # "max_depth": [2, 4, 6],
        # "learning_rate": [0.2, 0.1, 0.05],  # eta
        # "subsample": [0.4, 0.6, 0.8],
    }  # fmt: skip

    xgb_model = xgb.XGBClassifier(**xgb_params)

    grid_search = sklearn.model_selection.GridSearchCV(
        xgb_model, param_grid, cv=5, scoring="accuracy", verbose=2
    )

    logger.info("Finding optimal hyperparameters...")
    grid_search.fit(X_val, y_val)

    logger.info(f"Best hyperparameters: {grid_search.best_params_}")
    logger.info(f"Best score: {grid_search.best_score_}")

    logger.info("Hyperparameter optimization complete.")

    return grid_search.best_params_
