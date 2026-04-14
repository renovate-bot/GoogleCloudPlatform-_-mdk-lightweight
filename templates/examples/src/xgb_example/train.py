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
"""This module provides sample functionality to train an XGBoost model."""

import mdk.data
import mdk.config
import xgboost as xgb
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
        train_uri (str): BigQuery of the training data.  (This data is what is
            used to perform the fit.)
        test_uri (str): BigQuery URI of the test data.  (The test data is not
            used to train; the test data performance is only used for
            inforomational purposes to print info to the console, and is then
            discarded.)
        hyperparameters (dict): A dict that contains a mapping from
            hyperparameter names to hyperparameter values.  Specifically, it
            contains values for "max_depth", "learning_rate", and "subsample".
        environment: The environment to use (e.g., 'prod', 'stage', 'train').

    Returns:
        xgboost.Booster: A trained XGBoost model.
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

    # XGBoost Model Parameters (These are examples; tune them for your problem)
    xgb_params = {
        "objective": "multi:softprob",  # For multiclass classification
        "eval_metric": "mlogloss",  # Evaluation metric
        "eta": 0.05,  # Learning rate
        "max_depth": 6,  # Maximum depth of a tree
        "subsample": 0.8,  # Subsample ratio of the training instance
        "colsample_bytree": 0.8,  # Subsample ratio of columns when constructing each tree
        "random_state": 42,
        "booster": "gbtree",
        "boosting_rounds": 200,
        "nthread": -1,  # Use all available threads
        "gamma": 0.1,
        "lambda": 1,  # L2 regularization
        "alpha": 0.5,  # L1 regularization
    }

    # If we have been passed hyperparameters, we incorporate those here:
    if hyperparameters:
        logger.info(f"Using provided hyperparameters: {hyperparameters}")
        xgb_params.update(hyperparameters)

    # Infer how many classes we have from the data.
    target_column = training_config["target_column"]
    xgb_params["num_class"] = len(df_train[target_column].unique())

    # Convert to DMatrix format (make sure they are in numpy format)
    X_train = df_train.drop(columns=[target_column]).to_numpy()
    y_train = df_train[target_column].to_numpy()
    X_test = df_test.drop(columns=[target_column]).to_numpy()
    y_test = df_test[target_column].to_numpy()

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=None)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=None)
    logger.info("Successfully converted data to DMatrix format.")

    num_boost_round = xgb_params.pop("boosting_rounds", 100)
    early_stopping_rounds = xgb_params.pop("early_stopping_rounds", None)

    model = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dtest, "test")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=10,  # Print evaluation metrics every 10 rounds
    )

    # Only show best iteration if using early stoping
    if early_stopping_rounds:
        logger.info(f"\nTraining completed. Best iteration: {model.best_iteration}")
        logger.info(f"Best validation score: {model.best_score}")

    logger.info("Model training complete.")

    # Save our model.
    model_filename = training_config["model_filename"]
    output_filename = save_model(model=model, filename=model_filename)
    logging.info(f"Model saved to: {output_filename}")

    return output_filename
