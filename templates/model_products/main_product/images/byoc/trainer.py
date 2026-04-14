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

"""This module provides sample functionality to train a TensorFlow model."""

import argparse
import logging
import os
import sys

from tensorflow.keras import layers, models, optimizers
import google.cloud.bigquery


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)


def getDataframeFromBigQuery(
    client,
    uri,
):
    """Given a URI beginning with bq://, select all rows from this table and
    return it as a dataframe.

    Args:
        client (google.cloud.bigquery.Client):
        uri (str):

    Returns:
        pandas.DataFrame:
    """
    SCHEME = "bq://"
    if not uri.startswith(SCHEME):
        raise ValueError(f"Expected uri to begin with {SCHEME}")
    table = uri.removeprefix(SCHEME)

    sql = f"select * from `{table}`"
    logging.info(f"Running sql query: {sql}")
    df = client.query_and_wait(sql).to_dataframe()

    return df


def train(
    project: str,
    target_column: str,
    train_uri: str,
    test_uri: str,
    hyperparameters: dict | None,
    model_output_gcs_uri: str,
    verbose: bool = False,
):
    """Example model that trains a dataset using TensorFlow.

    Args:
        project (str): The project ID.
        target_column (str): The name of the column to predict.
        train_uri (str): BigQuery URI of the training data.
        test_uri (str): BigQuery URI of the test data.
        hyperparameters (dict): A dict that contains a mapping from
            hyperparameter names to hyperparameter values. For TensorFlow,
            it can include "epochs", "learning_rate", "batch_size", etc.
        model_output_gcs_uri (str): The GCS location that the model should be saved to.
        verbose: Verbose flag for training.
    """
    # Load our dataset from BigQuery.
    # Project must be set or it will attempt to run in a tenant project (e.g sed028714119aa78c-tp)
    client = google.cloud.bigquery.Client(project=project)
    logging.info(f"Loading training data from {train_uri}")
    df_train = getDataframeFromBigQuery(client, train_uri)
    logging.info(f"Loading test data from {test_uri}")
    df_test = getDataframeFromBigQuery(client, test_uri)

    logging.info(f"First row of training dataframe:\n{df_train.head(1)}")
    logging.info(f"First row of test dataframe:\n{df_test.head(1)}")

    # TensorFlow Model Parameters (Defaults, will be overridden by hyperparameters)
    tf_params = {
        "epochs": 10,
        "learning_rate": 0.001,
        "batch_size": 32,
        "optimizer": "Adam",
        "loss": "sparse_categorical_crossentropy",
        "metrics": ["accuracy"],
    }

    # If we have been passed hyperparameters, we incorporate those here:
    if hyperparameters:
        logging.info(f"Using provided hyperparameters: {hyperparameters}")
        tf_params.update(hyperparameters)

    # Infer feature columns
    feature_columns = [col for col in df_train.columns if col != target_column]
    num_classes = len(df_train[target_column].unique())

    logging.info(f"Detected {num_classes} classes for target column '{target_column}'.")
    logging.info(f"Using feature columns: {feature_columns}")

    # Prepare data for TensorFlow (assuming numerical features)
    X_train = df_train[feature_columns].to_numpy().astype("float32")
    y_train = (
        df_train[target_column].to_numpy().astype("int")
    )  # Ensure labels are integers
    X_test = df_test[feature_columns].to_numpy().astype("float32")
    y_test = df_test[target_column].to_numpy().astype("int")

    # Determine input shape for the Keras model
    input_shape = (X_train.shape[1],)
    logging.info(f"Input shape for model: {input_shape}")

    # Define a simple TensorFlow Keras Sequential model
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),  # Explicit Input layer
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(
                num_classes, activation="softmax"
            ),  # Output layer for multi-class classification
        ]
    )

    # Choose and configure optimizer
    optimizer_name = tf_params["optimizer"].lower()
    learning_rate = tf_params["learning_rate"]
    if optimizer_name == "adam":
        optimizer = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = optimizers.SGD(learning_rate=learning_rate)
    else:
        logging.warning(
            f"Unsupported optimizer '{optimizer_name}'. Defaulting to Adam."
        )
        optimizer = optimizers.Adam(learning_rate=learning_rate)

    # Compile the model
    model.compile(
        optimizer=optimizer, loss=tf_params["loss"], metrics=tf_params["metrics"]
    )
    model.summary()
    logging.info("TensorFlow model compiled.")

    # Train the model
    logging.info(
        f"Starting model training for {tf_params['epochs']} epochs with batch size {tf_params['batch_size']}..."
    )
    model.fit(
        X_train,
        y_train,
        epochs=tf_params["epochs"],
        batch_size=tf_params["batch_size"],
        validation_data=(X_test, y_test),
        verbose=verbose,
    )
    logging.info("Model training complete.")

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test, verbose=verbose)
    logging.info(f"Test Loss: {loss:.4f}")
    logging.info(f"Test Accuracy: {accuracy:.4f}")

    logging.info(f"Saving TensorFlow model to {model_output_gcs_uri}")
    model.export(model_output_gcs_uri)
    logging.info("TensorFlow model saved successfully.")


def main():
    parser = argparse.ArgumentParser(description="Train a TensorFlow model.")

    # Core parameters for the training function
    parser.add_argument(
        "--train-dataset",
        type=str,
        required=True,
        help="BigQuery URI of the training data (e.g., bq://project.dataset.table).",
    )
    parser.add_argument(
        "--test-dataset",
        type=str,
        required=True,
        help="BigQuery URI of the test data (e.g., bq://project.dataset.table).",
    )
    parser.add_argument(
        "--base-output-dir",
        type=str,
        required=True,
        help=(
            "The GCS location that the model should be saved to."
            "The string name of the output artifact must be appended to the end. "
            "e.g. {base_output_dir}/trained_model"
        ),
    )
    parser.add_argument("--project", type=str, required=True, help="The project ID.")
    parser.add_argument(
        "--target-column",
        type=str,
        default="ClassIndex",
        help="The name of the column to predict.",
    )

    # Hyperparameters for TensorFlow model training
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        choices=["Adam", "SGD"],
        help="Name of the optimizer to use (e.g., Adam, SGD).",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="sparse_categorical_crossentropy",
        help="Loss function to use (e.g., sparse_categorical_crossentropy, binary_crossentropy, categorical_crossentropy).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="accuracy",
        help="Comma-separated list of metrics to use (e.g., accuracy, AUC).",
    )

    # General script parameters (from original example, not directly used by `train` but can be passed)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (may be overridden by log-level).",
    )

    args, unknown_args = parser.parse_known_args()

    # Set log level based on argument
    os.environ["LOG_LEVEL"] = args.log_level
    logging.getLogger().setLevel(args.log_level.upper())
    logging.info(f"Log level set to {args.log_level}")

    logging.info("--- Known Script Arguments Received ---")
    for arg_name, arg_value in vars(args).items():
        logging.info(f"{arg_name.replace('_', '-')}: {arg_value}")
    logging.info("---------------------------------")
    logging.info(
        f"Note: The following arguments were not explicitly handled by this script: {unknown_args}"
    )

    # Prepare hyperparameters dictionary for the train function
    hyperparameters_dict = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "loss": args.loss,
        "metrics": [m.strip() for m in args.metrics.split(",")],
    }

    try:
        train(
            project=args.project,
            target_column=args.target_column,
            train_uri=args.train_dataset,
            test_uri=args.test_dataset,
            hyperparameters=hyperparameters_dict,
            model_output_gcs_uri=f"{args.base_output_dir}/trained_model",
            verbose=args.verbose,
        )

    except Exception as e:
        logging.error(f"An error occurred during training: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
