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

"""This module provides sample functionality that simulates training a model, used for demoing
Bring-your-own-container (BYOC) functionality."""

import argparse
import logging
import os
import sys
import time  # Import time for simulating delays

logger = logging.getLogger(__name__)


def getDataframeFromBigQuery(
    client,
    uri,
):
    """Given a URI, pretends to retrieve data and returns a dummy result.

    Args:
        client (any): A dummy client object (not used for actual operations).
        uri (str): The URI to "load" data from.

    Returns:
        dict: A dummy representation of a DataFrame.
    """
    logger.info(f"Pretending to load data from BigQuery URI: {uri}")
    time.sleep(0.5)  # Simulate network latency/data loading
    # In a real scenario, this would return a pandas DataFrame.
    # Here, we just return a dict to represent some dummy data.
    return {"col1": [1, 2, 3], "col2": [4, 5, 6], "target": [0, 1, 0]}


def train(
    project: str,
    target_column: str,
    train_uri: str,
    test_uri: str,
    hyperparameters: dict | None,
    model_output_gcs_uri: str,
    verbose: bool = False,
):
    """Example model that pretends to train a dataset.

    Args:
        project (str): The project ID.
        target_column (str): The name of the column to predict.
        train_uri (str): BigQuery URI of the training data.
        test_uri (str): BigQuery URI of the test data.
        hyperparameters (dict): A dict that contains a mapping from
            hyperparameter names to hyperparameter values.
        model_output_gcs_uri (str): The GCS location that the model should be saved to.
        verbose: Verbose flag for training (will mainly affect log output).
    """
    logger.info(f"Starting dummy training process for project: {project}")

    # Pretend to load our dataset from BigQuery.
    # The 'client' is now just a placeholder.
    dummy_client = object()  # A simple dummy object
    logger.info(f"Pretending to load training data from: {train_uri}")
    dummy_df_train = getDataframeFromBigQuery(dummy_client, train_uri)
    logger.info(f"Pretending to load test data from: {test_uri}")
    dummy_df_test = getDataframeFromBigQuery(dummy_client, test_uri)

    logger.info(f"Dummy training data first row (simulated):\n{dummy_df_train}")
    logger.info(f"Dummy test data first row (simulated):\n{dummy_df_test}")

    # Default model parameters (will be overridden by hyperparameters)
    tf_params = {
        "epochs": 10,
        "learning_rate": 0.001,
        "batch_size": 32,
        "optimizer": "Adam",
        "loss": "sparse_categorical_crossentropy",
        "metrics": ["accuracy"],
    }

    # If hyperparameters are provided, update our dummy parameters
    if hyperparameters:
        logger.info(f"Incorporating provided hyperparameters: {hyperparameters}")
        tf_params.update(hyperparameters)
    else:
        logger.info("No hyperparameters provided, using defaults.")

    logger.info(f"Simulating training with parameters: {tf_params}")
    logger.info(f"Target column: '{target_column}'")

    # Simulate data preparation
    logger.info("Simulating data preprocessing and feature extraction...")
    time.sleep(1)  # Simulate some work

    # Simulate model definition
    logger.info("Simulating model definition (e.g., a simple neural network)...")
    time.sleep(0.8)

    # Simulate model compilation
    logger.info(
        f"Simulating model compilation with optimizer '{tf_params['optimizer']}' and loss '{tf_params['loss']}'..."
    )
    time.sleep(0.7)

    # Simulate the training loop
    epochs = tf_params.get("epochs", 10)
    for epoch in range(1, epochs + 1):
        if verbose:
            logger.info(f"Epoch {epoch}/{epochs} - Simulating training step...")
        else:
            # Only log epoch progress at intervals if not verbose
            if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
                logger.info(f"Epoch {epoch}/{epochs} - Training in progress...")
        time.sleep(0.3)  # Simulate one epoch of training

    logger.info("Model training simulation complete.")

    # Simulate evaluation
    logger.info("Simulating model evaluation on test data...")
    time.sleep(0.5)
    dummy_loss = 0.5 + (
        1 - (tf_params.get("epochs", 10) / 20)
    )  # Simulate improvement with epochs
    dummy_accuracy = 0.75 + (
        tf_params.get("epochs", 10) / 100
    )  # Simulate improvement with epochs
    logger.info(f"Simulated Test Loss: {dummy_loss:.4f}")
    logger.info(f"Simulated Test Accuracy: {dummy_accuracy:.4f}")

    # Simulate model saving
    logger.info(f"Pretending to save model to GCS location: {model_output_gcs_uri}")
    time.sleep(1)  # Simulate upload time
    logger.info("Dummy model saved successfully.")
    logger.info("Training script finished successfully.")


def main():
    parser = argparse.ArgumentParser(description="Pretend to train a model.")

    # Core parameters (kept similar to original for argument demonstration)
    parser.add_argument(
        "--train-dataset-uri",
        type=str,
        required=True,
        help="BigQuery URI of the training data (e.g., bq://project.dataset.table).",
    )
    parser.add_argument(
        "--test-dataset-uri",
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
            "e.g. /trained_model"
        ),
    )
    parser.add_argument("--project", type=str, required=True, help="The project ID.")
    parser.add_argument(
        "--target-column",
        type=str,
        default="ClassIndex",
        help="The name of the column to predict.",
    )

    # Hyperparameters (kept for demonstration, even if only 'epochs' is directly used in simulation)
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

    # General script parameters
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
        help="Enable verbose output (e.g., per-epoch logging).",
    )

    args, unknown_args = parser.parse_known_args()

    # Set log level based on argument
    os.environ["LOG_LEVEL"] = args.log_level
    logging.getLogger().setLevel(args.log_level.upper())
    logger.info(f"Log level set to {args.log_level}")

    logger.info("--- Known Script Arguments Received ---")
    for arg_name, arg_value in vars(args).items():
        logger.info(f"{arg_name.replace('_', '-')}: {arg_value}")
    logger.info("---------------------------------")
    if unknown_args:
        logger.warning(
            f"Note: The following arguments were not explicitly handled by this script: {unknown_args}"
        )
    else:
        logger.info("No unknown arguments received.")

    # Prepare hyperparameters dictionary for the train function
    # These are passed to 'train' to demonstrate the flow, even if 'train' doesn't use all of them.
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
            train_uri=args.train_dataset_uri,
            test_uri=args.test_dataset_uri,
            hyperparameters=hyperparameters_dict,
            model_output_gcs_uri=f"{args.base_output_dir}/trained_model",
            verbose=args.verbose,
        )

    except Exception as e:
        logger.error(f"An error occurred during dummy training: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
