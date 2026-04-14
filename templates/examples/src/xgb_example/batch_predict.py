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

"""
This script defines a custom Kubeflow component for running batch predictions.

It loads a pre-trained XGBoost model and a dataset, performs inference,
and writes the prediction results back to a BigQuery table.
"""

from google.cloud import bigquery
import numpy as np
import pandas as pd
import pandas_gbq
import xgboost as xgb
from mdk.model.registry import get_model_uri_for_inference
import mdk.config
import mdk.util.storage
from mdk.model import load as load_model
import logging
import os

logger = logging.getLogger(__name__)


def batch_prediction(
    general_config_filename: str,
    gcp_config_filename: str,
    environment: str,
    access_token: str = None,
) -> str:
    """
    Executes a batch prediction job using a trained XGBoost model.

    This function performs the following steps:
    1.  Loads configuration from a YAML file.
    2.  Loads the input dataset from either BigQuery or Google Cloud Storage (GCS).
    3.  Retrieves the trained model from either the Vertex AI Model Registry
        or a specified GCS path.
    4.  Runs batch predictions on the input data.
    5.  Appends the predictions and a timestamp to the input data.
    6.  Uploads the results to a specified BigQuery table.

    Note: This assumes we have already called aiplatform.init() to set the
    GCP project ID.

    Args:
        general_config_filename (str): Filename of config file with model-
            related configuration info.
        gcp_config_filename (str): Filename of config file with GCP-related
            configuration info such as the GCP project ID.
        environment: The environment to use (e.g., 'prod', 'stage', 'train').
        access_token: Optional. An auth token. If not provided, one will be generated.

    Returns:
        The BigQuery table URI where the prediction results are stored.
    """
    logger.info("\n--- Running Batch Prediction ---")

    # Load config files.
    gcp_config = mdk.config.GCPConfig.from_yaml_file(gcp_config_filename).model_dump()
    general_config = mdk.config.readAndMergeYAMLConfig(
        config_filename=general_config_filename, environment=environment
    )
    training_config = general_config["training"]
    model_filename = training_config["model_filename"]

    # --- 1. Load Input Data ---
    inference_config = general_config["inference"]
    datasets_path = inference_config.get("datasets_path")
    class_names = inference_config.get("class_names")

    bq_client = bigquery.Client(project=gcp_config.get("project_id"))

    # Remove the 'bq://' prefix required for the BigQuery client and load data.
    clean_datasets_path = datasets_path.replace("bq://", "")
    query = f"SELECT * FROM `{clean_datasets_path}`"
    dataframe = bq_client.query_and_wait(query).to_dataframe()
    assert not dataframe.empty, (
        "dataframe should be loaded and contain data in all code paths"
    )

    # --- 2. Retrieve Trained Model ---
    model_ref_fields = set(mdk.config.ModelReferenceConfig.model_fields.keys())
    model_reference_config_data = {
        key: general_config["general"].get(key) for key in model_ref_fields
    }
    # Add in deployment_environment, this field is required by ModelReferenceConfig
    model_reference_config_data["deployment_environment"] = gcp_config.get(
        "deployment_environment"
    )

    try:
        model_uri = get_model_uri_for_inference(
            model_reference_config_data=model_reference_config_data,
            gcp_project_id=gcp_config.get("project_id"),
            gcp_region=gcp_config.get("region"),
            expanded_model_registry_endpoint=gcp_config.get(
                "expanded_model_registry_endpoint"
            ),
            access_token=access_token,
        )
        logger.info(f"Successfully retrieved model URI: {model_uri}")
    except Exception as e:
        raise ValueError(f"Error retrieving model URI: {e}")

    # Download the model artifact (e.g., model.pkl) from GCS and load it.
    mdk.util.storage.download(os.path.join(model_uri, model_filename), model_filename)
    model = load_model(model_filename)

    # --- 3. Run Predictions ---
    logger.info("\n--- Generating Predictions ---")

    # Prepare the dataframe for prediction by creating an XGBoost DMatrix.
    dpredict = xgb.DMatrix(dataframe)

    # Ensure class names are sorted to correctly map prediction indices to labels.
    class_names = np.sort(np.array(class_names))

    # Get prediction probabilities from the model.
    y_pred_proba = model.predict(dpredict)

    # Determine the predicted class by finding the index with the highest probability.
    predicted_class_indices = np.argmax(y_pred_proba, axis=1)

    # Map the predicted indices back to the actual class names.
    y_pred = class_names[predicted_class_indices]

    logger.info(f"Predicted probabilities for first 5 samples:\n{y_pred_proba[:5]}")
    logger.info(f"Predicted classes for first 5 samples:\n{y_pred[:5]}")

    # --- 4. Store Results ---
    # Add the prediction results and a current timestamp to the dataframe.
    dataframe["Predicted"] = y_pred
    dataframe["Timestamp"] = pd.Timestamp.now()

    # Write the dataframe with predictions back to the specified BigQuery output table.
    bq_output_table = inference_config["bq_output_table"]
    pandas_gbq.to_gbq(
        dataframe=dataframe,
        destination_table=bq_output_table.replace("bq://", ""),  # Clean 'bq://' prefix.
        project_id=gcp_config.get("project_id"),
        if_exists="append",  # Append new predictions without overwriting existing data.
    )

    logger.info(f"Successfully wrote predictions to {bq_output_table}")

    # Return the path of the BigQuery output table as an output artifact.
    return bq_output_table


def _get_location_format(input_uri: str) -> str:
    """Extracts location substring and validates location options.
    Args:
        input_uri: The full resource URI.
    Returns:
        String value either 'bq' or 'gs'.
    Raises:
        ValueError if location format not supported.
    """
    location_format = input_uri.split(":")[0]
    if location_format not in ("bq", "gs"):
        raise ValueError(
            f"Location format: {location_format} "
            f"is not supported. Must be either gs or bq."
        )
    return location_format


def _get_file_format(input_uri: str) -> str:
    """Extracts file extension substring.
    Args:
        input_uri: The full resource URI.
    Returns:
        String value file extension, without the period.
    """
    _, file_ext = os.path.splitext(input_uri)
    return file_ext[1:]
