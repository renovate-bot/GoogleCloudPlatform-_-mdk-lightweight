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

This module is for implementing batch prediction logic.
"""

import mdk.data
import mdk.config
import mdk.util.storage
from mdk.model import load_model
from mdk.model.registry import get_model_uri_for_inference
import google.cloud.bigquery
import pandas_gbq
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)


def batch_predict(
    general_config_filename: str,
    gcp_config_filename: str,
    environment: str,
    access_token: str = None,
) -> str:
    """Runs batch predictions on new data using a trained model and saves them.

    Args:
        general_config_filename (str): Filename of config file with model-
            related configuration info.
        gcp_config_filename (str): Filename of config file with GCP-related
            configuration info such as the GCP project ID.
        environment: The environment to use (e.g., 'prod', 'stage', 'train').
        access_token: Optional. An auth token. If not provided, one will be generated.
    """
    # Load our config file.
    general_config = mdk.config.readAndMergeYAMLConfig(
        config_filename=general_config_filename, environment=environment
    )
    training_config = general_config["training"]
    model_filename = training_config["model_filename"]

    inference_config = general_config["inference"]
    data_uri = inference_config["datasets_path"]
    bq_output_table = inference_config["bq_output_table"]

    gcp_config = mdk.config.GCPConfig.from_yaml_file(gcp_config_filename).model_dump()
    project_id = gcp_config.get("project_id")
    region = gcp_config.get("region")

    # Get the model URI.
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
            gcp_project_id=project_id,
            gcp_region=region,
            expanded_model_registry_endpoint=gcp_config.get(
                "expanded_model_registry_endpoint"
            ),
            access_token=access_token,
        )
        logger.info(f"Successfully retrieved model URI: {model_uri}")
    except Exception as e:
        raise ValueError(f"Error retrieving model URI: {e}")

    # Load our dataset to predict on.
    client = google.cloud.bigquery.Client(project=project_id)
    df = mdk.data.getDataframeFromBigQuery(client, data_uri)  # noqa: F841

    # Download the model artifact (e.g., model.pkl) from GCS.
    mdk.util.storage.download(os.path.join(model_uri, model_filename), model_filename)
    # Load our trained model.
    model = load_model(model_filename)  # noqa: F841

    ############################################################################
    ############################################################################

    # Implement your prediction logic here:

    #    Generate predictions from the model.
    #    The method (e.g., .predict(), .predict_proba()) depends on your model type.
    #
    #    Example:
    #    predictions = model.predict(X_pred)
    #    prediction_probabilities = model.predict_proba(X_pred)[:, 1] # For classifier

    df_predictions = pd.DataFrame()  # Placeholder for the final predictions DataFrame

    # (For an example of how to implement this for an XGBoost model, please
    #  see the corresponding batch_predict.py in the example directory)

    ############################################################################
    ############################################################################

    # Save our predictions to the specified BigQuery URI.

    try:
        pandas_gbq.to_gbq(
            dataframe=df_predictions,
            destination_table=bq_output_table.replace("bq://", ""),
            project_id=project_id,
            if_exists="append",  # Append new predictions without overwriting existing data.
        )
        logger.info(
            f"Successfully wrote {len(df_predictions)} predictions to {bq_output_table}"
        )

    except Exception as e:
        e.add_note(f"Error writing predictions to BigQuery: {e}")
        raise

    return bq_output_table
