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

"""This example evaluates an XGBoost model as an example model implementation."""

import mdk.data
import numpy as np
import pandas as pd
import sklearn.metrics
import xgboost as xgb
import google.cloud.bigquery
import collections
import logging
import mdk.config
from mdk.model import load as load_model

logger = logging.getLogger(__name__)

EvaluateReturnType = collections.namedtuple(
    "ModelEvaluationReturnType",
    ["class_names", "scalars", "class_report_md_table", "conf_mat_non_norm"],
)


def evaluate(
    general_config_filename: str,
    model_filename: str,
    test_uri: str,
    environment: str,
) -> EvaluateReturnType:
    """Example code to evaluate an XGBoost model.

    Args:
        general_config_filename (str): Filename of config file with model-
            related configuration info.
        model_filename (str): Filename of trained XGBoost model that is to be
            evaluated.
        test_uri (str): BigQuery URI of the test data against which metrics are
            to be calculated.
        environment: The environment to use (e.g., 'prod', 'stage', 'train').

    Returns:
        ModelEvaluationReturnType: A named tuple giving the class labels, scalar
            metrics, Markdown table with a classification report, and the
            confusion matrix.
    """
    logger.info("Running model evaluation...")

    # Read our config file.
    general_config = mdk.config.readAndMergeYAMLConfig(
        config_filename=general_config_filename, environment=environment
    )
    training_config = general_config["training"]

    # Load our model.
    model = load_model(model_filename)

    # Load our test data.
    client = google.cloud.bigquery.Client()
    df_test = mdk.data.getDataframeFromBigQuery(client, test_uri)
    class_names = ["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"]

    # Explicitly set labels to int
    target_column = training_config["target_column"]
    # df_test[target_column] = df_test[target_column].astype(int)

    # Get the actual classes.
    y_act = df_test[target_column].astype(int)

    # Get predicted probabilities from the test data.
    X_test = df_test.drop(columns=[target_column])
    dmat = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dmat)

    # Map the probabilities to a specific, most likely class (represented as an integer).
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Calculate our scalar metrics.
    scalars = {}
    scalars["Accuracy Score"] = sklearn.metrics.accuracy_score(y_act, y_pred)
    # One vs Rest
    scalars["ROC AUC Macro (OVR)"] = sklearn.metrics.roc_auc_score(
        y_act, y_pred_proba, multi_class="ovr", average="macro"
    )
    scalars["ROC AUC Weighted (OVR)"] = sklearn.metrics.roc_auc_score(
        y_act, y_pred_proba, multi_class="ovr", average="weighted"
    )
    # One vs One
    scalars["ROC AUC Macro (OVO)"] = sklearn.metrics.roc_auc_score(
        y_act, y_pred_proba, multi_class="ovo", average="macro"
    )
    scalars["ROC AUC Weighted (OVO)"] = sklearn.metrics.roc_auc_score(
        y_act, y_pred_proba, multi_class="ovo", average="weighted"
    )

    # Build the Classification Report.
    class_report = sklearn.metrics.classification_report(
        y_act, y_pred, output_dict=True
    )
    class_report_df = pd.DataFrame.from_dict(class_report).T
    class_report_df.index.name = "Label"
    class_report_df.index = class_names + ["accuracy", "macro avg", "weighted avg"]
    class_report_md_table = class_report_df.to_markdown(index=True)

    # Build the confusion matrix.
    conf_mat_non_norm = sklearn.metrics.confusion_matrix(y_act, y_pred)

    # Return our metrics.
    return EvaluateReturnType(
        class_names, scalars, class_report_md_table, conf_mat_non_norm
    )
