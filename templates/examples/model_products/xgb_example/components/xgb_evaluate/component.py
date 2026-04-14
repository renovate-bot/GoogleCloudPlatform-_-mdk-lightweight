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

import sys

sys.path.append("examples/src")
import xgb_example
import mdk.config
import mdk.util.framework
import mdk.util.storage
import mdk.custom_job
import google.cloud.aiplatform
from kfp import dsl
import json
import logging
import os
import datetime
import inspect

logger = logging.getLogger(__name__)

METRICS_FILENAME = "eval_scalar_metrics.json"
CM_FILENAME = "eval_confusion_matrix.json"


# Kubeflow documentation on inputs and outputs:
#
# https://www.kubeflow.org/docs/components/pipelines/user-guides/data-handling/artifacts/


@dsl.component(target_image=mdk.util.framework.getTargetImage(__file__, "xgb_evaluate"))
def xgb_evaluate(
    gcp_config_filename: str,
    pipeline_config_filename: str,
    general_config_filename: str,
    trained_model: dsl.Input[dsl.Model],
    test_dataset: dsl.Input[dsl.Dataset],
    kfp_classification_report: dsl.Output[dsl.Markdown],
    metrics_plots: dsl.Output[dsl.ClassificationMetrics],
    scalar_metrics: dsl.Output[dsl.Metrics],
):
    ############################################################################
    # INITIALIZATION:
    # --------------------------------------------------------------------------
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # Read our config files.
    gcp_config = mdk.config.GCPConfig.from_yaml_file(gcp_config_filename).model_dump()
    pipeline_config = mdk.config.readYAMLConfig(pipeline_config_filename)
    general_config = mdk.config.readAndMergeYAMLConfig(
        config_filename=general_config_filename,
        environment=gcp_config["deployment_environment"],
    )
    training_config = general_config["training"]

    # Set our project ID.
    google.cloud.aiplatform.init(project=gcp_config.get("project_id"))  # fmt: skip
    os.environ["GOOGLE_CLOUD_PROJECT"] = gcp_config.get("project_id")

    # Get our model filename.
    model_filename = training_config["model_filename"]

    # Optional: handle custom job execution if specified.
    staging_bucket = f"{gcp_config.get('pipeline_staging_dir')}/custom_job/staging"
    base_output_dir = f"{gcp_config.get('pipeline_staging_dir')}/custom_job/{timestamp}"
    custom_job_executed = mdk.custom_job.handle_custom_job_if_configured(
        gcp_config=gcp_config,
        model_config=training_config,
        pipeline_config=pipeline_config,
        component_name=inspect.currentframe().f_code.co_name,
        staging_bucket=staging_bucket,
        base_output_dir=base_output_dir,
        input_artifact_map={
            "trained_model": trained_model.uri,
            "test_dataset": test_dataset.uri,
        },
    )
    if custom_job_executed:
        kfp_classification_report.uri = f"{base_output_dir}/kfp_classification_report"
        metrics_plots.uri = f"{base_output_dir}/metrics_plots"
        scalar_metrics.uri = f"{base_output_dir}/scalar_metrics"
    else:
        # The rest of this component is run by default if "container_specs" are not specified
        ############################################################################
        # MANAGE INPUTS (see input arguments in function signature, above)
        # --------------------------------------------------------------------------

        # Download our trained model from GCS:
        mdk.util.storage.download(trained_model.uri, model_filename)

        # Grab our dataset URI.
        test_uri = test_dataset.uri

        ############################################################################
        # CALCULATION
        # --------------------------------------------------------------------------

        # Evaluate our model.
        metrics = xgb_example.evaluate.evaluate(
            general_config_filename,
            model_filename,
            test_uri,
            gcp_config["deployment_environment"],
        )

        ############################################################################
        # MANAGE OUTPUTS (see output arguments in function signature, above)
        # --------------------------------------------------------------------------

        # Persist our metrics.

        # Scalar metrics:
        # +++++++++++++++

        # Report metrics values to the Kubeflow system:
        for metric_name, metric_value in metrics.scalars.items():
            scalar_metrics.log_metric(metric_name, metric_value)

        # Upload the scalar metrics to the URI we were given to use as an output.
        with open(METRICS_FILENAME, "w") as fout:
            fout.write(json.dumps(scalar_metrics.metadata))
        mdk.util.storage.upload(METRICS_FILENAME, scalar_metrics.uri)

        # Confusion matrix:
        # +++++++++++++++++

        # Report the confusion matrix to the Kubeflow system:
        metrics_plots.log_confusion_matrix(
            metrics.class_names, metrics.conf_mat_non_norm.tolist()
        )

        # Upload the confusion matrix to the URI we were given to use as an output.
        with open(CM_FILENAME, "w") as fout:
            fout.write(json.dumps(metrics_plots.metadata))
        mdk.util.storage.upload(CM_FILENAME, metrics_plots.uri)

        # Classification report markdown table:
        # +++++++++++++++++++++++++++++++++++++

        # (Apparently VAI Pipelines handles copying this markdown table to the
        #   output URI.)
        with open(kfp_classification_report.path, "w") as fout:
            fout.write(metrics.class_report_md_table)
