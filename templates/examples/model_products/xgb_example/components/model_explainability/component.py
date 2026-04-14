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
import logging
import os
import mdk.util.framework
import mdk.util.storage
import mdk.config
import mdk.custom_job
from mdk.model import load as load_model
import google.cloud.aiplatform
from kfp import dsl
import datetime
import inspect

logger = logging.getLogger(__name__)


@dsl.component(
    target_image=mdk.util.framework.getTargetImage(__file__, "model_explainability")
)
def model_explainability(
    gcp_config_filename: str,
    pipeline_config_filename: str,
    general_config_filename: str,
    trained_model: dsl.Input[dsl.Model],
    test_dataset: dsl.Input[dsl.Dataset],
    shap_summary_plot: dsl.Output[dsl.Markdown],
    shap_beeswarm_plot: dsl.Output[dsl.HTML],
):
    """
    gcp_config_filename : The GCP config file which contains the project id need to initialize AI platform
    general_config_filename : The general config file that has parameters to specify if shap explainability is to be run or not and what is the sample size on which to run
    trained_model : The trained model artifact that is created during the train step which is used to generate shap values
    test_dataset : The test dataset on which to generate shap values - this is a BQ table.
    shap_summary_plot: This is an output artifact that stores the generated shap summary plot
    shap_beeswarm_plot : This is an output artifact that stores the generated beeswarm plot

    """
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
        shap_summary_plot.uri = f"{base_output_dir}/shap_summary_plot"
        shap_beeswarm_plot.uri = f"{base_output_dir}/shap_beeswarm_plot"
    else:
        # The rest of this component is run by default if "container_specs" are not specified
        ############################################################################
        # MANAGE INPUTS (see input arguments in function signature, above)
        # --------------------------------------------------------------------------
        # Get our model filename.
        model_filename = training_config[
            "model_filename"
        ]  # To download and load model for generating SHAP values
        target_class = training_config["target_column"]

        # Read parameters to run SHAP calculation
        run_shap_analysis = training_config[
            "run_shap_analysis"
        ]  # Boolean - to run SHAP or not
        shap_sample_size = training_config[
            "shap_sample_size"
        ]  # Number of examples to run SHAP calculation for
        logging.info(f"run_shap_analysis: {run_shap_analysis}")

        if run_shap_analysis:  # If SHAP calculation is to be run
            # Download our trained model from GCS:
            mdk.util.storage.download(trained_model.uri, model_filename)
            model = load_model(model_filename)

            # Get the uri for the test dataset (BQ):
            test_uri = test_dataset.uri

            ############################################################################
            # Generate SHAP Plot
            # --------------------------------------------------------------------------
            ## Get SHAP values, SHAP test set and list of classes in the dataset.

            shap_values, shap_dtest, classes_list = (
                xgb_example.model_explainability_tree.model_explainability_tree(
                    model, test_uri, target_class, shap_sample_size
                )
            )
            logger.info("SHAP values calculated.")

            # --- Generate and save SHAP Summary Plot  ---
            summary_plot_file = (
                xgb_example.model_explainability_tree._create_summary_plot(
                    shap_values, shap_dtest
                )
            )

            # --- Generate and save SHAP Beeswarm Plot  ---
            """
            The number of beeswarm plots generated are equal to the number of classes. The pipeline console markdown section cannot hold all the plots due to size constraints.
            Hence all individual plots are first combined into a single plot and then rendered as HTML in a separate tab.
            """
            # Generate individual beeswarm plots and add to a list
            beeswarm_plot_list = (
                xgb_example.model_explainability_tree._create_plot_list_of_classes(
                    classes_list, shap_values
                )
            )
            # Combine individual plot images into a combined single plot image
            beeswarm_plot_file = (
                xgb_example.model_explainability_tree.combine_pngs_with_matplotlib(
                    beeswarm_plot_list,
                    output_filename="combined_shap_plots.png",
                    ncols=3,
                )
            )
            # Convert the final plot to a HTML instead of markdown to show in separate page
            beeswarm_plot_html = (
                xgb_example.model_explainability_tree.convert_png_to_embedded_html(
                    beeswarm_plot_file, title="Beeswarm plots"
                )
            )

            ############################################################################
            # MANAGE OUTPUTS (see output arguments in function signature, above)
            # --------------------------------------------------------------------------

            # Write SHAP summary plot to pipeline console as a markdown element
            with open(shap_summary_plot.path, "w") as f:
                f.write(
                    xgb_example.model_explainability_tree._plot_to_markdown(
                        summary_plot_file, "SHAP Summary Plot"
                    )
                )

            # Write SHAP beeswarm plot to pipeline console as a HTML element.
            with open(shap_beeswarm_plot.path, "w") as f:
                f.write(beeswarm_plot_html)
