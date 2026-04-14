# How to Use the Custom Job Library

This document outlines how to use the `custom_job` package, which facilitates the creation and execution of Vertex AI Custom Jobs from Python. This module is particularly useful for orchestrating custom containerized tasks within larger systems like Kubeflow Pipelines (KFP).

---

## 1. Orchestrating Custom Jobs within a Kubeflow Pipeline Component

The `handle_custom_job_if_configured` function is designed to be called from within a Kubeflow Pipeline component. It checks if `container_specs` are defined for the current component in your pipeline configuration YAML. If these specifications are present, it orchestrates the creation and execution of a Vertex AI Custom Job, passing along the necessary configurations and arguments.

This approach allows you to seamlessly switch between running workloads within the component and offloading its execution to a managed Vertex AI Custom Job, simply by adjusting your pipeline configuration.

### Example: A Kubeflow Pipelines Component (`xgb_train.py`)

This example demonstrates a KFP component that can either execute its logic directly (e.g., in a local Docker container or a standard KFP executor) or delegate to a Vertex AI Custom Job, depending on the `pipeline_config`.

```python
# mdk/your_pipeline/components/xgb_train.py
import sys
import json
import logging
import os
import datetime
import inspect

import mdk.config
import mdk.custom_job
import mdk.util.framework
import mdk.util.storage
import xgb_example

import google.cloud.aiplatform
from kfp import dsl

logger = logging.getLogger(__name__)

HYPERPARAMETERS_JSON = "hyperparameters.json"

@dsl.component(target_image=mdk.util.framework.getTargetImage(__file__, "xgb_train"))
def xgb_train(
    gcp_config_filename: str,
    pipeline_config_filename: str,
    general_config_filename: str,
    train_dataset: dsl.Input[dsl.Dataset],
    test_dataset: dsl.Input[dsl.Dataset],
    hyperparameters: dsl.Input[dsl.Artifact],
    trained_model: dsl.Output[dsl.Model],
):
    """
    KFP component for XGBoost training. Can run locally or as a Vertex AI Custom Job.
    """
    logger.info("Starting xgb_train component execution.")
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # 1. Load Configurations
    # These configurations are typically loaded from files passed as KFP component inputs.
    gcp_config = mdk.config.GCPConfig.from_yaml_file(gcp_config_filename).model_dump()
    pipeline_config = mdk.config.readYAMLConfig(pipeline_config_filename)
    general_config = mdk.config.readAndMergeYAMLConfig(
        config_filename=general_config_filename,
        environment=gcp_config["deployment_environment"]
    )
    training_config = general_config["training"] # This will be passed as model_config

    # Set our project ID for Vertex AI SDK.
    google.cloud.aiplatform.init(project=gcp_config.get("project_id"))
    os.environ["GOOGLE_CLOUD_PROJECT"] = gcp_config.get("project_id")
    logger.info(f"Vertex AI initialized for project: {gcp_config.get('project_id')}")

    # 2. Check for Custom Job Execution
    # This is the core logic that decides whether to run as a Custom Job.
    staging_bucket = f"{gcp_config.get('pipeline_staging_dir')}/custom_job/staging"
    base_output_dir = f"{gcp_config.get('pipeline_staging_dir')}/custom_job/{timestamp}"

    # The component name is dynamically obtained for handle_custom_job_if_configured
    current_component_name = inspect.currentframe().f_code.co_name

    custom_job_executed = mdk.custom_job.handle_custom_job_if_configured(
        gcp_config=gcp_config,
        model_config=training_config, # The parameters for the model/job
        pipeline_config=pipeline_config,
        component_name=current_component_name,
        staging_bucket=staging_bucket,
        base_output_dir=base_output_dir,
        input_artifact_map={
            "train_dataset": train_dataset.uri,
            "test_dataset": test_dataset.uri,
            "hyperparameters": hyperparameters.uri,
        },
    )

    # 3. Handle Outputs Based on Execution Type
    if custom_job_executed:
        # If the custom job ran, its output is expected in base_output_dir.
        # We point the KFP output artifact to that location.
        trained_model.uri = f"{base_output_dir}/trained_model"
        logger.info(f"Custom Job handled execution. Model output URI set to: {trained_model.uri}")
    else:
        # If no custom job was configured, proceed with local execution within the component.
        logger.info("No Custom Job configured. Executing training locally within the component.")

        # MANAGE INPUTS
        train_uri = train_dataset.uri
        test_uri = test_dataset.uri

        mdk.util.storage.download(hyperparameters.uri, HYPERPARAMETERS_JSON)
        with open(HYPERPARAMETERS_JSON, "r") as fin:
            hyperparameters_dict = json.loads(fin.read())
        logger.info(f"Hyperparameters loaded: {hyperparameters_dict}")

        # CALCULATION (Local Training Logic)
        model_filename = xgb_example.train.train(
            general_config_filename, train_uri, test_uri, hyperparameters_dict, gcp_config["deployment_environment"]
        )
        logger.info(f"Local training completed. Model saved as: {model_filename}")

        # MANAGE OUTPUTS (Local Upload)
        # We append the model's actual filename to the KFP-provided output URI.
        trained_model.uri = f"{trained_model.uri}/{model_filename}"
        mdk.util.storage.upload(model_filename, trained_model.uri, mkdir=True)
        logger.info(f"Local model uploaded to: {trained_model.uri}")

    logger.info("xgb_train component execution completed.")
```

#### How it works:

1.  **Configuration Loading**: The component first loads its various configuration files (GCP, pipeline, general) which determine its behavior.
2.  **`handle_custom_job_if_configured` Call**: The component passes its relevant configuration, its own name (`inspect.currentframe().f_code.co_name`), and the URIs of its input artifacts.
3.  **Conditional Execution**:
    *   **If a Custom Job is configured**: `handle_custom_job_if_configured` detects the `container_specs` for `xgb_train` in `pipeline_config`. It then constructs, launches, and waits for a Vertex AI Custom Job to complete. The component's local logic (the `else` block) is skipped. The `trained_model.uri` is then set to where the Custom Job is expected to have deposited its output.
    *   **If no Custom Job is configured**: `handle_custom_job_if_configured` returns `False`, and the component proceeds to execute its training logic directly. In this scenario, it downloads the necessary artifacts, runs `xgb_example.train.train`, and uploads the resulting model to the `trained_model.uri`.




## 2. Input/Output Contract for Custom Job Containers

When `handle_custom_job_if_configured` launches a Vertex AI Custom Job, it constructs command-line arguments for the Python script or command running inside your container. To ensure seamless data flow with Kubeflow Pipelines, your container's entrypoint script must adhere to a specific I/O contract.

#### 1. Input Artifacts and Configuration Parameters

All parameters passed to `handle_custom_job_if_configured` via `model_config` and `input_artifact_map` (e.g., `{"train_dataset": train_dataset.uri}`) are automatically converted into command-line flags for your container script.

*   **Conversion Rule**: Dictionary keys are transformed from `snake_case` to `kebab-case` and prefixed with `--`.
    *   `model_config` keys (e.g., `n_estimators`, `data_path`) become `--n-estimators`, `--data-path`.
    *   `input_artifact_map` keys (e.g., `train_dataset`, `hyperparameters`) become `--train-dataset`, `--hyperparameters`.
*   **Values**: The values are passed as strings. For `dsl.Input[dsl.Dataset]` and `dsl.Input[dsl.Artifact]`, this will be the GCS URI of the artifact.

**Additional Arguments Always Passed:**

*   `--base-output-dir`: The GCS path where output artifacts for this KFP component run should be stored.
*   `--project`: The GCP project ID where the job is running.

#### 2. Output Artifacts

To link the output of your custom job back to a KFP output artifact (e.g., `trained_model: dsl.Output[dsl.Model]`), your container script must write its output to a specific GCS path.

*   **Output Path Convention**: The KFP component will set the `trained_model.uri` to a path like `f"{base_output_dir}/trained_model"`. Your container script *must* write its model to this exact GCS location.
*   **Inside the Container**: Your container script receives `--base-output-dir` as an argument. It should construct the output path by combining this argument with the *name* of the KFP output artifact.

#### Example Container Script (`src/xgb_train_script.py`):

This script would run inside the custom job container and demonstrates how to parse inputs and write outputs according to the contract.

```python
# src/xgb_train_script.py (This script runs inside your custom job container)
import argparse
import logging
import os
import json

# Assuming mdk.util.storage and xgb_example.train are available in your container image
import mdk.util.storage
import xgb_example

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="XGBoost training script for Vertex AI Custom Job.")

    # Input artifacts (from input_artifact_map in KFP component)
    parser.add_argument('--train-dataset', type=str, required=True,
                        help="GCS URI of the training dataset.")
    parser.add_argument('--test-dataset', type=str, required=True,
                        help="GCS URI of the test dataset.")
    parser.add_argument('--hyperparameters', type=str, required=True,
                        help="GCS URI of the hyperparameters artifact.")

    # Model configuration parameters (from model_config in KFP component)
    parser.add_argument('--n-estimators', type=int, default=100,
                        help="Number of boosting rounds for XGBoost.")
    parser.add_argument('--learning-rate', type=float, default=0.1,
                        help="Learning rate for XGBoost.")
    parser.add_argument('--data-path', type=str,
                        help="Optional data path from model config (example).")

    # Required arguments for custom job output and context
    parser.add_argument('--base-output-dir', type=str, required=True,
                        help="Base GCS directory for all outputs from this custom job.")
    parser.add_argument('--project', type=str, required=True,
                        help="GCP Project ID.")

    args = parser.parse_args()

    logger.info(f"Running Custom Job in project: {args.project}")
    logger.info(f"Input train_dataset URI: {args.train_dataset}")
    logger.info(f"Input test_dataset URI: {args.test_dataset}")
    logger.info(f"Input hyperparameters URI: {args.hyperparameters}")
    logger.info(f"XGBoost n_estimators: {args.n_estimators}")
    logger.info(f"XGBoost learning_rate: {args.learning-rate}")
    logger.info(f"Base output directory: {args.base_output-dir}")

    # --- Load Data and Hyperparameters ---
    # Download hyperparameters from GCS (using mdk.util.storage or similar)
    local_hyperparameters_file = "hyperparameters.json"
    mdk.util.storage.download(args.hyperparameters, local_hyperparameters_file)
    with open(local_hyperparameters_file, "r") as f:
        hyperparameters_dict = json.load(f)
    # The `hyperparameters_dict` can be updated with CLI args if needed
    hyperparameters_dict['n_estimators'] = args.n_estimators
    hyperparameters_dict['learning_rate'] = args.learning_rate

    # --- Execute Training ---
    # Assuming xgb_example.train.train can take these arguments
    # (Note: In your KFP component, `general_config_filename` was passed.
    # Here, you might need a simplified version or pass individual config values)
    model_filename = xgb_example.train.train(
        train_uri=args.train_dataset,
        test_uri=args.test_dataset,
        hyperparameters_dict=hyperparameters_dict,
        # ... other necessary args derived from general_config
    )

    # --- Save Output Model ---
    # The KFP component defines an output artifact named 'trained_model'.
    # We must save the output to f"{args.base_output_dir}/trained_model".
    target_output_gcs_uri = os.path.join(args.base_output_dir, "trained_model")

    # Assuming your model_filename (e.g., "my_model.pkl") is a local file after training
    # upload it to the designated GCS path.
    mdk.util.storage.upload(model_filename, target_output_gcs_uri, mkdir=True)
    logger.info(f"Trained model saved to GCS: {target_output_gcs_uri}")

if __name__ == "__main__":
    main()
```



#### Example Pipeline Configuration (`pipeline_config.yml`):

The below YAML snippets shows how you would configure your pipeline to trigger a Vertex AI Custom Job for the `xgb_train` component.

```yaml
  # Bring-Your-Own-Container (BYOC) example scenarios:
  # BYOC Scenario 1: Running command/args with full_uri container (e.g. "us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-17.py310:latest”)
  train:
    description: "Model training workflow"
    function: train
    module_path: model_products/main_product/components/train/component.py
    image_artifact: "us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-17.py310:latest"
    container_specs:
      command:
      - bash
      - -c
      - echo 'hello world'
  # BYOC Scenario 2: Running script within full_uri container
  train:
    description: "Model training workflow"
    function: train
    module_path: model_products/main_product/components/train/component.py
    image_artifact: "us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-17.py310:latest"
    container_specs:
      script_path: src/model_workflow/demo_byoc.py
  # BYOC Scenario 3: Running command/args with Docker container (e.g. "byoc:latest”)
  train:
    description: "Model training workflow"
    function: train
    module_path: model_products/main_product/components/train/component.py
    image_artifact: "byoc:latest"
    container_specs:
      command:
      - python
      - trainer.py
```
