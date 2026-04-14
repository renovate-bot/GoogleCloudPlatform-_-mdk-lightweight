# Kubeflow Pipelines: Conditional Logic Guide

This document provides guidance on implementing and utilizing conditional logic within your Kubeflow Pipelines (KFP). Conditional logic allows you to dynamically alter the execution flow of your pipelines based on the outcomes of previous tasks or input parameters.

## Understanding Conditional Logic in KFP

Kubeflow Pipelines leverages the Kubernetes-native workflow engine, Argo Workflows, which supports advanced features like conditional execution. In KFP, conditional logic is primarily managed using the `kfp.dsl.Condition` construct (updated to `kfp.dsl.If` in V2+). This allows you to define branches in your pipeline that will only execute if a specified condition is met.

**Key Use Cases for Conditional Logic:**

*   **Branching Execution Paths:** Based on model evaluation metrics, deploy a model only if it meets certain performance thresholds.
*   **Skipping Steps:** Conditionally skip computationally expensive steps if previous steps have already produced satisfactory results or if specific conditions are met (e.g., resuming from a checkpoint).
*   **Error Handling and Rollback:** Implement custom error handling or rollback procedures based on task failures.
*   **Dynamic Pipeline Configuration:** Adjust pipeline behavior based on input parameters, such as different execution environments (development, staging, production).

## Implementing Conditional Logic: A Detailed Example

The following Python code demonstrates how to implement conditional logic in a Kubeflow Pipeline. This example showcases a machine learning pipeline that can either run a full training process or resume from pre-existing trained model outputs.

**File:** `conditional_pipeline.py`

```python
"""This module provides a pipeline for Kubefulow / Vertex AI Pipelines."""

import mdk.util.framework
import kfp.dsl

from typing import Optional

MR_CONFIG_FILENAME = "config/examples/xgb/model_registry_config.yml"
GCP_CONFIG_FILENAME_TEMPLATE = "config/{environment}/gcp_config.yml"
MODEL_CONFIG_FILENAME_TEMPLATE = "config/examples/xgb/{environment}/model_config.yml"


@kfp.dsl.pipeline(name="xgb_training_pipeline")
def pipeline(
    environment: str,
    # --- Parameters for Resuming a Pipeline ---
    deploy_model_to_endpoint: bool = False,
    resume_from_training_outputs: bool = False,
    trained_model_gcs_uri: Optional[str] = None,
    test_dataset_gcs_uri: Optional[str] = None,
):
    """This is pipeline for Kubefulow / Vertex AI Pipelines, comprised of
    a series of Kubeflow components.

    Args:
        environment (str): One of: ("dev", "stage" or "prod"). This will be
            used to infer the filename of the GCP config file the project uses
            to store the project ID, etc.
        deploy_model_to_endpoint (bool): Determines whether to deploy the model to a Vertex
            Endpoint for online prediction.
        resume_from_training_outputs (bool): If True, skips preprocess, hyperopt, and train
            and attempts to use provided URIs for trained_model and test_dataset.
        trained_model_gcs_uri (Optional[str]): GCS URI of the trained model to use
            when resuming. Required if `resume_from_training_outputs` is True.
        test_dataset_gcs_uri (Optional[str]): GCS URI of the test dataset to use
            when resuming. Required if `resume_from_training_outputs` is True.
    """
    print("Creating tasks...", flush=True)

    # Load our component tasks from file:
    xgb_preprocess = mdk.util.framework.loadComponentSpec("xgb_preprocess")
    xgb_hyperopt = mdk.util.framework.loadComponentSpec("xgb_optimize_hyperparameters")
    xgb_train = mdk.util.framework.loadComponentSpec("xgb_train")
    xgb_evaluate = mdk.util.framework.loadComponentSpec("xgb_evaluate")
    upload_to_model_registry = mdk.util.framework.loadComponentSpec("upload_to_model_registry")  # fmt: skip
    model_deployment = mdk.util.framework.loadComponentSpec("model_deployment")

    # Infer our config filenames.
    gcp_config_filename = GCP_CONFIG_FILENAME_TEMPLATE.format(environment=environment)
    model_config_filename = MODEL_CONFIG_FILENAME_TEMPLATE.format(
        environment=environment
    )

    # Execute our component tasks:

    # Kubeflow documentation on inputs and outputs:
    # (Note that since this is containerized, we use the traditional artifact syntax.)
    #
    # https://www.kubeflow.org/docs/components/pipelines/user-guides/data-handling/artifacts/

    with kfp.dsl.If(resume_from_training_outputs == False, name='run-full-pipeline'):
        # Normal execution path: Run preprocess, hyperopt, train
        xgb_preprocess_task = xgb_preprocess(
            gcp_config_filename=gcp_config_filename,
            model_config_filename=model_config_filename,
        )

        xgb_hyperopt_task = xgb_hyperopt(
            gcp_config_filename=gcp_config_filename,
            model_config_filename=model_config_filename,
            val_dataset=xgb_preprocess_task.outputs["val_dataset"],
        ).after(xgb_preprocess_task)

        xgb_train_task = xgb_train(
            gcp_config_filename=gcp_config_filename,
            model_config_filename=model_config_filename,
            train_dataset=xgb_preprocess_task.outputs["train_dataset"],
            test_dataset=xgb_preprocess_task.outputs["test_dataset"],
            hyperparameters=xgb_hyperopt_task.outputs["hyperparameters"],
        ).after(xgb_preprocess_task).after(xgb_hyperopt_task)

        xgb_evaluate_task = xgb_evaluate(
            gcp_config_filename=gcp_config_filename,
            model_config_filename=model_config_filename,
            trained_model=xgb_train_task.outputs["trained_model"],
            test_dataset=xgb_preprocess_task.outputs["test_dataset"],
        ).after(xgb_train_task)

        upload_to_model_registry_task = upload_to_model_registry(
            gcp_config_filename=gcp_config_filename,
            model_registry_config_filename=MR_CONFIG_FILENAME,
            trained_model=xgb_train_task.outputs["trained_model"],
            scalar_metrics=xgb_evaluate_task.outputs["scalar_metrics"],
            pipeline_job_name=kfp.dsl.PIPELINE_JOB_NAME_PLACEHOLDER
        ).after(xgb_evaluate_task)

        with kfp.dsl.If(deploy_model_to_endpoint == True):
            # Additional component for Online Prediction
            model_deployment(
                gcp_config_filename=gcp_config_filename,
                model_registry_config_filename=MR_CONFIG_FILENAME,
                uploaded_model=upload_to_model_registry_task.outputs["uploaded_model"],
            ).after(upload_to_model_registry_task)

    with kfp.dsl.Else(name='run-pipeline-after-training'):
        # Resume pipeline after training
        imported_trained_model = kfp.dsl.importer(
            artifact_uri=trained_model_gcs_uri,
            artifact_class=kfp.dsl.Artifact,
        ).set_display_name("Import Trained Model")

        imported_test_dataset = kfp.dsl.importer(
            artifact_uri=test_dataset_gcs_uri,
            artifact_class=kfp.dsl.Artifact,
        ).set_display_name("Import Test Dataset")

        xgb_evaluate_task = xgb_evaluate(
            gcp_config_filename=gcp_config_filename,
            model_config_filename=model_config_filename,
            trained_model=imported_trained_model.outputs["artifact"],
            test_dataset=imported_test_dataset.outputs["artifact"],
        ).after(imported_trained_model, imported_test_dataset)

        upload_to_model_registry_task = upload_to_model_registry(
            gcp_config_filename=gcp_config_filename,
            model_registry_config_filename=MR_CONFIG_FILENAME,
            trained_model=imported_trained_model.outputs["artifact"],
            scalar_metrics=xgb_evaluate_task.outputs["scalar_metrics"],
            pipeline_job_name=kfp.dsl.PIPELINE_JOB_NAME_PLACEHOLDER
        ).after(xgb_evaluate_task)

        with kfp.dsl.If(deploy_model_to_endpoint == True):
            # Additional component for Online Prediction
            model_deployment(
                gcp_config_filename=gcp_config_filename,
                model_registry_config_filename=MR_CONFIG_FILENAME,
                uploaded_model=upload_to_model_registry_task.outputs["uploaded_model"],
            ).after(upload_to_model_registry_task)
```

## Limitations with Local Pipeline Execution

While the Kubeflow Pipelines SDK is designed to allow for local development and testing of components and pipelines, there are specific limitations regarding certain advanced features, including conditional logic.

**Conditional logic (`dsl.Condition`) and parallel execution (`dsl.ParallelFor`) are not supported when running pipelines locally.**

**Reasoning for this limitation:**

The local execution environment in Kubeflow Pipelines is intended for rapid iteration on individual components and simpler pipeline structures. It relies on local Docker execution and does not replicate the full capabilities and complexity of a remote Kubernetes-based execution engine like Argo Workflows. Features such as sophisticated DAG manipulation, conditional branching, and parallel execution require the robust orchestration capabilities that are only available in a remote KFP deployment.

For a comprehensive explanation of local execution limitations, please refer to the official Kubeflow documentation:
[https://www.kubeflow.org/docs/components/pipelines/user-guides/core-functions/execute-kfp-pipelines-locally/](https://www.kubeflow.org/docs/components/pipelines/user-guides/core-functions/execute-kfp-pipelines-locally/)

**Encountering this limitation locally will result in an error similar to:**

```
NotImplementedError: 'dsl.Condition' is not supported by local pipeline execution.
```

**Therefore, if your pipeline utilizes `kfp.dsl.Condition`, `kfp.dsl.ParallelFor`, or `kfp.dsl.ExitHandler` for managing its execution flow, it *must* be compiled and run on a remote Kubeflow Pipelines or Vertex AI Pipelines environment.**
