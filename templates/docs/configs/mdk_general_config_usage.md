# MDK General Configuration Guide (`config.yml`)

## Introduction

This guide provides a comprehensive overview of the `config.yml` file used within the MDK framework. This file serves as the central "source of truth" for a specific machine learning model, defining its behavior across the entire MLOps lifecycle, from training and deployment to monitoring and CI/CD.

The MDK framework uses this file to configure its various components, ensuring that all parts of the pipeline operate with a consistent and version-controlled set of parameters.

## Core Concepts

Before diving into the specific keys, it's important to understand two core concepts that govern how this configuration is loaded and interpreted.

### 1. Environment-Specific Overrides

The `environments` block at the top of the file allows you to specify different values for certain keys depending on the active environment (e.g., `train`, `stage`, `prod`).

**How it works:**
1. The MDK framework identifies the current environment (usually from the `deployment_environment` key in your GCP configuration file).
2. It first loads all the default configurations from the main body of the `config.yml` file.
3. It then checks the `environments` block for a section matching the current environment. If found, any keys defined within that section will **override** the default values.

This powerful feature allows you to maintain a single configuration file while seamlessly adjusting parameters like dataset paths, model labels, and monitoring baselines for different stages of your MLOps process.

```yaml
environments:
  train: # These values are used when deployment_environment is 'train'
    model_registry:
      model_labels:
        status: development
    inference:
      datasets_path: "bq://...-train.ml_dataset.subset_no_labels"
  prod: # These values are used when deployment_environment is 'prod'
    model_registry:
      model_labels:
        status: production
    inference:
      datasets_path: "bq://...-prod.ml_dataset.subset_no_labels"
```

### 2. Separation of Concerns: `config.yml` vs. `state/<env>.yml`

The MDK framework separates two types of configuration:

1.  **`config.yml` (This File):** Contains **application and model logic**. It defines *what* your model is, *how* it should be trained, and *how* it should be deployed and monitored. These are parameters you would typically version control with your model code.
2.  **`state/<env>.yml` (GCP Configuration):** Contains **infrastructure and environment state**. It defines *where* things run, including your GCP Project ID, GCS buckets, service accounts, and artifact repositories. This file maps to the `GCPConfig` Pydantic model and is kept separate as infrastructure details may be managed independently of model code.

---

## Configuration Reference

This section provides a detailed reference for each top-level key in `config.yml`.

### `general` - General Model Configuration

This section defines the core identity of your model, used by all MDK components. It roughly corresponds to the `ModelReferenceConfig` Pydantic model.

| Key | Description | Required? | Example Value |
| :--- | :--- | :--- | :--- |
| `model_name` | The unique, human-readable name for your model. This is used as the display name in Vertex AI Model Registry and Endpoints. | **Yes** | `"xgb_example_model"` |
| `model_inference_reference` | Defines the strategy for retrieving a specific model version for inference, deployment, or monitoring. | **Yes** | `"latest"` |

#### Model Inference Reference Strategies

The `model_inference_reference` key supports the following strategies:

*   **`"latest"`**: Retrieves the most recently trained version of the model from the Expanded Model Registry.
*   **`"primary"`**: Retrieves the model version currently marked as the "champion" or primary serving model in the Expanded Model Registry.
*   **Semantic Version (e.g., `"1.2.3"`)**: Retrieves the model version matching the specified semantic version string.
*   **GCS URI (e.g., `"gs://..."`)**: Bypasses the model registry and uses the model artifacts directly from the specified Google Cloud Storage path.

### `training` - Training Pipeline Configuration

This section is **free-form** and contains parameters that are passed directly as command-line arguments to your training script. The keys and values you define here should correspond to the arguments expected by your training code. Some examples are shown below.

| Key | Description | Required? | Example Value |
| :--- | :--- | :--- | :--- |
| `model_filename` | The name of the file to save the trained model as. | **Yes** | `"model.pkl"` |
| `target_column` | The name of the label column in your dataset. | No | `"ClassIndex"` |
| `val_size` | The proportion of the dataset for the validation split. | No  | `0.2` |
| `n_estimators` | A hyperparameter for your model. | No | `100` |
| `run_shap_analysis` | A boolean flag to control a step in your training script. | No | `True` |
| `cron_schedule` | Cron schedule for re-running the training job. Use a prefix to apply a timezone: "TZ=America/New_York 1 * * * *" | No | `"0 0 * * 0"` |

If you specify a `cron_schedule`, the training pipeline will run according to this schedule. Tagging scheduled pipeline jobs to experiments is not currently supported by Vertex AI. If you wish to see the input parameters and outputs related to a scheduled pipeline job, please look under the pipeline job in the console directly rather than in the experiments page.

### `inference` - Batch Inference Configuration

This section is **free-form** and defines parameters for batch prediction jobs. The keys and values here should align with the arguments expected by your batch inference script. Some examples are shown below.

| Key | Description | Required? | Example Value |
| :--- | :--- | :--- | :--- |
| `datasets_path` | The GCS or BigQuery URI of the dataset to run predictions on. | **Yes** | `"bq://...train.ml_dataset.subset_no_labels"` |
| `bq_output_table` | The destination BigQuery table for prediction results. | **Yes** | `"bq://...train.ml_dataset.xgb_predictions"` |
| `run_monitoring` | A flag to control whether a model monitoring job should be run after inference. | No | `True` |
| `class_names` | An ordered list of class names to map model output. | No | `["BARBUNYA", "BOMBAY", ...]` |
| `cron_schedule` | Cron schedule for re-running the inference job. Use a prefix to apply a timezone: "TZ=America/New_York 1 * * * *" | No | `"0 0 * * 0"` |

If you specify a `cron_schedule`, the batch pipeline will run according to this schedule. Tagging scheduled pipeline jobs to experiments is not currently supported by Vertex AI. If you wish to see the input parameters and outputs related to a scheduled pipeline job, please look under the pipeline job in the console directly rather than in the experiments page.

### `model_registry` - Model Registry Metadata

This section configures the metadata used when registering a new model version with both the Vertex AI Model Registry and the custom Expanded Model Registry.

| Key | Description | Required? | Example Value |
| :--- | :--- | :--- | :--- |
| `major_version` | The major version number for the Expanded Model Registry. | **Yes** | `0` |
| `minor_version` | The minor version number for the Expanded Model Registry. | **Yes** | `1` |
| `model_version_aliases` | A list of string aliases for this model version in Vertex AI (e.g., `"best-performer"`). | No | `[]` |
| `model_version_description` | A detailed Markdown description of the model version. | No | `"This model is an XGBoost Classifier..."` |
| `serving_container_image_uri` | The pre-built or custom container image for serving the model on Vertex AI. | No | `"us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.2-1:latest"` |
| `serving_container_ports` | A list of ports exposed by the serving container. | No | `[8080]` |
| `model_labels` | A dictionary of key-value pairs to organize and filter models in Vertex AI. | No | `{"team": "my-team", "task": "classification"}` |
| `path_to_production_projects`| A mapping of environments to GCP project IDs for tracking model promotions. | No | `{"train": "...", "stage": "...", "prod": "..."}` |
| `is_sensitive_data`| A boolean flag indicating if the model was trained on sensitive data. | No | `False` |
| `model_status` | The lifecycle status of the model (e.g., `training`, `staged`, `production`). | No | `"training"` |
| `publish_status`| The publish status in the Expanded Model Registry (e.g., `champion`, `challenger`). | No | `"challenger"` |
| `is_primary_deployment`| Indicates if this version is the primary serving model in an environment. | No | `False` |

### `deployment` - Model Deployment Configuration

This section controls how the model is deployed to a Vertex AI Endpoint for online serving. It maps to the `DeploymentConfig` Pydantic model.

| Key | Description | Required? | Example Value |
| :--- | :--- | :--- | :--- |
| `endpoint_name` | The display name of the Vertex AI Endpoint. If omitted, defaults to `general.model_name`. | No | `"xgb_endpoint_standard"` |
| `machine_type` | The machine type for the deployed model (e.g., `'n1-standard-2'`). | No | `"n2-standard-4"` |
| `min_replica_count`| The minimum number of compute instances for the model. | No | `1` |
| `max_replica_count`| The maximum number of compute instances to scale up to. | No | `1` |
| `is_primary_deployment`| If `True`, deploys the model with 100% of the traffic, demoting any existing models. | No | `False` |
| `shadow_mode`| If `True`, deploys the model to a new, separate endpoint for shadow testing (no production traffic). | No | `False` |
| `traffic_split`| An explicit traffic split. Overrides all other traffic logic. Keys are deployed model IDs, with `"0"` as a placeholder for the new model. | No | `{"77...776": 90, "0": 10}` |
| `run_monitoring` | A flag to control whether a model monitoring job should set up on the endpoint. | No | `False` |

### `model_monitoring` - Model Monitoring Configuration

This section configures Vertex AI Model Monitoring jobs to detect data skew and drift. It maps to the `ModelMonitoringConfig` Pydantic model.

| Key | Description | Required? | Example Value |
| :--- | :--- | :--- | :--- |
| `target_dataset_uri` | The GCS or BigQuery URI of the recent prediction data to compare against the baseline. Used for batch inference. | No | `"bq://...subset_no_labels"` |
| `target_endpoint` | The full resource name of the deployed Vertex AI Endpoint to monitor. Used for real-time inference. | No | `"projects/PROJECT_NUMBER/locations/REGION/endpoints/ENDPOINT_ID"` |
| `target_dataset_query` | Standard SQL for BigQuery to be used instead of the target_dataset_uri. | No | `"SELECT * FROM ..."` |
| `window` | The time window for collecting endpoint logs for real-time monitoring (e.g., '24h'). Format: 'w\|W': Week, 'd\|D': Day, 'h\|H' | No | `"24h"` |
| `feature_fields_schema_map`| **Required.** A mapping of feature names to their data types (`integer`, `float`, `string`, etc.). Available field types can be found [here](https://docs.cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform_v1beta1.types.ModelMonitoringSchema.FieldSchema). | **Yes** | `{"Area": "integer", "Perimeter": "float", ...}` |
| `model_monitor_job_display_name` | The display name for the monitoring job. A timestamp is often appended. | No | `"xgb-example-monitoring-job-"` |
| `model_monitor_display_name`| The display name for the persistent `ModelMonitor` resource in Vertex AI. | No | `"xgb-example-model-monitor"` |
| `cron_schedule` | An optional cron string (e.g., `"0 0 * * *"`) to run the monitoring job on a schedule. If omitted, the job runs once on-demand. | No | `None` |
| `user_emails` | A list of email addresses to receive monitoring alerts. | No | `['no-reply@...']` |

One of `target_dataset_uri`, `target_endpoint`, or `target_dataset_query`. If they are all specified, the Monitor will use the `target_dataset_uri` as the target input.

If you are doing monitoring for batch inference, specify the `target_dataset_uri` and leave `target_endpoint`. Likewise, if you are doing real-time inference, specify `target_endpoint` and leave `target_dataset_uri`. If both are specified, the MDK will assume you are trying to do real-time inference and will use the `target_endpoint`. Lastly, if you are running the deployment pipeline, you can leave the `target_endpoint` field blank, and it will use the endpoint that is created and deployed as part of this pipeline.

Specify the `cron_schedule` if you are doing real-time inference monitoring as this is a recurring monitoring workflow.


#### `feature_drift` and `prediction_drift` (Sub-sections of `model_monitoring`)

These sections configure the specific thresholds and metrics for drift detection. They use default values if not specified.

| Key | Description | Default | Example Value |
| :--- | :--- | :--- | :--- |
| `default_categorical_alert_threshold` | The threshold (0.0 to 1.0) for drift in categorical features. | `0.1` | `0.1` |
| `default_numeric_alert_threshold`| The threshold (0.0 to 1.0) for drift in numeric features. | `0.1` | `0.1` |
| `categorical_metric_type` | The distance measure for categorical features. | `"l_infinity"` | `"l_infinity"` |
| `numeric_metric_type` | The distance measure for numeric features. | `"jensen_shannon_divergence"` | `"jensen_shannon_divergence"`|

#### `retraining` (Sub-section of `model_monitoring`)

This configures the automatic retraining trigger in response to a monitoring alert.

| Key | Description | Required? | Example Value |
| :--- | :--- | :--- | :--- |
| `set_up_retraining` | If `True`, the MDK will set up the necessary artifacts to enable automatic retraining. | No | `True` |
| `training_pipeline_name` | The name of the KFP training pipeline to trigger. | **Yes**, if `set_up_retraining` is `True`. | `"xgb_training_pipeline"` |
| `inference_pipeline_name`| The name of the KFP inference pipeline. | No | `"xgb_inference_pipeline"` |
| `app_root` | The root directory of the application within the retraining trigger's container. | No | `"/app"` |

### `ci_cd` - CI/CD Configuration

This section is **free-form** and intended to hold parameters for your CI/CD system (e.g., Github Actions).

| Key | Description | Example Value |
| :--- | :--- | :--- |
| `pipeline_names` | A list of pipelines to execute during a CI/CD trigger. | `["xgb_training_pipeline"]` |
