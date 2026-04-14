# How to Use the Model Registry Library

This document outlines the two primary ways to use the `mdk.model.registry` package: as a library in your Python code and as a standalone command-line tool.

## 1. Using it as a Library (The Intended Way)

This is the most flexible approach, allowing you to integrate the model registration process into larger automation scripts, such as Vertex AI Pipeline components. The library exposes a simple `upload_model` function and a `RegistryAppConfig` model for configuration.

### Example: `your_pipeline_component.py`

```python
import logging
from pydantic import ValidationError

# Import the public API and configuration model from the new location
from mdk.model.registry import upload_model
from mdk.model.registry.models import RegistryAppConfig

# The application using the library is responsible for configuring logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Define paths to your configuration files and other parameters
ENVIRONMENT = "train"
GCP_CONFIG_PATH = f"model_products/main_product/state/{ENVIRONMENT}.yml"
GENERAL_CONFIG_PATH = "model_products/main_product/config/config.yml"
ARTIFACT_URI = "gs://your-bucket/path/to/model/folder/"
PERFORMANCE_METRICS = {"auc": 0.95, "accuracy": 0.92}

try:
    # The RegistryAppConfig model loads, combines, and validates all configurations
    # from your YAML files. This is the recommended way to create the config.
    print("Loading and validating configuration...")
    app_config = RegistryAppConfig.from_yaml_files(
        gcp_config_path=GCP_CONFIG_PATH,
        general_config_path=GENERAL_CONFIG_PATH,
    )

    print("Configuration loaded. Uploading model to registry...")

    # Call the high-level facade function with all necessary parameters
    # The orchestrator handles the rest.
    uploaded_model_resource_name = upload_model(
        config=app_config,
        artifact_folder_uri=ARTIFACT_URI,
        registry_provider_name="vertex",
        performance_metrics_summary=PERFORMANCE_METRICS,
        vertex_ai_pipeline_job_run_id="pipeline-run-12345"
    )

    print("\n--- Model Upload Succeeded ---")
    print(f"Vertex AI Model Resource Name: {uploaded_model_resource_name}")

except FileNotFoundError as e:
    print(f"Error: A configuration file was not found: {e.filename}")
except ValidationError as e:
    print(f"Error: Configuration is invalid. Please check your YAML files.\n{e}")
except Exception as e:
    print(f"An unexpected error occurred during the upload: {e}")
```

## 2. Using it from the Command Line (for Quick Execution)

The package includes a command-line interface (`cli.py`) for easy execution from a shell. This is ideal for:

- Simple automation
- Testing
- Manual runs without writing additional Python code

The script handles loading and validating the same YAML configuration files used by the library interface.

### 🔧 Basic Usage

Run the following command from the project root:

```bash
# This command runs the cli.py script as a module from the project root.
# Provide the paths to your configuration files and the model artifact URI.

python -m mdk.model.registry.cli \
  --gcp-config "model_products/main_product/state/train.yml" \
  --general-config "model_products/main_product/config/config.yml" \
  --artifact-uri "gs://your-bucket/path/to/model/folder/" \
  --provider "vertex" \
  --metrics-file "results/metrics.json" \
  --pipeline-job-id "projects/123/locations/us-central1/pipelineJobs/pipeline-123"
```

### 📄 Expected Output from CLI

```
INFO: Starting model upload process with provider: 'vertex_ai'...
INFO: Uploading model 'my-cool-model' to Vertex AI Model Registry...
INFO: Successfully uploaded model to Vertex AI.
INFO: Preparing payload for the expanded model registry...
INFO: Uploading to expanded model registry at: https://...
INFO: Successfully registered model with the expanded model registry.

--- Model Upload Succeeded ---
Provider: vertex_ai
Model Resource Name: projects/123/locations/us-central1/models/456
Model Version ID: 1
--------------------------------
```
