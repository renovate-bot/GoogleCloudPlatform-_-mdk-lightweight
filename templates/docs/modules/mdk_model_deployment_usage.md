# How to Use the Model Deployment Library

This document outlines the two primary ways to use the `deployment` package: as a library in your Python code and as a standalone command-line tool.

### 1. Using it from another Python script (the intended way)

This is the most flexible approach, allowing you to integrate model deployment into larger MLOps pipelines or automation scripts. The library exposes a simple `deploy_model` function and a `DeploymentAppConfig` model for configuration.

```python
# in your_pipeline_script.py
import logging
from pydantic import ValidationError

# Import the public API and configuration model
from mdk.model.deployment import deploy_model
from mdk.model.deployment.models import DeploymentAppConfig

logger = logging.getLogger(__name__)

# Define paths to your configuration files
ENVIRONMENT = "train"
GCP_CONFIG_PATH = f"model_products/main_product/state/{ENVIRONMENT}.yml"
GENERAL_CONFIG_PATH = "model_products/main_product/config/config.yml"

try:
    # Use the DeploymentAppConfig model to load, combine, and validate all configurations
    # from your YAML files. This is the recommended way to create the config.
    logger.info("Loading and validating deployment configuration...")
    app_config = DeploymentAppConfig.from_yaml_files(
        gcp_config_path=GCP_CONFIG_PATH,
        general_config_path=GENERAL_CONFIG_PATH,
    )

    logger.info("Configuration loaded. Starting model deployment...")

    # Call the high-level facade function
    endpoint_resource_name = deploy_model(config=app_config)

    logger.info("\n--- Model Deployment Succeeded ---")
    logger.info(f"Endpoint Resource Name: {endpoint_resource_name}")

except FileNotFoundError as e:
    logger.error(f"Error: A configuration file was not found: {e.filename}")
except ValidationError as e:
    logger.error(f"Error: Configuration is invalid. Please check your YAML files.\n{e}")
except Exception as e:
    logger.error(f"An unexpected error occurred during deployment: {e}")

```

### 2. Using it from the Command Line (for quick execution)

The package includes a command-line interface (`cli.py`) for easy execution from a shell. This is ideal for simple automation, testing, or manual deployments.

The script handles loading and validating the same YAML configuration files.

```bash
# This command runs the cli.py script as a module from the project root.
# Provide the paths to your configuration files.

python -m mdk.model.deployment.cli \
  --gcp-config "model_products/main_product/state/train.yml" \
  --general-config "model_products/main_product/config/config.yml"

```

**Expected Output from CLI:**

```
INFO: Loading and validating configuration files...
INFO: Configuration validated successfully.
INFO: Starting model deployment using 'gcp_vertex_ai' provider.
INFO: Initializing Vertex AI for project 'your-gcp-project' in 'us-central1'
... (other logs from the service, e.g., searching for endpoint, deploying, updating registry) ...
INFO: Model deployment completed successfully.

--- Model Deployment Succeeded ---
Endpoint Resource Name: projects/your-gcp-project/locations/us-central1/endpoints/1234567890
--------------------------------
```
