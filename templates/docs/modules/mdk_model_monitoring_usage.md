# How to Use the Monitoring Library

This document outlines the two primary ways to use the `monitoring` package: as a library in your Python code and as a standalone command-line tool.

### 1. Using it from another Python script (the intended way)

This is the most flexible approach, allowing you to integrate monitoring setup into larger automation scripts or applications. The library exposes a simple `set_up_monitoring` function and a `MonitoringAppConfig` model for configuration.

```python
# in another_script.py
import logging
from pydantic import ValidationError

# Import the public API and configuration model
from mdk.monitoring import set_up_monitoring
from mdk.monitoring.models import MonitoringAppConfig

# The application using the library is responsible for configuring logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Define paths to your configuration files
ENVIRONMENT = "train"
GCP_CONFIG_PATH = f"model_products/main_product/state/{ENVIRONMENT}.yml"
GENERAL_CONFIG_PATH = "model_products/main_product/config/config.yml"

try:
    # Use the MonitoringAppConfig model to load, combine, and validate all configurations
    # from your YAML files. This is the recommended way to create the config.
    print("Loading and validating configuration...")
    app_config = MonitoringAppConfig.from_yaml_files(
        gcp_config_path=GCP_CONFIG_PATH,
        general_config_path=GENERAL_CONFIG_PATH,
    )

    print("Configuration loaded. Setting up model monitoring...")

    # Call the high-level facade function
    job_url, schedule_url = set_up_monitoring(config=app_config)

    print("\n--- Model Monitoring Setup Succeeded ---")
    print(f"Vertex AI Console URL: {job_url}")

except FileNotFoundError as e:
    print(f"Error: A configuration file was not found: {e.filename}")
except ValidationError as e:
    print(f"Error: Configuration is invalid. Please check your YAML files.\n{e}")
except Exception as e:
    print(f"An unexpected error occurred during setup: {e}")

```

### 2. Using it from the Command Line (for quick execution)

The package includes a command-line interface (`cli.py`) for easy execution from a shell. This is ideal for simple automation, testing, or manual runs.

The script handles loading and validating the same YAML configuration files.

```bash
# This command runs the cli.py script as a module from the project root.
# Provide the paths to your configuration files and the target environment.

python -m monitoring.cli \
  --gcp-config "config/gcp_config.yml" \
  --general-config "model_products/main_product/config/config.yml"

```

**Expected Output from CLI:**

```
INFO: Loading and validating configuration files...
INFO: Configuration validated successfully.
INFO: Starting model monitoring setup using 'gcp_vertex_ai' provider.
INFO: Initializing Vertex AI for project 'your-gcp-project' in 'us-central1'
... (other logs from the service) ...
INFO: Model monitoring setup completed successfully.

--- Model Monitoring Setup Succeeded ---
Vertex AI Console URL: https://console.cloud.google.com/vertex-ai/model-monitoring/...
--------------------------------------
```

## Real-time Inference Monitoring Debugging

There are 2 potential scenarios to be aware of and avoid when using model monitoring for real-time inference:

- **Scenario 1: Race Condition During Monitoring Setup**
A race condition can occur if the deployment and monitoring components in your pipeline both query for the "latest" model independently.
  -  *Problem:*
     -  The Deployment component fetches Model_A (the "latest" model) and deploys it to the endpoint.
     -  After deployment but before monitoring setup, a new Model_B is registered, becoming the new "latest."
     -  The Monitoring component runs, fetches the "latest" model (which is now Model_B), and sets up a monitor for it.
  -  *Result:* You have a mismatch. The endpoint is serving Model_A, but the monitor is configured to watch Model_B, which is not deployed on this endpoint (it only exists in the Model Registry).
  - *Resolution:* Either be very careful when using "latest" for the model inference reference (e.g. do not run training jobs in between setting up monitoring for a given model name), or use the semantic model version of the model_name for more assurance.


- **Scenario 2: Monitoring Persists on Old Model Version After Re-deployment**
Model monitoring jobs are configured against a specific model version, not the endpoint resource itself.
  -  *Problem:*
     -  You deploy Model_v1 to my-endpoint and configure Monitor_v1 for it.
     -  Later, you re-deploy a new Model_v2 to the same endpoint resource (my-endpoint).
     -  The endpoint now serves Model_v2, and all new traffic is sent to it.
  -  *Result:* The original Monitor_v1 is still associated with Model_v1, which is no longer receiving traffic. The new traffic to Model_v2 is not monitored.
  - *Resolution:* Reset the monitoring job to use the correct model version, or deploy the endpoint to have traffic routed to the current model version for the endpoint.
