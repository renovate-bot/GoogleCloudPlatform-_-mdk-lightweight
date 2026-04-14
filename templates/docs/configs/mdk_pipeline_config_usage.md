# MDK Pipeline Configuration Guide (`pipeline_config.yml`)
## Introduction

This guide provides a comprehensive overview of the `pipeline_config.yml` file. This file acts as the central registry for a given "model product," defining references to all the reusable building blocks—**images**, **components**, and **pipelines**—that constitute your MLOps workflows.

While this file can be edited, it is generally considered an **advanced configuration**. Most day-to-day work involves adjusting the `config.yml` file, not this one. You should only edit `pipeline_config.yml` when you are:

*   Adding a new reusable custom Docker image.
*   Defining a new Kubeflow Pipelines (KFP) component.
*   Defining a new KFP pipeline.
*   Configuring an existing component to use a specialized container or hardware (the "Bring-Your-Own-Container" or BYOC pattern).
## Core Concepts

The MDK framework uses this file to understand the structure of your pipelines. The three main sections work together:

1.  **`images`**: Defines the location to custom Docker images that can be built and used by your components. This is the foundation.
2.  **`components`**: Defines the location to individual steps in your pipeline (e.g., `xgb_train`, `xgb_evaluate`). Each component is mapped to a specific Docker image.
3.  **`pipelines`**: Defines the location to the end-to-end workflow by assembling components into a directed acyclic graph (DAG).

The MDK build and compilation tools (`build_images.py`, `compile_pipeline.py`) read this file to automate the process of building container images and compiling KFP-compatible YAML specifications.

---
## Configuration Reference
### `images`

This section registers custom Docker images that the MDK can build. Each entry represents a unique image.

| Key | Description | Required? | Example Value |
| :--- | :--- | :--- | :--- |
| `artifact` | The image name and tag that will be used within this configuration file and in Artifact Registry. | **Yes** | `"standard:latest"` |
| `build_config_dir` | The path to the directory containing the `Dockerfile` and any other build-time assets for this image. | **Yes** | `"model_products/main_product/images/standard"` |

Note: The `standard` image is required and should not be removed.
**To add a new buildable image:**

1.  Create a new directory (e.g., `images/my_new_image`) containing your `Dockerfile`.
2.  Add a new entry under the `images` key in `pipeline_config.yml`:

```yaml
images:
  standard:
    artifact: "standard:latest"
    build_config_dir: "model_products/main_product/images/standard"
  my_new_image: # Your new image entry
    artifact: "my-image:1.0"
    build_config_dir: "model_products/main_product/images/my_new_image"
```
### `pipelines`

This section registers your end-to-end Kubeflow Pipelines. Each entry corresponds to a Python script that defines a KFP pipeline.

| Key | Description | Required? | Example Value |
| :--- | :--- | :--- | :--- |
| `description` | A human-readable description of the pipeline's purpose. | No | `"Sample pipeline for XGBoost training."` |
| `function` | The name of the Python function decorated with `@dsl.pipeline` inside your module. | **Yes** | `"pipeline"` |
| `module_path` | The path to the Python file that defines the pipeline function. | **Yes** | `"examples/model_products/xgb_example/pipelines/xgb_training_pipeline/pipeline.py"` |
### `components`

This section registers the individual KFP components that are the building blocks of your pipelines.

| Key | Description | Required? | Example Value |
| :--- | :--- | :--- | :--- |
| `description` | A human-readable description of the component's purpose. | No | `"Model training workflow"` |
| `function` | The name of the Python function decorated with `@dsl.component` inside your module. | **Yes** | `"xgb_train"` |
| `module_path` | The path to the Python file that defines the component. The filename must be `component.py`. | **Yes** | `"examples/model_products/xgb_example/components/xgb_train/component.py"` |
| `image_artifact`| Specifies the Docker image to use for this component's runtime environment. See details below. | **Yes** | `"standard:latest"` or a full URI. |
| `cpu` | **(Optional)** The number of vCPUs to allocate for the component. Expects an integer or string (e.g., `4`, `"8"`). Vertex AI defaults to using an `e2-standard-4` which has `4` vCPUs. | `No` | `cpu: 8` |
| `mem` | **(Optional)** The amount of memory to allocate for the component. Expects a number or string (e.g., `16`, `"32G"`, `"64G"`). If a number is provided (e.g., `32`), the MDK will automatically append `"G"` (resulting in `"32G"`). Vertex AI defaults to using an `e2-standard-4` which has `16G` memory. | `No` | `mem: 32` |
| `accelerator_type` | **(Optional)** The type of GPU to attach (e.g., `"NVIDIA_TESLA_T4"`, `"NVIDIA_TESLA_V100"`). | `No` | `accelerator_type: "NVIDIA_TESLA_T4"` |
| `accelerator_limit` | **(Optional)** The number of GPUs to attach. Expects an integer (e.g., `1`). | `No` | `accelerator_limit: 1` |
| `selector_constraint` | **(Optional)** A Kubernetes node selector constraint, used to schedule the component on specific nodes or node pools. This can be a string (e.g., `"cloud.google.com/gke-accelerator: nvidia-tesla-t4"`) or just the value (e.g., `'nvidia-tesla-t4'`). | `No` | `selector_constraint: 'nvidia-tesla-t4'` |

For more details around `cpu`, `mem`, `accelerator_type`, `accelerator_limit`, and `selector_constraint`, please see the [Vertex AI documentation](https://docs.cloud.google.com/vertex-ai/docs/pipelines/machine-types).
#### The `image_artifact` Key

This key is flexible and supports two primary ways of specifying an image:

1.  **Short Image Reference**: Use the `artifact` name defined in the `images` section (e.g., `"standard:latest"`). The MDK will build this image and use it when compiling the component.
2.  **Full Image Reference**: Provide a complete, direct URI to an image in any container registry (e.g., Artifact Registry, Docker Hub, GCR). This is common for using pre-built public images, such as those provided by Google for Vertex AI Training.
    *   **Example**: `"us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-17.py310:latest"`

### Configuring Resources for Standard Components

For components that run directly within the Kubeflow Pipelines cluster (i.e., not as a Vertex AI Custom Job via `container_specs`), you can configure their hardware resources directly within the component's definition in `pipeline_config.yml`. This allows you to request more CPU, memory, or specific accelerators for computationally intensive steps.

The MDK internally reads these settings and applies them using the component's `set_cpu_limit`, `set_memory_limit`, `set_accelerator_type`, `set_accelerator_limit`, and `add_node_selector_constraint` methods.

Here's how each resource parameter is configured:

*   **`cpu`**:
    *   **Purpose**: Sets the number of virtual CPUs (vCPUs) to allocate for the component.
    *   **Format**: An integer or a string representing an integer (e.g., `4`, `"8"`).
    *   **Example**: `cpu: 8`
    *   **Default**: Vertex AI defaults to using an `e2-standard-4` which has `4` vCPUs.

*   **`mem` (Memory)**:
    *   **Purpose**: Sets the amount of RAM (memory) allocated to the component.
    *   **Format**: A number or a string.
        *   If a plain number is provided (e.g., `32`), the MDK automatically appends `"G"` (e.g., it becomes `"32G"`).
        *   If units are explicitly included in a string (e.g., `"16G"`, `"32G"`, `"512M"`), the string will be used as is.
    *   **Example**:
        ```yaml
        mem: 32      # Interpreted as "32G"
        mem: "64G"  # Interpreted as "64G"
        ```
    *   **Default**: Vertex AI defaults to using an `e2-standard-4` which has `16G` memory.

*   **`accelerator_type`**:
    *   **Purpose**: Specifies the type of GPU to attach to the component.
    *   **Example**: `"NVIDIA_TESLA_T4"`, `"NVIDIA_TESLA_V100"`
    *   **Note**: If an accelerator type is specified, you must also specify `accelerator_limit`.

*   **`accelerator_limit`**:
    *   **Purpose**: Sets the number of GPUs to attach of the specified `accelerator_type`.
    *   **Format**: An integer (e.g., `1`, `2`).
    *   **Example**: `accelerator_limit: 1`

*   **`selector_constraint`**:
    *   **Purpose**: Allows you to specify Kubernetes node selector labels. This is useful for targeting specific hardware (e.g., GPU-enabled nodes) or node pools within your cluster.
    *   **Format**: Can be a string or a dictionary.
        *   String example: `"cloud.google.com/gke-accelerator: nvidia-tesla-t4"`
        *   Dictionary example:
            ```yaml
            selector_constraint: "cloud.google.com/gke-accelerator: nvidia-tesla-t4"
            ```

#### Example: Configuring Resources for a Standard Component

Here's an example demonstrating how to configure various hardware resources for a component that will run on the standard KFP cluster:

```yaml
components:
  my_data_processing_step:
    description: "A data processing step requiring more CPU and memory"
    function: process_data_func
    module_path: components/my_data_processing_step/component.py
    image_artifact: "standard:latest"

    # Configure resources directly under the component definition
    cpu: 16       # Request 16 vCPUs
    mem: 64       # Request 64G of memory

  my_gpu_inference_step:
    description: "A component for GPU-accelerated inference"
    function: run_inference_func
    module_path: components/my_gpu_inference_step/component.py
    image_artifact: "standard:latest"

    cpu: 8
    mem: "32G"   # Explicitly 32 Gibibytes
    accelerator_type: "NVIDIA_TESLA_T4"
    accelerator_limit: 1
    selector_constraint:
      cloud.google.com/gke-accelerator: nvidia-tesla-t4
```

---
## Deep Dive: Bring-Your-Own-Container (BYOC) and Custom Jobs

The MDK provides a powerful "Bring-Your-Own-Container" (BYOC) feature that allows you to offload the execution of **any KFP component** to a fully configurable **Vertex AI Custom Job**. This is ideal for tasks that require:

*   Specialized hardware (e.g., GPUs, high-memory machines).
*   A specific, pre-built environment (e.g., Google's TensorFlow or PyTorch training containers).
*   Heavy computational workloads that you don't want running on the standard KFP cluster node pools.
### Enabling BYOC with `container_specs`

You enable BYOC for a component by adding a `container_specs` block to its definition in `pipeline_config.yml`. The presence of this block signals to the MDK that this component's logic should be executed as a Vertex AI Custom Job.
### The `container_specs` Block

This block allows you to configure every aspect of the Vertex AI Custom Job. The available parameters correspond directly to the MDK's Pydantic models for custom jobs.

| Parameter | Description |
| :--- | :--- |
| **Execution Control** | |
| `script_path` | Path to a Python script *outside the container* to be copied and executed in the container. Use this for script-based jobs. |
| `command` | A list of strings defining the command and its arguments to run in the container (e.g., `["python", "trainer.py"]`). Use this for direct command jobs. |
| `requirements`| A list of Python packages to be installed via `pip` before `script_path` is run. To use this, the `HTTPS_PROXY` env var must be set to enable pip installing. |
| `python_module_name` | The name of the Python module to run if `script_path` is part of a package. |
| **Hardware Configuration** | |
| `machine_type` | The machine type for the job (e.g., `"n1-standard-8"`, `"a2-highgpu-1g"`). Defaults to `"n1-standard-4"`. |
| `accelerator_type` | The type of GPU to attach (e.g., `"NVIDIA_TESLA_T4"`, `"NVIDIA_TESLA_V100"`). |
| `accelerator_count`| The number of GPUs to attach. Defaults to `1` if `accelerator_type` is specified. |
| `replica_count`| The number of worker replicas for the job. Defaults to `1`. |
| **Job Environment & Metadata** | |
| `env_vars` | A dictionary of environment variables to set inside the container. Should be in the form: `{"key", "value"}` |
| `labels` | A dictionary of labels to apply to the Vertex AI Custom Job for organization and filtering. |
| `service_account`| The service account the job will run as. Overrides the default from `GCPConfig`.|
| `network` | The VPC network to run the job in. |
| `timeout` | The job timeout in seconds. |
| `base_output_dir` | GCS directory for job outputs. The MDK sets this automatically. |
| `staging_bucket`| GCS bucket for staging job artifacts. The MDK sets this automatically. |
| **Advanced Settings** | |
| `persistent_resource_id` | The ID of a persistent resource to run the job on for faster startup. |
| `enable_web_access` | If `True`, enables web access to the training container. |
| `experiment` / `experiment_run` | The Vertex AI Experiment and Experiment Run to associate the job with. |
| `tensorboard` | The resource name of a TensorBoard instance to link to the job. |

Any container arguments are pulled from the corresponding section within the general `config.yml` and turned into flags for the `args` section of the container. For example, if you specify `model_filename: 'model.pkl'` within the `training` portion of `config.yml`, this will be converted to `['--model-filename', 'model.pkl']` args for the container. This allows you to have complete control over arguments for BYOC. The section within the `config.yml` to define the arguments must match the component you want to run as a Custom Job (e.g. `train` uses the values under the `training` section).
### BYOC Scenarios

Here are three common scenarios for using BYOC, illustrating the flexibility of the `container_specs` and `image_artifact` keys.
#### Scenario 1: Direct Command with a Pre-built Public Image
**Goal**: Run a simple shell command inside a standard, pre-built container from Google.

```yaml
components:
  xgb_train:
    description: "Model training workflow"
    function: xgb_train
    module_path: examples/model_products/xgb_example/components/xgb_train/component.py
    image_artifact: "us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-17.py310:latest" # Full URI to a public image
    container_specs:
      command:
      - bash
      - -c
      - echo 'hello world from a pre-built container!'
```
*   **`image_artifact`**: A full URI is provided, so the MDK will not try to build this image.
*   **`container_specs.command`**: This exact command will be executed as the entrypoint for the Vertex AI Custom Job. The component's training config and `input_artifact_map` will be passed as command-line arguments.
#### Scenario 2: Python Script with a Pre-built Public Image
**Goal**: Run a custom Python script, but leverage a pre-built environment from Google to avoid managing dependencies in a custom Dockerfile.

```yaml
components:
  xgb_train:
    description: "Model training workflow"
    function: xgb_train
    module_path: examples/model_products/xgb_example/components/xgb_train/component.py
    image_artifact: "us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-17.py310:latest" # Full URI
    container_specs:
      script_path: src/model_workflow/demo_byoc.py # Your script
      requirements: # Dependencies to install on the fly
        - "xgboost==1.7.5"
        - "pandas"
```
*   **How it works**: The MDK stages your `script_path` and uses it as the entrypoint for the Custom Job. Before your script runs, the packages listed in `requirements` are installed. This is a great balance between customization and convenience. Note that you must have your proxy settings set up to enable pip installing within the container.
#### Scenario 3: Direct Command with a Custom-Built Local Image
**Goal**: Run a command within a custom Docker image that you have defined and built using the MDK.

```yaml
images:
  byoc: # Image is registered here
    artifact: "byoc:latest"
    build_config_dir: "model_products/main_product/images/byoc"

components:
  xgb_train:
    description: "Model training workflow"
    function: xgb_train
    module_path: examples/model_products/xgb_example/components/xgb_train/component.py
    image_artifact: "byoc:latest" # Reference to the image registered above
    container_specs:
      command:
      - python
      - trainer.py # A script baked into your 'byoc:latest' image
```
*   **How it works**: `build_images.py` will build the image defined in the `images` section. `compile_pipeline.py` and `handle_custom_job_if_configured` will then use this custom-built image for the Vertex AI Custom Job, executing the specified `command`.
### Input/Output Contract for Custom Jobs

For your BYOC container to integrate seamlessly with Kubeflow Pipelines, it must correctly handle inputs and outputs. The MDK automatically passes KFP artifacts and configurations as command-line arguments to your container. Your script inside the container must be able to parse these arguments and write its outputs to a specific GCS location.

For a detailed guide on this contract, including example container scripts, please refer to the **"Input/Output Contract for Custom Job Containers"** section in:**`docs/modules/mdk_custom_job_usage.md`**
```
