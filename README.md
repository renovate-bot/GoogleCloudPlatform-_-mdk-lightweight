# MLOps Development Kit (MDK) - Lightweight

This is a lightweight version of the MLOps Development Kit (MDK) designed to streamline machine learning workflows on Google Cloud Platform, particularly with Vertex AI.

This document contains an overview of how to get started using the MDK, including environment setup, infrastructure deployment, and instructions for both standard and "Lite" modes.

---

## 📋 Prerequisites

Before running the Terraform configuration or executing pipelines, ensure you have completed the following setup:

### 1. Authentication & Permissions
*   **Authenticated with Google Cloud**:
    Run the following commands to set up Application Default Credentials (ADC) which Terraform and MDK will use:
    ```bash
    gcloud auth login
    gcloud auth application-default login
    ```

*   **Configured Docker for Artifact Registry**:
    ```bash
    gcloud auth configure-docker ${REGION}-docker.pkg.dev
    ```
    *(Replace `${REGION}` with the region you will use, e.g., `us`)*

*   **Required Permissions**: Your account needs sufficient permissions in the target GCP project.
    *   `roles/serviceusage.serviceUsageAdmin` (Enable APIs)
    *   `roles/resourcemanager.projectIamAdmin` (Manage IAM bindings)
    *   `roles/storage.admin` (Manage GCS)
    *   `roles/artifactregistry.admin` (Manage Artifact Registry)
    *   `roles/bigquery.admin` (Manage BigQuery)

*   **Network**: Depending on your networking setup, you might need to be on a **VPN** for this.

---

## 🛠 Setup

### Local Environment Setup
To use and develop this code, you will need a Python virtual environment. We recommend using `uv`.

1.  **Install `uv`**: Ensure `uv` is installed on your system. (Unlike many Python development tools, it is not recommended to install `uv` in a virtual environment). See [Astral Docs](https://docs.astral.sh/uv/getting-started/installation/) for instructions.

2.  **Create and activate a virtual environment**: In the top-level directory of the source code, run the following:

    ```bash
    # Create a virtual environment:
    uv venv
    uv sync

    # Give your virtual environment access to this source code, in editable mode:
    uv pip install -e .
    ```

3.  **Activate the environment**:
    *   **For Linux/Mac**:
        ```bash
        source .venv/bin/activate
        ```
    *   **For Windows**:
        ```bash
        .venv\Scripts\activate
        ```

---

## 🌍 Terraform Lite Setup

### Deployment Steps
Follow these steps to deploy the infrastructure:

1.  **Navigate to the Terraform directory**:
    ```bash
    cd terraform-lite
    ```

2.  **Configure Variables**:
    Open `lite.tfvars` and update the following variables to match your environment:
    *   `project_id`: The GCP project ID where resources will be deployed.
    *   `user_group_ids`: A list of user emails or Google Group IDs that will access the demo. **Each must be prefixed with `user:` or `group:`** (e.g., `["user:your-name@example.com"]`).

3.  **Deploy**:
    Run the following commands to initialize and apply the configuration:
    ```bash
    terraform init
    terraform apply -var-file=lite.tfvars
    ```

### Troubleshooting: Artifact Registry Permission Issues
If you run `mdk run` and encounter a permission error reading from the Artifact Registry (e.g., Vertex AI cannot pull your Docker image), this may be due to the Vertex AI Custom Container service account not having roles granted yet.

By default, `first_time = true` is set to avoid Terraform failures if the service account is not yet created by Google Cloud.

To resolve a permission issue:
1.  **Update `lite.tfvars`**: Set `first_time` to `false`.
2.  **Apply Terraform again**: `terraform apply -var-file=lite.tfvars`
3.  **Retry**: Run your `mdk run` command again.

---

## 💡 MDK Lite Mode Guide

Lite Mode is designed for **rapid local testing** or **initial prototyping** on Vertex AI with lowered setup overhead.

Its primary function is to **bypass operations targeting the Expanded Model Registry (EMR)** while keeping all Vertex AI Native components (Vertex AI Pipeline submittals, Vertex Model Registry uploads) fully operational.

### Initializing an App Repo
To scaffold a new workspace or application repository inside Lite Mode, first change directories to where you want you new workspace/app repository to be located.
```
cd ..
```

Then run `mdk init` and optionally pass the `--lite` flag during your initial setup initialization:
```bash
# Run initialization (Defaults to Lite Mode)
mdk init
# or 
mdk init --lite
```
To initialize in **Standard Mode** (with Expanded Model Registry support), pass `--lite=False`:
```bash
mdk init --lite=False
```

### Running Your Pipeline
*   **On Vertex AI remote (Standard submission)**: When `lite: true` is set inside your `pipeline_config.yml`, no special flags are needed.
    ```bash
    mdk run xgb_training_pipeline
    ```
*   **On Local Machine Layout (`--local`)**: To execute the pipeline locally inside Lite mode, include **both** flags:
    ```bash
    mdk run --local --lite xgb_training_pipeline
    ```

---

## 🚀 Running an Example

*(Note: The following instructions apply to projects initialized from templates or containing the example structures.)*

To run an example pipeline, follow these steps:

1.  **Upload Data**: Run the data upload script to make the example BigQuery tables available:
    ```bash
    # For Linux/Mac
    cd examples/scripts
    chmod +x load-dry-beans-data-to-bq.sh
    ./load-dry-beans-data-to-bq.sh

    # For Windows
    dir examples/scripts
    bash load-dry-beans-data-to-bq.sh
    ```
2.  **List Pipelines**:
    ```bash
    cd ../..
    mdk list
    ```
3.  **Submit a pipeline for execution**:
    ```bash
    mdk run xgb_training_pipeline
    ```

---

## 🛠 Iterative Development

*(Note: The following instructions apply to projects initialized from templates.)*

In the `src/model_workflow` directory, there are several files that correspond to different steps in an ML pipeline (train, evaluate, batch_predict, etc). These files can be updated with your specific workloads.

Any external variables that are necessary for your workloads (e.g. hyperparameters, GCS file URIs, dataset paths, etc) can be placed in the general config file `model_products/main_product/config/config.yml`.

### Adding Components, Images, or Pipelines
1.  **New Component**: Go into `model_products/main_product/components`, copy an existing component, rename the directory, and update `component.py`.
2.  **New Image**: Go into `model_products/main_product/images`. You can edit files under `byoc` or copy and rename it.
3.  **New Pipeline**: Go into `model_products/main_product/pipelines`, copy an existing pipeline, rename the directory, and update `pipeline.py`.
4.  **Registration**: You must register any new components, images, or pipelines within the pipeline config file `model_products/main_product/config/pipeline_config.yml`.

---

## 👥 Contributors

[Sean Rastatter](mailto:srastatter@google.com)

[Rawan Badawi](mailto:rawanbadawi@google.com)

[Avi Gupta](mailto:guptaavi@google.com)

[Saumya Bhushan](mailto:saumyabhushan@google.com)

[Juan David Vargas-Lopez](mailto:jdvargas@google.com)

[Neil Bushong](mailto:bushong@google.com)

[Vipul Raja](mailto:vipulraja@google.com)

[Arun Kumar](mailto:uarun@google.com)

[Kash Arcot](mailto:arcotk@google.com)

[Jitendra Jaladi](mailto:jjaladi@google.com)

[Jimit Rangras](mailto:jimitrangras@google.com)

---

## 🤝 Contributing
Contributions are welcome! See the [Contributing Guide](CONTRIBUTING.md).

## 📣 Feedback
We value your input! Your feedback helps us improve MDK and make it more useful for the community.

## 🙋 Getting Help

If you encounter any issues or have specific suggestions, please first consider raising an issue on our GitHub repository.

## 📄 Relevant Terms of Service

[Google Cloud Platform TOS](https://cloud.google.com/terms)

[Google Cloud Privacy Notice](https://cloud.google.com/terms/cloud-privacy-notice)

## 📜 Disclaimer
**This is not an officially supported Google product.**

This project is not eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).

Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
