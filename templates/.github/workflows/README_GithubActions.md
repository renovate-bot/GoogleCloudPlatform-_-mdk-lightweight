# CI/CD Workflows for MLOps

This document provides an overview of the GitHub Actions workflows used in this repository for Continuous Integration (CI) and Continuous Deployment (CD) of MLOps pipelines on Google Cloud Vertex AI.

## Workflows

There are three primary workflows:

1.  **`environment-promotion.yml`**: The main entry-point workflow that orchestrates CI/CD across different environments (`train`, `stage`, `prod`).
2.  **`build_and_deploy_pipelines.yml`**: A reusable workflow responsible for security scanning, building container images, and triggering Vertex AI pipelines.
3.  **`publish_or_rollback_model.yml`**: A workflow for managing model deployments and rollbacks in the Expanded Model Registry based on declarative configuration files.

---

## 1. Environment Promotion Workflow (`environment-promotion.yml`)

This workflow acts as a router, triggering the appropriate build and deployment process based on the Git branch and event type. It ensures that code changes are tested and deployed systematically across environments.

### Triggers

This workflow runs on:

-   **Push**: To `train`, `stage`, and `main` branches. A push to a branch triggers a pipeline run in the corresponding environment.
-   **Pull Request**: To `train`, `stage`, and `prod` branches. A pull request triggers security scans and a local container build (without pushing) to validate changes before merging.
-   **Manual Dispatch (`workflow_dispatch`)**: Allows manually triggering a pipeline run for a specific environment (`train`, `stage`, or `prod`).

### Logic

The workflow contains three jobs, one for each environment:

-   **`train_build_and_push`**:
    -   **Triggered by**: Push/PR to `train` branch, or manual dispatch for `train`.
    -   **Action**: Calls `build_and_deploy_pipelines.yml` for the `train` environment.

-   **`stage_build_and_push`**:
    -   **Triggered by**: Push/PR to `stage` branch, or manual dispatch for `stage`.
    -   **Action**: Calls `build_and_deploy_pipelines.yml` for the `stage` environment.

-   **`prod_build_and_push`**:
    -   **Triggered by**: Push/PR to `main` branch, or manual dispatch for `prod`.
    -   **Action**: Calls `build_and_deploy_pipelines.yml` for the `prod` environment.

---

## 2. Build and Deploy Pipeline Workflow (`build_and_deploy_pipelines.yml`)

This is a reusable workflow that contains the core CI/CD logic.

### On Pull Requests

When triggered by a pull request, this workflow focuses on validation and security:

1.  **Code Scanning**: Runs CodeQL analysis to find security vulnerabilities in the codebase.
2.  **Image Build & Scan**:
    -   Determines which container images are affected by the changes.
    -   Builds the container images locally within the runner.
    -   Scans the locally built images for vulnerabilities using Wiz.io.

This ensures that code and its container dependencies are secure before being merged.

### On Push or Manual Dispatch

When triggered by a push to an environment branch or a manual dispatch, the workflow executes the full deployment process:

1.  **Determine Images**: Identifies the container images that need to be built based on configuration.
2.  **Build & Push Containers**:
    -   Checks if an image with the same content (based on a digest of the source files) already exists in Google Artifact Registry to avoid redundant builds.
    -   If the image doesn't exist, it builds a new container image.
    -   Tags the image with the Git commit SHA for traceability.
    -   Pushes the new image to Google Artifact Registry.
3.  **Compile & Trigger Pipelines**:
    -   For each pipeline defined in `config/ci_cd.yml`:
        -   Compiles the Vertex AI pipeline definition, injecting the newly built container image URIs.
        -   Uploads the compiled pipeline specification to a Google Cloud Storage bucket.
        -   Publishes a message to a Google Cloud Pub/Sub topic. This message contains the GCS path to the pipeline spec, triggering a Cloud Run that starts the Vertex AI Pipeline run.

---

## 3. Model Publish/Rollback Workflow (`publish_or_rollback_model.yml`)

This workflow provides a GitOps-based approach to managing model deployments. It allows users to promote, roll back, or update model metadata by simply changing YAML configuration files in a pull request.

### Triggers

This workflow runs when a pull request targeting the `train`, `stage`, or `main` branch is **merged**, and only if it contains changes to files under `config/update_deployments/`.

### How It Works

1.  **Create a PR**: A user (e.g., a Data Scientist or ML Engineer) creates a new YAML file or modifies an existing one in the `config/update_deployments/` directory. This file declaratively states the desired operation (e.g., promote a model version to be the "primary" serving model).

    *Example `config/update_deployments/promote-dry-beans-v2.yml`:*
    ```yaml
    operation: 'promote_model'
    model_name: 'dry-beans-classifier'
    model_version: '2'
    ```

2.  **Merge the PR**: Once the PR is reviewed and merged, the workflow is triggered.

3.  **Parse Configs**: The workflow finds all changed YAML files within the `config/update_deployments/` directory from the merged PR.

4.  **Execute Operations**: It passes the content of these YAML files to a Python script (`.github/scripts/github_actions_model_publish_ops.py`).

5.  **Update Registry**: The Python script authenticates with Google Cloud and calls the Expanded Model Registry service to perform the actions defined in the configuration files, such as updating a model's status from "candidate" to "promoted" or changing which version is marked as primary.

This process creates an auditable, version-controlled history of all model deployment operations.
