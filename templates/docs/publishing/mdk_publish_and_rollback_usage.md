# MDK Declarative Model Publishing Guide
## Introduction

The MDK framework includes a powerful, GitOps-driven workflow for managing the lifecycle of your models after they have been trained and registered. This "declarative publishing" system allows you to promote models to "champion" status, perform rollbacks, and update metadata simply by modifying a central YAML file and merging a pull request.

This approach provides a clear, version-controlled, and auditable trail for all model lifecycle operations, moving these critical steps out of individual notebooks or scripts and into a structured, automated process.
## Core Concepts

The workflow is centered around a single configuration file located in the `publish_and_rollback/` directory of your model product.

*   **One Centralized Configuration File**: You will find a single `publish_and_rollback/operations.yaml`. This file contains separate sections for `train`, `stage`, and `prod`, allowing you to manage all environments from one place.
*   **Declarative Operations**: Instead of running imperative commands (e.g., `mdk deploy --publish ...`), you *declare* the desired state in the appropriate environment section within this YAML file.
*   **GitOps Trigger**: A GitHub Actions workflow monitors changes to this file. When a pull request that modifies `operations.yaml` is merged into a protected branch (`train`, `stage`, or `main`), the workflow is triggered.
*   **Automated Execution**: The GitHub Action parses the file, selects the configuration block corresponding to the target branch (e.g., the `stage:` block for the `stage` branch), and securely executes the requested operations by calling the Expanded Model Registry API and, if necessary, the Vertex AI API.

### What is a "Champion" Model?

In this context, the **champion** model (or "primary" model) is the single, officially designated version of a model that should be used for inference in a given environment. All other versions are considered either **challengers** (potential candidates to become champion) or **archived** (older, inactive versions).

This workflow is primarily concerned with managing which model holds the champion title.
## Key Terminology: Publishing vs. Promotion

Before diving into the operations, it is crucial to understand the distinction between **Model Publishing** and **Model Promotion** within the MDK framework. These terms represent two different, but related, stages of a model's journey to production, both driven by GitOps principles.

In short: **Promotion gets a model *into* a new environment, while Publishing decides which model is *active* within that environment.**
### Model Promotion: Advancing Code to Trigger Retraining

In the MDK framework, **Model Promotion** is the process of advancing your model's **source code and configuration** from a lower-level environment to a higher-level one, which in turn triggers a new training run in that higher environment.

*   **What it is:** Promotion is about validating and merging the *codebase* (e.g., new feature engineering logic, updated model architecture, new configurations in `config.yml`). It is **not** about copying a model artifact.
*   **The Trigger:** A pull request is merged into an environment's main branch (e.g., merging from a feature branch into `stage`, or from `stage` into `main`).
*   **The Action:** A CI/CD workflow (e.g., GitHub Actions) detects this merge and automatically initiates a **full training pipeline** *within the target environment*.
*   **The Outcome:** A **brand new model artifact is created from scratch** in the target environment, using that environment's data and resources. This new model is then registered in that environment's Model Registry as a new candidate version.

In essence, you promote the *recipe* for the model, and the new environment bakes a fresh cake.
### Model Publishing: Activating a Model within an Environment
**Model Publishing** is the act of changing a model's *status* within a single, existing environment to designate it as the official, active version for inference. This happens *after* a model has been trained in or promoted to that environment.

*   **What it is:** This is a metadata update. You are "flipping a switch" in the Expanded Model Registry to designate a specific model version as the **champion**.
*   **The Trigger:** A pull request is merged that modifies the declarative configuration file: `publish_and_rollback/operations.yaml`.
*   **The Action:** A targeted GitHub Action reads the change in the YAML file and calls the Expanded Model Registry API to update the status of the specified model version. If the new champion is for online serving, the workflow also automatically reconfigures the associated Vertex AI Endpoint to direct 100% of traffic to it.
*   **The Outcome:** The targeted model becomes the active "champion," and the previous champion is demoted.
### At a Glance: Promotion vs. Publishing

| Aspect | Model Promotion | Model Publishing |
| :--- | :--- | :--- |
| **Goal** | Advance the model's **code and configuration** to a higher environment to create a new, native model version. | Select the official, **active** model version from the candidates already present within an environment. |
| **Action**| Triggers a full training pipeline, creating a **new model artifact from source**. | Updates metadata flags (`is_primary`, `publish_status`) for an **existing model artifact**. |
| **Scope**| **Cross-environment** (e.g., `train` -> `stage`, `stage` -> `prod`). | **Intra-environment** (e.g., within `stage` or within `prod`). |
| **Trigger** | A PR merge to a main development branch (`train`, `stage`, `main`). | A PR merge modifying the `publish_and_rollback/operations.yaml` file. |
| **Analogy**| A software team merges a feature branch into `main`. The CI/CD system automatically **builds the entire application from source** and deploys the new build to the production servers. | An operations engineer **flips a switch** in a load balancer to direct all user traffic to the newly deployed application version that is already sitting on the servers. |
### Visual Workflow

The end-to-end lifecycle can be visualized as follows:

`Code Change in Dev` -> **[PR to `stage` branch]** -> `[MERGE]` -> **[CI/CD PIPELINE TRIGGERS]** -> `Runs Training Pipeline in 'stage' Env` -> `New Model Created in 'stage' Registry`
*At this point, the new model is just a candidate in the `stage` environment.*

`New PR to 'stage' branch (modifies the 'stage' section in operations.yaml)` -> **[MERGE]** -> **[PUBLISHING WORKFLOW TRIGGERS]** -> `Updates Model Status in 'stage' Registry to 'Champion'` -> `Model is now the active 'Champion' in 'stage' Env`

---
## The Publishing Workflow in Action

Here is the step-by-step process for using the declarative publishing workflow:

1.  **Create a Feature Branch**: Start by creating a new branch from the target environment's branch (e.g., create `feature/promote-new-model` from the `stage` branch).
2.  **Declare the Operation**: Open the central `publish_and_rollback/operations.yaml` file and locate the section for your target environment (e.g., the `stage:` block).
3.  **Uncomment and Configure**: Inside that environment's `operations` list, find the operation you wish to perform (e.g., "Publishing New Champion"), uncomment the block, and configure its `target` and `status` sections.
4.  **Commit and Push**: Commit the changes to the `operations.yaml` file and push your feature branch to the remote repository.
5.  **Create a Pull Request**: Open a pull request to merge your feature branch into the target environment's branch (e.g., from `feature/promote-new-model` into `stage`).
6.  **Review and Approve**: Your team reviews the declarative change. This PR serves as an audit log and an approval gate for a critical model lifecycle change.
7.  **Merge the PR**: Once approved, merge the pull request.
8.  **Automatic Execution**: The GitHub Actions workflow will automatically trigger, read the committed changes in the YAML file, and execute the defined operations against the specified environment.

After the action completes, you can comment out the operation block in a subsequent PR to "reset" the file, ensuring the same operation isn't accidentally re-run on the next change.

---
## Configuration Reference (`operations.yaml`)

The central `operations.yaml` file is organized by environment under a top-level `environments` key. The workflow will automatically select the configuration that matches the target branch of your PR. For a PR to the `main` branch, it will use the `prod` configuration.

```yaml
environments:
  train:
    api_settings:
      # ... train-specific settings
    operations:
      # ... list of operations for train
  stage:
    api_settings:
      # ... stage-specific settings
    operations:
      # ... list of operations for stage
  prod:
    api_settings:
      # ... prod-specific settings
    operations:
      # ... list of operations for prod
```

### `api_settings` (per environment)

This section provides the context for the GitHub Action, telling it which environment and backend services to target for a given operation. **You should not need to edit this section.** It is found within each environment block (e.g., `environments.stage.api_settings`).

| Key | Description |
| :--- | :--- |
| `expanded_model_registry_endpoint` | The URL of the Expanded Model Registry API. |
| `project_id` | The GCP Project ID for the target environment. |
| `region` | The GCP Region for the target environment. |
| `deployment_environment`| The name of the environment (e.g., "stage"), which must match the top-level key for the environment block. |

### `operations` (per environment)

This is a list where you define one or more actions to be executed for a specific environment. Each item in the list is an operation object.

### Targeting a Model

Every operation requires a `target` block to identify which model version the action applies to. The system uses the MDK's `ModelReferenceConfig` to resolve this.

| Key | Description |
| :--- | :--- |
| `model_name` | **Required.** The logical name of the model (e.g., `"xgboost_model"`). |
| `model_inference_reference` | **Required.** The strategy to find the model version. Supported options:<ul><li>`"latest"`: Targets the most recently registered version of `model_name`.</li><li>`"primary"`: Targets the *current* champion model.</li><li>**Semantic Version** (e.g., `"0.1.1"`): Targets the exact model version with this semantic tag.</li><li>**Vertex AI Resource Name & Version**: You can also use `vertex_ai_model_resource_name` and `vertex_ai_model_version_id` for an explicit, non-ambiguous target.</li></ul> |

---
## Available Operations
### 1. `publish_primary` - Promote a New Champion

This is the most common operation. It promotes a target model to "champion" status and demotes the current champion. If the champion model is being used for `online-inference`, this workflow will also automatically update the corresponding Vertex AI Endpoint to direct 100% of traffic to the new champion model.

```yaml
- name: "Publishing New Champion"
  type: "publish_primary"
  target:
    model_name: xgboost_model
    model_inference_reference: "latest" # Promote the newest version
  # Describes the final state of the NEW champion model
  champion_status:
    publish_status: "champion"
    deployment_status:
      status: "active"
      reason: "new champion deployed by GitOps"
      inference_type: "online-inference"
  # Describes the final state of the OLD champion model
  demoted_status:
    publish_status: "archived"
    deployment_status:
      status: "inactive"
      reason: "replaced by new champion"
      inference_type: "online-inference"
```

*   **`type`**: Must be `"publish_primary"`.
*   **`target`**: Identifies the model to be promoted. Using `"latest"` is common for promoting a newly trained and validated model.
*   **`champion_status`**: A dictionary defining the metadata to apply to the newly promoted model. The `deployment_status` sub-dictionary is a flexible JSON object that will be stored in the Expanded Model Registry.
*   **`demoted_status`**: A dictionary defining the metadata to apply to the model that *was* the champion before this operation.

For *deployment_status*, you may specify `inference_type` as either `"batch-inference"` or `"online-inference"`. If you specify `"online-inference"`, the workflow will attempt to update the Vertex AI endpoint associated with the champion to ensure that it receives 100% traffic. This means that the `deployment_endpoint_id` field must be set in the Expanded Model Registry for this champion model record. If you are using the `deployment_pipeline` provided as part of MDK, it will set this value automatically for you.

### 2. `rollback_primary` - Roll Back to a Previous Champion

This operation is used to quickly revert to a previously known-good state. It can be performed automatically or by targeting a specific version.
#### Automatic Rollback

This rolls back to the model that was the champion *before* the current one.

```yaml
- name: "Automatic Rollback"
  type: "rollback_primary"
  target:
    model_name: xgboost_model
    # The 'latest' reference here is a placeholder; the API will find the *previous* primary.
    model_inference_reference: "latest"
  champion_status:
    publish_status: "champion"
    deployment_status:
      status: "active"
      reason: "automatic rollback triggered"
      inference_type: "online-inference"
  demoted_status:
    publish_status: "challenger" # You may want to mark the faulty model as a challenger for investigation
    deployment_status:
      status: "demoted"
      reason: "automatic rollback demoted current primary"
      inference_type: "online-inference"
```
#### Targeted Rollback

This promotes a *specific, older version* back to champion status.

```yaml
- name: "Targeted Rollback"
  type: "rollback_primary"
  target:
    model_name: xgboost_model
    model_inference_reference: "0.1.0" # Roll back to this exact version
  champion_status:
    # ... status for the model being restored
  demoted_status:
    # ... status for the model being replaced
```
### 3. `update_deployment_status` - Update Model Metadata

This operation allows you to change the metadata of a specific model version in the Expanded Model Registry without affecting its champion/challenger status. This is useful for updating tracking information, adding notes, or associating models with challengers.

```yaml
- name: "Update Challenger Status"
  type: "update_deployment_status"
  target:
    model_name: xgboost_model
    model_inference_reference: "0.1.0" # Target a specific version
  # The 'updated_status' block contains the metadata fields to update.
  # Any fields you include will be overwritten; fields you omit will be unchanged.
  updated_status:
    publish_status: "challenger" # e.g., Change from 'archived' to 'challenger'
    deployment_status:
      status: "active_canary"
      traffic_split_percentage: 10
      validation_result: "in_progress"
    challenger_model_ids: # Link this model to other challengers
      - "1db0c39b-4650-450b-8400-25f3d86d839c"
```

*   **`type`**: Must be `"update_deployment_status"`.
*   **`target`**: Identifies the model whose metadata you want to change.
*   **`updated_status`**: A dictionary where you specify the new values for `publish_status`, `deployment_status`, or other trackable fields like `challenger_model_ids`.
