# **GitHub Workflows for AI Platform Ops**

This directory contains **GitHub Actions workflows** that automate CI/CD for AI Platform Ops. The workflows handle **code validation, testing, deployment, and release automation** for various services.

---

## **Workflow Overview**

| Workflow File                      | Purpose |
|-------------------------------------|---------|
| [`ci.yml`](#ciyml)                  | Detects file changes and triggers corresponding CI workflows |
| [`code_checks.yml`](#code_checksyml) | Runs linting, static analysis, and security scans |
| [`build-and-push-image.yml`](#build-and-push-imageyml) | Builds and pushes Docker images to Artifactory |
| [`publish-nexus.yml`](#publish-nexusyml) | Publishes Python packages to Nexus |
| [`release.yml`](#releaseyml)        | Manages versioning and release processes |
| [`event_alerting_ci.yml`](#event_alerting_ciyml) | Runs CI checks for Event Alerting |
| [`event_alerting_deploy.yml`](#event_alerting_deployyml) | Deploys Event Alerting to the target environment |
| [`pipeline_executor_ci.yml`](#pipeline_executor_ciyml) | Runs CI checks for Pipeline Executor |
| [`webex_alerting_ci.yml`](#webex_alerting_ciyml) | Runs CI checks for Webex Alerting |
| [`webex_alerting_deploy.yml`](#webex_alerting_deployyml) | Deploys Webex Alerting to the target environment |
| [`logger_ci.yml`](#logger_ciyml) | Runs CI checks for ai logger |

---

## **Workflow Details**

### **`ci.yml`**
- **Purpose:**
  - Detects changes in the repository and triggers **CI workflows** for affected services.
  - Uses [`dorny/paths-filter`](https://github.com/dorny/paths-filter) to determine file changes.
- **Triggers:**
  - `push` and `pull_request` events on `main` and `develop` branches.
- **Key Steps:**
  - Detects changes in `event_alerting/`, `pipeline_executor/`, and `webex_alerting/`.
  - Triggers relevant **CI workflows** based on detected changes.

---

### **`code_checks.yml`**
- **Purpose:**
  - Performs **code linting, security scans, and static type checking**.
- **Triggers:**
  - Called by **CI workflows** using `workflow_call`.
- **Key Steps:**
  - Installs **`uv`** and dependencies.
  - Runs **Ruff** (linting), **MyPy** (type checking), and **Bandit** (security scanning).
  - Runs **Pytest** for unit tests.

---

### **`build-and-push-image.yml`**
- **Purpose:**
  - **Builds and pushes Docker images** to **Artifactory**.
- **Triggers:**
  - Called from **CI workflows** when new changes are detected.
- **Key Steps:**
  - Builds Docker images for **Event Alerting, Pipeline Executor, and Webex Alerting**.
  - Tags and pushes images to **Artifactory**.
  - Supports **multi-stage builds**.

---

### **`publish-nexus.yml`**
- **Purpose:**
  - Publishes **Python packages** to the **Nexus Repository**.
- **Triggers:**
  - Called during the **release pipeline**.
- **Key Steps:**
  - Verifies **package version** before publishing.
  - Uses a **secure authentication method** for uploading packages.
  - Publishes packages to **Nexus**.

---

### **`release.yml`**
- **Purpose:**
  - Manages **versioning** and **releases**.
- **Triggers:**
  - **Manual triggers** via **GitHub Actions UI**.
- **Key Steps:**
  - Uses the **`bump_version`** action to update `version.txt`.
  - Creates a **release branch** and a **pull request**.
  - Merges changes and **publishes a new release**.

---

### **`event_alerting_ci.yml`**
- **Purpose:**
  - Runs **CI checks** for **Event Alerting**.
- **Triggers:**
  - Called by `ci.yml` when changes in `event_alerting/` are detected.
- **Key Steps:**
  - Calls `code_checks.yml` for **linting, security scanning, and testing**.

---

### **`event_alerting_deploy.yml`**
- **Purpose:**
  - Deploys **Event Alerting** to the target environment.
- **Triggers:**
  - Runs **after a successful CI build**.
- **Key Steps:**
  - Pulls the latest **Docker image** from **Artifactory**.
  - Deploys the service to **Kubernetes**.

---

### **`pipeline_executor_ci.yml`**
- **Purpose:**
  - Runs **CI checks** for **Pipeline Executor**.
- **Triggers:**
  - Called by `ci.yml` when changes in `pipeline_executor/` are detected.
- **Key Steps:**
  - Calls `code_checks.yml` for **linting, security scanning, and testing**.

---

### **`webex_alerting_ci.yml`**
- **Purpose:**
  - Runs **CI checks** for **Webex Alerting**.
- **Triggers:**
  - Called by `ci.yml` when changes in `webex_alerting/` are detected.
- **Key Steps:**
  - Calls `code_checks.yml` for **linting, security scanning, and testing**.

---

### **`webex_alerting_deploy.yml`**
- **Purpose:**
  - Deploys **Webex Alerting** to the target environment.
- **Triggers:**
  - Runs **after a successful CI build**.
- **Key Steps:**
  - Pulls the latest **Docker image** from **Artifactory**.
  - Deploys the service to **Kubernetes**.

---

### **`logger_ci.yml`**
- **Purpose:**
  - Runs **CI checks** for **AI Event Logger**.
- **Triggers:**
  - Called by `ci.yml` when changes in `ai_logger/` are detected.
- **Key Steps:**
  - Calls `code_checks.yml` for **linting, security scanning, and testing**.

---

## **Usage & Debugging**

### **Manually Triggering Workflows**
Some workflows, such as `release.yml`, can be triggered manually from GitHub:

1. Navigate to **Actions** in the GitHub repository.
2. Select the desired workflow (e.g., `release.yml`).
3. Click **Run workflow** and specify the required inputs.

---

### **Viewing Workflow Runs**
To check **workflow execution**:
- Go to **GitHub Actions** (`.github/workflows`).
- Click on any workflow run to **view logs and debug failures**.

---

### **Re-running Failed Workflows**
- If a workflow fails, navigate to **GitHub Actions**, find the failed run, and click **Re-run job**.

---

## **Workflow Dependencies**

| Workflow                 | Dependent Workflows |
|--------------------------|--------------------|
| `ci.yml`                 | `event_alerting_ci.yml`, `pipeline_executor_ci.yml`, `webex_alerting_ci.yml` |
| `event_alerting_ci.yml`   | `code_checks.yml` |
| `pipeline_executor_ci.yml` | `code_checks.yml` |
| `webex_alerting_ci.yml`   | `code_checks.yml` |
| `release.yml`            | `publish-nexus.yml` |
| `build-and-push-image.yml` | **Deploy workflows** (Triggered after a successful build) |

---
