# Scikit-Learn Example Model (MDK Template)

This directory contains a template for a **Scikit-Learn** model pipeline on Vertex AI. Unlike standard batch pipelines that read directly from CSVs or raw BigQuery tables, this example demonstrates how to integrate with **Vertex AI Feature Store** to manage feature consistency and serving.

## 📋 Prerequisites

Before running the training pipeline (`sklearn_training_pipeline`), you must ensure the Feature Store infrastructure is set up. The source data is typically hosted in the central `shared` project.

### 1. Tools
* **Google Cloud SDK:** Ensure `gcloud` and `bq` CLI tools are installed and authenticated.
* **Python Environment:** You will need `google-cloud-aiplatform` installed to run the setup scripts.

### 2. IAM Permissions
The user or service account running the setup scripts must have:
* **Vertex AI Administrator** (`roles/aiplatform.admin`) to create the Feature Group.
* **BigQuery Data Viewer** (`roles/bigquery.dataViewer`) on the source project (`oshared` or your local project).

---

## 🛠️ Setup Guide

### Step 1: Install Requirements
```bash
pip install google-cloud-aiplatform
```

### Step 2: Verify Source Data (BigQuery)
The Feature Store requires a source BigQuery table.

**Option A: Use Central Data (Recommended)**
Check if the source table exists in the central project:
* **Project:** `<SHARED_PROJECT_ID>-shared`
* **Table:** `ml_data_dev.dry_beans`

If you have access to this table, **skip to Step 3**.

**Option B: Create Data Locally (If missing in Central)**
If the central table does not exist, or if you are working in an isolated sandbox without access to `shared project`, run the provided bash script to create the data in your own project.

1.  Open `scripts/load-dry-beans-data-for-fs.sh` and update the `PROJECT_ID` and `DATA_BUCKET` variables to match your environment.
2.  Run the script:

    ```bash
    chmod +x examples/scripts/load-dry-beans-data-for-fs.sh
    ./examples/scripts/load-dry-beans-data-for-fs.sh
    ```
    > *This creates the `dry_beans` table in your project's `ml_data_dev` dataset.*

### Step 3: Create Feature Store Group
Use the Python script to register the Feature Group in Vertex AI.

1.  Open `examples/scripts/create_feature_store_group.py`.
2.  Update the **Configuration** section at the bottom of the file:
    * `PROJECT_ID`: Set this to **your** GCP project ID (where the Feature Group will be created).
    * `BQ_SOURCE_URI`:
        * **If you used Option A (Central):** Use `bq://<SHARED_PROJECT_ID>.ml_data_dev.dry_beans`.
        * **If you used Option B (Local):** Use `bq://<YOUR_PROJECT_ID>.ml_data_dev.dry_beans`.
3.  Run the script:

    ```bash
    python examples/scripts/create_feature_store_group.py
    ```

### Step 4: Verify Resources
1.  Go to the [Vertex AI Console](https://console.cloud.google.com/vertex-ai).
2.  Navigate to **Feature Store** > **Feature Groups**.
3.  Confirm that `dry_beans_fg` is listed and **Active**.

---

## 🚀 Run the Pipeline
Now that the data and feature infrastructure are ready, run the training pipeline using the MDK CLI:

```bash
mdk run sklearn_training_pipeline
