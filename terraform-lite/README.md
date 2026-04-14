# Terraform Lite Infrastructure

This directory contains Terraform configuration to deploy the Google Cloud infrastructure required for the MDK (MLOps Dev Kit) lite.

## Prerequisites & Permissions

Before running the Terraform configuration, ensure you have:

1. **Authenticated with Google Cloud**:
   Run the following command to set up Application Default Credentials (ADC) which Terraform will use:
   ```bash
   gcloud auth application-default login
   ```

2. **Required Permissions**:
   Your account needs sufficient permissions in the target GCP project.
   *   **Granular Roles**:
       *   `roles/serviceusage.serviceUsageAdmin` (Enable APIs)
       *   `roles/resourcemanager.projectIamAdmin` (Manage IAM bindings)
       *   `roles/storage.admin` (Manage GCS)
       *   `roles/artifactregistry.admin` (Manage Artifact Registry)
       *   `roles/bigquery.admin` (Manage BigQuery)

## Deployment Steps

1. **Navigate to this directory**:
   ```bash
   cd terraform-lite
   ```

2. **Configure Variables**:
   Open `lite.tfvars` and update the following variables to match your environment:
   *   `project_id`: The GCP project ID where resources will be deployed.
   *   `user_group_ids`: A list of user emails or Google Group IDs that will access the lite. **Each must be prefixed with `user:` or `group:`** (e.g., `["user:your-name@example.com"]`).

3. **Deploy (Local State)**:
   Run the following commands to initialize and apply the configuration with local state. This will create your base infrastructure, including a GCS bucket to store remote Terraform state:
   ```bash
   terraform init
   terraform apply -var-file=lite.tfvars
   ```

## Migrating to Remote Terraform State (Recommended)

After running your initial local deployment, an empty Terraform state bucket named `<project_id>-tfstate` was created for you. It is highly recommended to migrate your local state to this bucket to allow for secure storage, state locking, and collaboration.

1. **Configure the Remote Backend**:
   Create a `backend.tf` file (or uncomment it if you have one) in this directory with the following configuration:
   ```hcl
   terraform {
     backend "gcs" {
       bucket = "<your-project-id>-tfstate"
       prefix = "terraform/state"
     }
   }
   ```
   *Note: Replace `<your-project-id>` with your actual `project_id`.*

2. **Migrate the State**:
   Run the initialization command again to push your local state to the newly configured GCS bucket:
   ```bash
   terraform init
   ```
   When prompted, answer **`yes`** to copy the existing state to the new Google Cloud Storage backend.

## Troubleshooting: Artifact Registry Permission Issues

If you run `mdk run` and encounter a permission error reading from the Artifact Registry, set `first_time = false` in `lite.tfvars` and run `terraform apply` again.

---

<!-- BEGIN_TF_DOCS -->
## Requirements

| Name | Version |
|------|---------|
| <a name="requirement_terraform"></a> [terraform](#requirement\_terraform) | ~>1.0 |
| <a name="requirement_google"></a> [google](#requirement\_google) | >= 7.0.1, < 8.0.0 |
| <a name="requirement_google-beta"></a> [google-beta](#requirement\_google-beta) | >= 7.0.1, < 8.0.0 |

## Providers

| Name | Version |
|------|---------|
| <a name="provider_google"></a> [google](#provider\_google) | 7.23.0 |
| <a name="provider_google-beta"></a> [google-beta](#provider\_google-beta) | 7.23.0 |

## Modules

| Name | Source | Version |
|------|--------|---------|
| <a name="module_bigquery"></a> [bigquery](#module\_bigquery) | git::https://github.com/GoogleCloudPlatform/cloud-foundation-fabric.git//modules/bigquery-dataset | v45.0.0 |
| <a name="module_docker_artifact_registry"></a> [docker\_artifact\_registry](#module\_docker\_artifact\_registry) | git::https://github.com/GoogleCloudPlatform/cloud-foundation-fabric.git//modules/artifact-registry | v45.0.0 |
| <a name="module_gcs_bucket"></a> [gcs\_bucket](#module\_gcs\_bucket) | git::https://github.com/GoogleCloudPlatform/cloud-foundation-fabric.git//modules/gcs | v45.0.0 |

## Resources

| Name | Type |
|------|------|
| [google-beta_google_project_service_identity.aiplatform_sa](https://registry.terraform.io/providers/hashicorp/google-beta/latest/docs/resources/google_project_service_identity) | resource |
| [google_project_iam_member.iam_access](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/project_iam_member) | resource |
| [google_project_service.apis](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/project_service) | resource |
| [google_service_account.pipeline_sa](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/service_account) | resource |
| [google_storage_bucket_object.cloud_resources_yaml_main_bucket](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/storage_bucket_object) | resource |
| [google_project.project](https://registry.terraform.io/providers/hashicorp/google/latest/docs/data-sources/project) | data source |

## Inputs

| Name | Description | Type | Default | Required |
|------|-------------|------|---------|:--------:|
| <a name="input_artifact_registry"></a> [artifact\_registry](#input\_artifact\_registry) | Artifact Registry configuration | <pre>object({<br>    name                  = optional(string, "ml-pipelines-repo")<br>    description           = optional(string, "MLOps pipeline components.")<br>    docker_immutable_tags = optional(bool, false)<br>  })</pre> | `{}` | no |
| <a name="input_bigquery_config"></a> [bigquery\_config](#input\_bigquery\_config) | A single configuration object for defining a BigQuery Dataset, its IAM, and its contained Tables, Views, and Routines. | <pre>list(object({<br>    dataset_id                 = optional(string, null)<br>    delete_contents_on_destroy = optional(bool, false)<br>  }))</pre> | `[]` | no |
| <a name="input_first_time"></a> [first\_time](#input\_first\_time) | Flag to skip granting IAM access to Vertex AI Custom Container service account on the first run, as it might not be fully initialized or available immediately. | `bool` | `true` | no |
| <a name="input_gcs_config"></a> [gcs\_config](#input\_gcs\_config) | Comprehensive configuration settings for the Google Cloud Storage (GCS) bucket and related IAM, lifecycle, and HMAC settings. | <pre>list(object({<br>    bucket_name            = optional(string, null)<br>    bucket_storage_class   = optional(string, "STANDARD")<br>    force_destroy          = optional(bool, false)<br>    soft_delete_retention  = optional(string, "7776000")<br>    enable_data_protection = optional(bool, true)<br>    data_retention         = optional(string, "0")<br>  }))</pre> | `[]` | no |
| <a name="input_labels"></a> [labels](#input\_labels) | A map of labels to apply to all resources. | `map(string)` | <pre>{<br>  "environment": "lite",<br>  "mdk": "true"<br>}</pre> | no |
| <a name="input_location"></a> [location](#input\_location) | The location of the resources. | `string` | `"us"` | no |
| <a name="input_pipeline_sa_name"></a> [pipeline\_sa\_name](#input\_pipeline\_sa\_name) | The name of the pipeline service account. | `string` | `"pipeline-sa"` | no |
| <a name="input_project_id"></a> [project\_id](#input\_project\_id) | The GCP project ID to deploy the MDK lite resources into. | `string` | n/a | yes |
| <a name="input_region"></a> [region](#input\_region) | Region for GCP resources. | `string` | `"us-central1"` | no |
| <a name="input_user_group_ids"></a> [user\_group\_ids](#input\_user\_group\_ids) | A list of Google Group IDs or user emails that will run the mdk lite. Each must be prefixed with group: or user:. | `list(string)` | n/a | yes |

## Outputs

| Name | Description |
|------|-------------|
| <a name="output_all_managed_buckets"></a> [all\_managed\_buckets](#output\_all\_managed\_buckets) | A complete map of all outputs from every instance of the gcs\_bucket module, keyed by the bucket name. |
| <a name="output_artifact_registry_repositories"></a> [artifact\_registry\_repositories](#output\_artifact\_registry\_repositories) | Details of the Artifact Registry repositories. |
| <a name="output_project_id"></a> [project\_id](#output\_project\_id) | Details about the GCP project. |
| <a name="output_service_accounts"></a> [service\_accounts](#output\_service\_accounts) | Email addresses of the key service accounts created. |
| <a name="output_storage_buckets"></a> [storage\_buckets](#output\_storage\_buckets) | A map of the GCS buckets created for the project. |
<!-- END_TF_DOCS -->
