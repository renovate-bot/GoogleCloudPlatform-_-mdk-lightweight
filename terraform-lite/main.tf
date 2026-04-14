# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 1. Enable Necessary APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "aiplatform.googleapis.com",
    "compute.googleapis.com",
    "storage.googleapis.com",
    "artifactregistry.googleapis.com",
    "iam.googleapis.com",
    "cloudresourcemanager.googleapis.com"
  ])

  service            = each.key
  disable_on_destroy = false
}

# 2. Create Artifact Registry


module "docker_artifact_registry" {
  source      = "git::https://github.com/GoogleCloudPlatform/cloud-foundation-fabric.git//modules/artifact-registry?ref=v45.0.0"
  project_id  = var.project_id
  location    = try(var.location, "us")
  name        = try(var.artifact_registry["name"], "ml-pipelines-repo")
  description = try(var.artifact_registry["description"], "Docker repository for MLOps pipeline components.")
  labels      = local.labels
  format = {
    docker = {
      standard = {
        immutable_tags = try(var.artifact_registry["docker_immutable_tags"], false)
      }
    }
  }
  depends_on = [
    google_project_service.apis["artifactregistry.googleapis.com"]
  ]
}

resource "google_artifact_registry_repository" "python_artifact_registry" {
  project       = var.project_id
  location      = try(var.location, "us")
  repository_id = "mdk-python-repo"
  description   = "Python repository for MDK packages."
  format        = "PYTHON"
  labels        = local.labels

  depends_on = [
    google_project_service.apis["artifactregistry.googleapis.com"]
  ]
}

# 3. Create GCS Bucket for Staging and Data



module "gcs_bucket" {
  source        = "git::https://github.com/GoogleCloudPlatform/cloud-foundation-fabric.git//modules/gcs?ref=v45.0.0"
  for_each      = { for obj in concat(var.gcs_config, [{ bucket_name = local.pipeline_staging_bucket_name }]) : obj.bucket_name => obj }
  project_id    = var.project_id
  name          = each.value.bucket_name
  location      = try(var.location, "US")
  storage_class = try(each.value.bucket_storage_class, "MULTI_REGIONAL")
  labels        = local.labels

  force_destroy         = try(each.value.force_destroy, true)
  soft_delete_retention = try(each.value.soft_delete_retention, 0)
  versioning            = try(each.value.enable_data_protection, true)

  retention_policy = try(each.value.data_retention, "0") != "0" ? {
    retention_period = each.value.data_retention
    is_locked        = false
  } : null

  depends_on = [
    google_project_service.apis["storage.googleapis.com"]
  ]
}

# 3.5 Create Terraform State Bucket
module "tfstate_bucket" {
  source        = "git::https://github.com/GoogleCloudPlatform/cloud-foundation-fabric.git//modules/gcs?ref=v45.0.0"
  project_id    = var.project_id
  name          = "${var.project_id}-tfstate"
  location      = try(var.location, "US")
  storage_class = "STANDARD"
  labels        = local.labels

  force_destroy         = false
  soft_delete_retention = 0
  versioning            = true

  depends_on = [
    google_project_service.apis["storage.googleapis.com"]
  ]
}

# 4. Create Pipeline Service Account
resource "google_service_account" "pipeline_sa" {
  account_id   = var.pipeline_sa_name
  display_name = "Vertex AI Pipeline Service Account for MDK Lite"

  depends_on = [google_project_service.apis["iam.googleapis.com"]]
}

# 5. Grant Roles to Service Account
resource "google_project_service_identity" "aiplatform_sa" {
  provider = google-beta
  project  = var.project_id
  service  = "aiplatform.googleapis.com"
  timeouts {
    create = "30m"
  }
}



resource "google_project_iam_member" "iam_access" {
  for_each = {
    for binding in local.project_iam_members :
    "${binding.member}-${binding.role}" => binding

  }

  project = var.project_id
  role    = each.value.role
  member  = each.value.member

  depends_on = [
    google_project_service_identity.aiplatform_sa,
    google_service_account.pipeline_sa,
    google_project_service.apis["iam.googleapis.com"]
  ]
}


# 6. Create Vertex AI Datasets for Training and Inference

module "bigquery" {
  source     = "git::https://github.com/GoogleCloudPlatform/cloud-foundation-fabric.git//modules/bigquery-dataset?ref=v45.0.0"
  for_each   = { for obj in var.bigquery_config : obj.dataset_id => obj }
  project_id = var.project_id
  id         = each.value.dataset_id
  location   = try(var.location, "us")
  labels     = local.labels

  options = {
    delete_contents_on_destroy = try(each.value.delete_contents_on_destroy, false)
  }
}


# gcs_config.tf
#
# This file compiles output variables and resource identifiers into a single
# map (`all_outputs`) which is then encoded as YAML and uploaded to a GCS bucket
# as `cloud-infrastructure/gcp/cloud-resources.yaml`.
#
# The MLOps Development Kit (MDK) reads this `cloud-resources.yaml` file to
# discover the endpoints, buckets, and service accounts it needs to interact
# with during Vertex AI Pipeline runs or local executions.

locals {
  all_outputs = {
    gcp_project = {
      id     = var.project_id
      number = data.google_project.project.number
    }
    gcp_region = var.location

    storage_buckets = merge(
      {
        for k, v in module.gcs_bucket : k => {
          name   = v.name
          labels = local.labels
        }
      },
      {
        tfstate = {
          name   = module.tfstate_bucket.name
          labels = local.labels
        }
      }
    )

    artifact_registry_repositories = {
      ml_pipelines_repo = {
        name   = module.docker_artifact_registry.name
        url    = module.docker_artifact_registry.url
        id     = module.docker_artifact_registry.id
        labels = local.labels
      }
      python_repo = {
        name   = google_artifact_registry_repository.python_artifact_registry.repository_id
        url    = "https://${try(var.location, "us")}-python.pkg.dev/${var.project_id}/${google_artifact_registry_repository.python_artifact_registry.repository_id}"
        id     = google_artifact_registry_repository.python_artifact_registry.id
        labels = local.labels
      }
    }

    bigquery = {
      bigquery_dataset = { for k, v in module.bigquery : k => {
        dataset_id = v.dataset_id
        labels     = local.labels
      } }
    }

    service_accounts = {
      # Hardcode format to match actual created pipeline sa or read dynamically if module output allows
      pipeline_runner = "${var.pipeline_sa_name}@${var.project_id}.iam.gserviceaccount.com"
    }

    # Global labels output
    tags = local.labels
  }

  warning_message = <<-EOT
#
# WARNING!
#
# This file is automatically generated and managed by Terraform.
# Please do not modify this file manually. Any changes will be overwritten.
# If this file is deleted, rerun Terraform to recreate it.
#
EOT
}

resource "google_storage_bucket_object" "cloud_resources_yaml_main_bucket" {
  name         = "cloud-infrastructure/gcp/cloud-resources.yaml"
  bucket       = local.pipeline_staging_bucket_name
  content      = format("%s\n%s", local.warning_message, yamlencode(local.all_outputs))
  content_type = "application/x-yaml"

  depends_on = [
    module.gcs_bucket,
    module.docker_artifact_registry,
    google_artifact_registry_repository.python_artifact_registry,
    module.bigquery,
    google_service_account.pipeline_sa,
    google_project_iam_member.iam_access,
    module.tfstate_bucket
  ]
}
