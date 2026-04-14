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

output "project_id" {
  description = "Details about the GCP project."
  value = {
    id     = var.project_id
    number = data.google_project.project.number
  }
}

output "storage_buckets" {
  description = "A map of the GCS buckets created for the project."
  value = {
    for k, bucket in module.gcs_bucket : k => {
      pipeline_data_bucket     = bucket.name
      pipeline_data_bucket_url = bucket.url
    }
  }
}

output "tfstate_bucket" {
  description = "The terraform state bucket."
  value = {
    name = module.tfstate_bucket.name
    url  = module.tfstate_bucket.url
  }
}

output "artifact_registry_repositories" {
  description = "Details of the Artifact Registry repositories."
  value = {
    ml_pipelines_repo = {
      name = module.docker_artifact_registry.name
      id   = module.docker_artifact_registry.id
      url  = module.docker_artifact_registry.url
    }
  }
}






output "service_accounts" {
  description = "Email addresses of the key service accounts created."
  value = {
    pipeline_runner = "${var.pipeline_sa_name}@${var.project_id}.iam.gserviceaccount.com"
  }
}



output "all_managed_buckets" {
  description = "A complete map of all outputs from every instance of the gcs_bucket module, keyed by the bucket name."
  value       = module.gcs_bucket
}
