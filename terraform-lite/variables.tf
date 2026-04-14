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

variable "project_id" {
  type        = string
  description = "The GCP project ID to deploy the MDK lite resources into."
}

variable "region" {
  type        = string
  description = "Region for GCP resources."
  default     = "us-central1"
}

variable "first_time" {
  type        = bool
  description = "Flag to skip granting IAM access to Vertex AI Custom Container service account on the first run, as it might not be fully initialized or available immediately."
  default     = true
}


variable "user_group_ids" {
  type        = list(string)
  description = "A list of Google Group IDs or user emails that will run the mdk lite. Each must be prefixed with group: or user:."
}


variable "gcs_config" {
  description = "Comprehensive configuration settings for the Google Cloud Storage (GCS) bucket and related IAM, lifecycle, and HMAC settings."
  type = list(object({
    bucket_name            = optional(string, null)
    bucket_storage_class   = optional(string, "STANDARD")
    force_destroy          = optional(bool, false)
    soft_delete_retention  = optional(string, "7776000")
    enable_data_protection = optional(bool, true)
    data_retention         = optional(string, "0")
  }))

  validation {
    condition = alltrue([
      for bucket in var.gcs_config : (
        try(bucket.bucket_storage_class, "") == "" || contains(["STANDARD", "MULTI_REGIONAL", "REGIONAL", "NEARLINE", "COLDLINE", "ARCHIVE"], bucket.bucket_storage_class)
      )
    ])
    error_message = "Valid storage class values are: STANDARD, MULTI_REGIONAL, REGIONAL, NEARLINE, COLDLINE, ARCHIVE."
  }

  default = []
}

# #---------------------------------------------------------------------------------------------
# # Artifact Registry Variables
# #---------------------------------------------------------------------------------------------

variable "artifact_registry" {
  description = "Artifact Registry configuration"
  type = object({
    name                  = optional(string, "ml-pipelines-repo")
    description           = optional(string, "MLOps pipeline components.")
    docker_immutable_tags = optional(bool, false)
  })
  default = {}
}

variable "pipeline_sa_name" {
  type        = string
  description = "The name of the pipeline service account."
  default     = "pipeline-sa"
}

variable "location" {
  type        = string
  description = "The location of the resources."
  default     = "us"
}

variable "bigquery_config" {
  description = "A single configuration object for defining a BigQuery Dataset, its IAM, and its contained Tables, Views, and Routines."
  type = list(object({
    dataset_id                 = optional(string, null)
    delete_contents_on_destroy = optional(bool, false)
  }))
  default = []
}

variable "labels" {
  description = "A map of labels to apply to all resources."
  type        = map(string)
  default = {
    "environment" = "lite"
    "mdk"         = "true"
  }
}
