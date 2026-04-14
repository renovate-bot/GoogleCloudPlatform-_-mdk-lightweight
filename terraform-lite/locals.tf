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

locals {
  apis = [
    "aiplatform.googleapis.com",
    "compute.googleapis.com",
    "storage.googleapis.com",
    "artifactregistry.googleapis.com",
    "iam.googleapis.com",
    "cloudresourcemanager.googleapis.com"
  ]

  user_roles = [
    "roles/serviceusage.serviceUsageConsumer",
    "roles/logging.viewer",
    "roles/storage.objectAdmin",
    "roles/logging.viewAccessor",
    "roles/iam.serviceAccountUser",
    "roles/artifactregistry.writer",
    "roles/artifactregistry.reader",
    "roles/run.invoker",
    "roles/secretmanager.secretAccessor",
    "roles/storage.bucketViewer",
    "roles/datalineage.producer",
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
    "roles/aiplatform.featurestoreDataWriter",
    "roles/aiplatform.user"
  ]

  iam_bindings_map = merge(
    { for user in var.user_group_ids : user => local.user_roles },
    {
      "serviceAccount:${var.pipeline_sa_name}@${var.project_id}.iam.gserviceaccount.com" = [
        "roles/aiplatform.user",
        "roles/serviceusage.serviceUsageConsumer",
        "roles/logging.viewer",
        "roles/storage.objectAdmin",
        "roles/logging.viewAccessor",
        "roles/iam.serviceAccountUser",
        "roles/artifactregistry.writer",
        "roles/run.invoker",
        "roles/secretmanager.secretAccessor",
        "roles/storage.bucketViewer",
        "roles/datalineage.producer",
        "roles/bigquery.dataEditor",
        "roles/bigquery.jobUser",
        "roles/aiplatform.featurestoreDataWriter",
        "roles/bigquery.user",
        "roles/iam.serviceAccountTokenCreator"
      ],

      "serviceAccount:service-${data.google_project.project.number}@gcp-sa-aiplatform.iam.gserviceaccount.com" = [
        "roles/artifactregistry.reader"
      ]
    },
    var.first_time ? {} : {
      "serviceAccount:service-${data.google_project.project.number}@gcp-sa-aiplatform-cc.iam.gserviceaccount.com" = [
        "roles/artifactregistry.reader"
      ]
    }
  )

  # Flatten the map into a list of { member, role } objects for for_each
  project_iam_members = flatten([
    for member, roles in local.iam_bindings_map : [
      for role in roles : {
        member = member
        role   = role
      }
    ]
  ])

  pipeline_staging_bucket_name = "${var.project_id}-data"

  labels = var.labels
}

data "google_project" "project" {
  project_id = var.project_id
}
