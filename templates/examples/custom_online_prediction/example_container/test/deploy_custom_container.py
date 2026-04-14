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

from google.cloud import aiplatform


PROJECT_ID = ""
LOCATION = "us"
BUCKET_URI = f"gs://{PROJECT_ID}-data"
MODEL_ARTIFACT_DIR = "custom-container-prediction-model"
REPOSITORY = "ml-pipelines-repo"
IMAGE = "sklearn-fastapi-server"
MODEL_DISPLAY_NAME = "sklearn-custom-container"

aiplatform.init(project=PROJECT_ID)

model = aiplatform.Model.upload(
    display_name=MODEL_DISPLAY_NAME,
    artifact_uri=f"{BUCKET_URI}/{MODEL_ARTIFACT_DIR}",
    serving_container_image_uri=f"{LOCATION}-docker.pkg.dev/{PROJECT_ID}/{REPOSITORY}/{IMAGE}",
)

endpoint = model.deploy(machine_type="n1-standard-4")
