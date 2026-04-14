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
LOCATION = "us-east4"
aiplatform.init(project=PROJECT_ID, location=LOCATION)

endpoint = aiplatform.Endpoint("4116813426951454720")
response = endpoint.predict(
    instances=[
        [
            32057,
            642.082,
            215.9182853,
            189.3033869,
            1.140593884,
            0.4809714697,
            32301.0,
            202.030295,
            0.7798049089,
            0.9924460543,
            0.9771286582,
            0.9356794155,
            0.006735448896,
            0.003184597937,
            0.8754959687,
            0.9985853471,
        ],
        [
            32057,
            642.082,
            215.9182853,
            189.3033869,
            1.140593884,
            0.4809714697,
            32301.0,
            202.030295,
            0.7798049089,
            0.9924460543,
            0.9771286582,
            0.9356794155,
            0.006735448896,
            0.003184597937,
            0.8754959687,
            0.9985853471,
        ],
    ]
)
print(response)
