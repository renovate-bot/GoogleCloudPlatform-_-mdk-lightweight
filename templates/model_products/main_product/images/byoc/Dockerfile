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

# Dockerfile
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the Python training script and its dependencies into the container
COPY model_products/main_product/images/byoc/trainer.py /app/trainer.py
COPY model_products/main_product/images/byoc/requirements.txt /app/requirements.txt

# Set our proxy so that we can install Python libraries (e.g. http://proxy.nas.medcity.net:80)
# See: https://glowing-adventure-wojng1m.pages.github.io/getting-started/onboarding/contractor/vdi-setup/
ARG HTTPS_PROXY=
ENV HTTPS_PROXY=${HTTPS_PROXY}

# Install any required Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Unset our proxy so that GCP works.
ENV HTTPS_PROXY=""

# For CustomTrainingJobOp, if 'command' is specified, it OVERRIDES the ENTRYPOINT.
# If 'command' is NOT specified, then ENTRYPOINT is used, and 'args' are appended to it.
# For simplicity and direct control from KFP, we will let KFP's 'command' override this.
ENTRYPOINT ["python", "trainer.py"]

# CMD will be ignored if 'command' is set in KFP.
CMD ["--help"]
