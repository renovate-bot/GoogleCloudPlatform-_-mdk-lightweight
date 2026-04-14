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

"""CLI for the model.deployment module."""

import argparse
import logging
import sys
import os
from pydantic import ValidationError

from mdk.model.deployment import deploy_model
from mdk.model.deployment.models import DeploymentAppConfig


def configure_logging():
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )


def main():
    configure_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Deploy a model from Vertex AI Model Registry to an Endpoint."
    )
    parser.add_argument(
        "--gcp-config",
        required=True,
        help="Path to the GCP resources YAML config file (state/<env>.yml).",
    )
    parser.add_argument(
        "--general-config",
        required=True,
        help="Path to the general YAML config file (config/<env>.yml).",
    )
    args = parser.parse_args()

    try:
        logger.info("Loading and validating configuration files...")
        app_config = DeploymentAppConfig.from_yaml_files(
            gcp_config_path=args.gcp_config,
            general_config_path=args.general_config,
        )
        logger.info("Configuration validated successfully.")

        endpoint_name = deploy_model(config=app_config)

        logger.info("\n--- Model Deployment Succeeded ---")
        logger.info(f"Endpoint Resource Name: {endpoint_name.resource_name}")
        logger.info("--------------------------------\n")

    except (FileNotFoundError, ValidationError):
        logger.error("Configuration error.", exc_info=True)
        sys.exit(1)
    except Exception:
        logger.error("An unexpected error occurred during deployment.", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
