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

"""CLI for the model.monitoring module."""

import argparse
import logging
import sys
import os

from pydantic import ValidationError

from mdk.model.monitoring import set_up_monitoring
from mdk.model.monitoring.models import MonitoringAppConfig


def configure_logging():
    """Configures the root logger for the application."""
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )


def main():
    """Main function to parse arguments and run the monitoring setup."""
    configure_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Set up a Vertex AI Model Monitor and Monitoring Job."
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
        app_config = MonitoringAppConfig.from_yaml_files(
            gcp_config_path=args.gcp_config,
            general_config_path=args.general_config,
            environment=args.environment,
        )
        logger.info("Configuration validated successfully.")

        # Call the library's facade function
        job_url, schedule_url = set_up_monitoring(config=app_config)

        logger.info("\n--- Model Monitoring Setup Succeeded ---")
        logger.info(f"Vertex AI Console URL: {job_url}")
        logger.info("--------------------------------------\n")

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e.filename}", exc_info=True)
        sys.exit(1)
    except ValidationError:
        logger.error(
            "Configuration validation failed. Please check your YAML files.",
            exc_info=True,
        )
        # Pydantic gives detailed errors, which are automatically logged
        sys.exit(1)
    except Exception:
        logger.error(
            "An unexpected error occurred during monitoring setup.", exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
