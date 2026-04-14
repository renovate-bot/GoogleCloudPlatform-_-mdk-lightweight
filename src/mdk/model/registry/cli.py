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

"""CLI for the model.registry module."""

import argparse
import logging
import sys
import os
import json
from pydantic import ValidationError

from mdk.model.registry import upload_model
from mdk.model.registry.models import RegistryAppConfig


def configure_logging():
    """Configures the root logger for the application."""
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )


def main():
    """Main function to parse arguments and run the model upload process."""
    configure_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Upload a model to a model registry (e.g., Vertex AI, MLflow)."
    )

    # --- Required Arguments ---
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
    parser.add_argument(
        "--artifact-uri",
        required=True,
        help="GCS URI of the folder containing the model artifact (e.g., 'gs://bucket/model_dir/').",
    )
    parser.add_argument(
        "--provider",
        default="vertex",
        choices=[
            "vertex",
        ],
        help="The model registry provider to use (default: vertex).",
    )
    parser.add_argument(
        "--metrics-file",
        help="Optional path to a JSON file containing model performance metrics.",
    )
    parser.add_argument(
        "--pipeline-job-id",
        help="Optional Vertex AI Pipeline job run ID to associate with the model version.",
    )

    args = parser.parse_args()

    try:
        # --- Prepare arguments for the orchestrator ---
        performance_metrics = None
        if args.metrics_file:
            logger.info(f"Loading performance metrics from {args.metrics_file}...")
            with open(args.metrics_file, "r") as f:
                performance_metrics = json.load(f)

        logger.info(
            f"Starting model upload process with provider: '{args.provider}'..."
        )

        logger.info("Loading and validating configuration files...")
        app_config = RegistryAppConfig.from_yaml_files(
            gcp_config_path=args.gcp_config,
            general_config_path=args.general_config,
        )
        logger.info("Configuration validated successfully.")

        # --- Call the library's main function ---
        uploaded_model_resource_name = upload_model(
            config=app_config,
            artifact_folder_uri=args.artifact_uri,
            registry_provider_name=args.provider,
            performance_metrics_summary=performance_metrics,
            vertex_ai_pipeline_job_run_id=args.pipeline_job_id,
        )

        logger.info("\n--- Model Upload Succeeded ---")
        logger.info(f"Provider: {args.provider}")
        # The returned object type depends on the provider
        if args.provider == "vertex":
            logger.info(f"Model Resource Name: {uploaded_model_resource_name}")
        logger.info("--------------------------------\n")

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e.filename}", exc_info=False)
        sys.exit(1)
    except ValidationError as e:
        logger.error(
            "Configuration validation failed. Please check your YAML files.",
            exc_info=False,
        )
        logger.error(f"Details:\n{e}")
        sys.exit(1)
    except Exception:
        logger.error(
            "An unexpected error occurred during the upload process.", exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
