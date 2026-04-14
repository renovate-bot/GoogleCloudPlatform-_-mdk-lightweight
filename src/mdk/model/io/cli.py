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

"""A command-line interface for loading and testing model serialization."""

import argparse
import logging
import sys
import os

# The CLI should use the same public facade as any other client.
# We alias 'save' to avoid a name collision with the local function in this file.
from mdk.model import load as load_model
from mdk.model import save as save_model


def configure_logging():
    """Configures the root logger for the application."""
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )


def handle_load(args):
    """Handler for the 'load' subcommand."""
    logger = logging.getLogger(__name__)
    try:
        logger.info(f"Attempting to load model from: {args.file}")
        model = load_model(filename=args.file)
        logger.info("Read successful.")
        logger.info("\n--- Model Read Succeeded ---")
        logger.info(f"File: {args.file}")
        logger.info(f"Loaded Object Type: {type(model)}")
        logger.info("----------------------------\n")
    except Exception:
        logger.error(f"Failed to load file: {args.file}", exc_info=True)
        sys.exit(1)


def handle_save(args):
    """Handler for the 'save' subcommand."""
    logger = logging.getLogger(__name__)
    framework = args.framework
    dummy_model = None

    try:
        logger.info(f"Creating a dummy '{framework}' model for save command...")
        if framework == "xgboost":
            import xgboost

            dummy_model = xgboost.XGBClassifier(
                n_estimators=2, max_depth=2, use_label_encoder=False
            )
            dummy_model.fit([[0], [1]], [0, 1])  # Fit with minimal data
        elif framework == "pytorch":
            import torch

            dummy_model = torch.nn.Sequential(torch.nn.Linear(10, 1))

        output_filename = save_model(model=dummy_model, filename=args.output)

        logger.info("Write successful.")
        logger.info("\n--- Model Write Succeeded ---")
        logger.info(f"Framework: {framework}")
        logger.info(f"Dummy model written to: {output_filename}")
        logger.info("-----------------------------\n")

    except ImportError:
        logger.error(
            f"Failed to import '{framework}'. Please ensure it is installed.",
            exc_info=False,
        )
        sys.exit(1)
    except Exception:
        logger.error("An unexpected error occurred during save.", exc_info=True)
        sys.exit(1)


def main():
    """Main function to parse arguments and run the specified command."""
    configure_logging()

    parser = argparse.ArgumentParser(
        description="A CLI tool to load and save machine learning models using mdk.model.io."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Read Command ---
    parser_load = subparsers.add_parser(
        "load", help="Read a serialized model from a file and print its type."
    )
    parser_load.add_argument("file", help="The path to the model file.")
    parser_load.set_defaults(func=handle_load)

    # --- Write Command ---
    parser_save = subparsers.add_parser(
        "save", help="Create a dummy model and save it to a file to test serialization."
    )
    parser_save.add_argument(
        "framework",
        choices=["xgboost", "pytorch"],
        help="The framework for the dummy model.",
    )
    parser_save.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional output filename. If not provided, a default will be used.",
    )
    parser_save.set_defaults(func=handle_save)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
