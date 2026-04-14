#!/usr/bin/env python

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

"""
MDK Configuration Utility (config_utils.py)

This file provides helper functions for reading and parsing the MDK's
decentralized configuration files ('config.yml', 'pipeline_config.yml',
and 'state/*.yml').

It can be imported as a Python module or run as a standalone CLI script
for use in CI/CD workflows. (Requires 'pyyaml')

CLI Usage:
----------
# Get a JSON list of all pipelines to be run by CI
$ python -m mdk.pipeline_tools.ci_cd_utils get-ci-pipelines

# Get the source code path for a specific pipeline
$ python -m mdk.pipeline_tools.ci_cd_utils get-pipeline-path xgb_training_pipeline

# Get a JSON list of images to build for a pipeline
$ python -m mdk.pipeline_tools.ci_cd_utils get-images-to-build xgb_training_pipeline --ar-repo "..."

# Query a value from a state/gcp config file
$ python -m mdk.pipeline_tools.ci_cd_utils query-gcp-config xgb_training_pipeline dev project_id

# Get the cron schedule for a training pipeline
$ python -m mdk.pipeline_tools.ci_cd_utils get-cron-schedule xgb_training_pipeline dev

# Find all config.yml files and inject Git metadata
$ python -m mdk.pipeline_tools.ci_cd_utils update-git-metadata \
    --git-repo-url "..." \
    --git-commit-hash "..." \
    --git-branch "..." \
    --trained-by "..."
"""

import argparse
import json
import logging
import pathlib
import sys
import yaml
import mdk.util.framework
import mdk.config
from typing import List

# --- Setup Logger ---
logger = logging.getLogger(__name__)

# --- Constants ---
MODEL_CONFIG_BASENAME = "config.yml"
PIPELINE_CONFIG_BASENAME = "pipeline_config.yml"
MODEL_PRODUCTS_DIRS = ("model_products", "examples/model_products")


# --- Core Library Functions ---


def get_all_ci_cd_pipelines(environment: str) -> List[str]:
    """
    Scans all model products and reads their main 'config.yml' to find
    pipelines listed in the 'ci_cd.pipeline_names' section.

    This is used by the CI/CD system to determine which pipelines to trigger.

    Returns:
        List[str]: A flat list of all unique pipeline names found
                   across all model products (e.g., ['xgb_training_pipeline']).
    """
    if not pathlib.Path("model_products").is_dir():
        raise RuntimeError(
            "This process must be run from the repository root"
            " (where 'model_products/' is a subdirectory)."
        )

    all_pipelines_list = []

    for products_dir_str in MODEL_PRODUCTS_DIRS:
        products_dir = pathlib.Path(products_dir_str)

        if not products_dir.is_dir():
            continue

        for model_product_root in products_dir.iterdir():
            if not model_product_root.is_dir():
                continue

            general_config_filename = (
                model_product_root / "config" / MODEL_CONFIG_BASENAME
            )

            if not general_config_filename.is_file():
                continue

            general_config = mdk.config.readAndMergeYAMLConfig(
                config_filename=general_config_filename, environment=environment
            )

            ci_cd_config = general_config.get("ci_cd", {})
            if ci_cd_config is None:
                ci_cd_config = {}

            pipeline_names = ci_cd_config.get("pipeline_names", [])
            if pipeline_names is None:
                pipeline_names = []

            if pipeline_names:
                all_pipelines_list.extend(pipeline_names)

    # Return a unique, sorted list
    return list(sorted(set(all_pipelines_list)))


def get_pipeline_path(pipeline_name: str) -> pathlib.Path:
    """
    Fetches the source directory path for a given pipeline.

    Args:
        pipeline_name: The unique name of the pipeline.

    Returns:
        pathlib.Path: The path to the pipeline's source directory.
    """
    master_map = mdk.util.framework.get_pipeline_mapping()
    if pipeline_name not in master_map:
        raise KeyError(
            f"Pipeline '{pipeline_name}' not found in any "
            f"{PIPELINE_CONFIG_BASENAME} file."
        )
    return master_map[pipeline_name]


def load_pipeline_cron_schedule(pipeline_name: str, environment: str) -> pathlib.Path:
    """
    Fetches the cron_schedule value from the model product's config.yml
    based on the pipeline type (training or inference).

    Args:
        pipeline_name: The unique name of the pipeline.
        environment (str): The environment used for config merging.

    Returns:
        str | None: The cron schedule string, or None if not found.
    """
    master_map = mdk.util.framework.get_pipeline_mapping()
    if pipeline_name not in master_map:
        raise KeyError(
            f"Pipeline '{pipeline_name}' not found in any "
            f"{PIPELINE_CONFIG_BASENAME} file."
        )
    pipeline_path = master_map[pipeline_name]
    config_path = pipeline_path.parent.parent / "config" / "config.yml"
    general_config = mdk.config.readAndMergeYAMLConfig(
        config_filename=config_path, environment=environment
    )
    cron_schedule = None
    if "training" in pipeline_name:
        cron_schedule = general_config["training"].get("cron_schedule")
    elif "inference" in pipeline_name:
        cron_schedule = general_config["inference"].get("cron_schedule")

    return cron_schedule


def getImagesToBuild(
    ar_repo: str,
    pipeline_dir: pathlib.Path,
    image_name_to_build: str | None = None,
) -> list[str]:
    """
    Parses the pipeline configuration and returns a list of full image URLs
    that are configured to be built.

    Args:
        ar_repo (str): Artifact Registry repository URL.
        pipeline_dir (pathlib.Path): The path to the directory containing
            pipeline.py, config files, etc. for the pipeline of interest.

        image_name_to_build (str | None): A specific image to list. If None,
            all images are listed.

    Returns:
        list[str]: A list of full image URLs.
    """
    pipeline_config_filename = (
        pipeline_dir.parent.parent
        / "config"
        / PIPELINE_CONFIG_BASENAME  # Using local constant for consistency
    )
    pipeline_config = mdk.config.readYAMLConfig(pipeline_config_filename)
    images_to_build = pipeline_config["images"]
    if image_name_to_build:
        if image_name_to_build not in images_to_build:
            err_msg = f"Image '{image_name_to_build}' not found in pipeline_config.yml"
            logger.error(err_msg)
            raise ValueError(err_msg)
        images_to_build = {image_name_to_build: images_to_build[image_name_to_build]}

    build_instructions = []
    for image_config in images_to_build.values():
        # Check for the keys we need
        if "artifact" not in image_config or "build_config_dir" not in image_config:
            logger.warning(
                f"Skipping image config, missing 'artifact' or 'build_config_dir': {image_config}"
            )
            continue

        image_url = f"{ar_repo}/{image_config['artifact'].split(':')[0]}"

        build_instructions.append(
            {
                "image_url": image_url,
                "build_config_dir": image_config["build_config_dir"],
            }
        )

    return build_instructions


def query_gcp_config(
    pipeline_name: str,
    environment: str,
    key: str,
) -> str:
    """Infer the correct GCP config file, and query it for the appropriate
    key / value pair.

    Args:
        pipeline_name (str): The pipeline whose model product we should use for
            the GCP config file
        environment (str): The environment we should use for querying the GCP
            config file.  Must be one of: {"dev", "stage", "prod"}
        key (str): The key to query in the GCP config file

    Returns:
        str: The value in the GCP config file corresponding to the key.
    """
    pipeline_mapping = mdk.util.framework.get_pipeline_mapping()
    if pipeline_name not in pipeline_mapping:
        raise RuntimeError(
            f"Unrecognized pipeline name: {pipeline_name}  Available pipeline"
            f" names: {pipeline_mapping.keys()}"
        )
    pipeline_dir = pipeline_mapping[pipeline_name]
    state_dir = pipeline_dir.parent.parent / "state"
    gcp_config_filename = state_dir / f"{environment}.yml"

    if not gcp_config_filename.is_file():
        raise FileNotFoundError(f"Config file not found at: {gcp_config_filename}")

    gcp_config = mdk.config.GCPConfig.from_yaml_file(str(gcp_config_filename))

    d = dict(gcp_config)
    if key not in d:
        raise ValueError(
            f"key '{key}' not found in {gcp_config_filename}. "
            f"Possible keys: {list(d.keys())}"
        )
    return d[key]


def _update_config(file_path: pathlib.Path, args: argparse.Namespace):
    """Safely reads, updates, and writes a single config.yml file."""
    try:
        logger.info(f"Processing: {file_path}")

        # Read the config
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)

        # Ensure model_registry section exists
        if config.get("model_registry") is None:
            config["model_registry"] = {}

        # Update/add Git metadata
        config["model_registry"]["git_repo_url"] = args.git_repo_url
        config["model_registry"]["git_commit_hash"] = args.git_commit_hash
        config["model_registry"]["git_branch"] = args.git_branch
        config["model_registry"]["trained_by"] = args.trained_by

        # Write the config back, preserving key order and indentation
        with open(file_path, "w") as f:
            yaml.dump(config, f, indent=2, sort_keys=False)

        logger.info(f"Successfully updated {file_path}")

    except Exception as e:
        logger.error(f"Failed to update {file_path}: {e}")
        sys.exit(1)


def find_and_update_configs(args: argparse.Namespace):
    """Finds all config.yml files and updates them."""

    found_files = False
    for products_dir_str in MODEL_PRODUCTS_DIRS:
        products_dir = pathlib.Path(products_dir_str)
        if not products_dir.is_dir():
            continue

        # Iterate over each product directory
        for model_product_root in products_dir.iterdir():
            if not model_product_root.is_dir():
                continue

            # Construct the path to the config file
            config_file = model_product_root / "config" / MODEL_CONFIG_BASENAME

            if config_file.is_file():
                found_files = True
                _update_config(config_file, args)

    if not found_files:
        logger.warning(
            f"No '{MODEL_CONFIG_BASENAME}' files found in {MODEL_PRODUCTS_DIRS}"
        )


# --- CLI (Command-Line Interface) Section ---


def _cli_get_ci_pipelines(args: argparse.Namespace):
    """Handler for the 'get-ci-pipelines' command."""
    try:
        pipelines_list = get_all_ci_cd_pipelines(args.environment)
        # Print as JSON list for easy parsing by CI scripts
        print(json.dumps(pipelines_list, indent=2))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _cli_get_pipeline_path(args: argparse.Namespace):
    """Handler for the 'get-pipeline-path' command."""
    try:
        path = get_pipeline_path(args.pipeline_name)
        # Print as plain text for easy use in shell scripts
        print(path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _cli_get_images_to_build(args: argparse.Namespace):
    """Handler for the 'get-images-to-build' command."""
    try:
        pipeline_dir = get_pipeline_path(args.pipeline_name)

        # This now returns a list of dictionaries
        images_data = getImagesToBuild(
            ar_repo=args.ar_repo,
            pipeline_dir=pipeline_dir,
            image_name_to_build=args.image_name,
        )
        # Print as JSON list of objects
        print(json.dumps(images_data, indent=2))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _cli_query_gcp_config(args: argparse.Namespace):
    """Handler for the 'query-gcp-config' command."""
    try:
        value = query_gcp_config(
            pipeline_name=args.pipeline_name, environment=args.environment, key=args.key
        )
        # Print as plain text for easy parsing by shell scripts
        print(value)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _cli_update_git_metadata(args: argparse.Namespace):
    """Handler for the 'update-git-metadata' command."""
    try:
        find_and_update_configs(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _cli_get_cron_schedule(args: argparse.Namespace):
    """Handler for the 'get-cron-schedule' command."""
    try:
        schedule = load_pipeline_cron_schedule(
            pipeline_name=args.pipeline_name, environment=args.environment
        )
        if schedule:
            # Print as plain text for easy parsing by shell scripts
            print(schedule)
        else:
            # Print nothing and exit successfully if no schedule is found
            sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    # Setup basic logging for all CLI commands
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="MDK Config Utility Script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(title="Commands", dest="command", required=True)

    # Sub-command: get-ci-pipelines
    ci_parser = subparsers.add_parser(
        "get-ci-pipelines",
        help="Get a JSON list of all pipelines listed in ci_cd sections.",
    )
    ci_parser.add_argument(
        "environment",
        choices=("dev", "stage", "prod"),
        type=str,
        help="The environment file to query (e.g., 'dev' -> 'state/dev.yml').",
    )
    ci_parser.set_defaults(func=_cli_get_ci_pipelines)

    # Sub-command: get-pipeline-path
    path_parser = subparsers.add_parser(
        "get-pipeline-path",
        help="Get the source directory path for a specific pipeline.",
    )
    path_parser.add_argument(
        "pipeline_name",
        type=str,
        help="The unique name of the pipeline (e.g., 'xgb_training_pipeline').",
    )
    path_parser.set_defaults(func=_cli_get_pipeline_path)

    # Sub-command: get-images-to-build
    images_parser = subparsers.add_parser(
        "get-images-to-build",
        help="Get a JSON list of container images to build for a pipeline.",
    )
    images_parser.add_argument(
        "pipeline_name",
        type=str,
        help="The unique name of the pipeline (e.g., 'xgb_training_pipeline').",
    )
    images_parser.add_argument(
        "--ar-repo",
        type=str,
        required=True,
        help="Artifact Registry repository URL (e.g., 'us-central1-docker.pkg.dev/my-project/my-repo').",
    )
    images_parser.add_argument(
        "--image-name",
        type=str,
        default=None,
        help="Optional: A specific image name to build (if omitted, builds all).",
    )
    images_parser.set_defaults(func=_cli_get_images_to_build)

    # Sub-command: query-gcp-config
    query_parser = subparsers.add_parser(
        "query-gcp-config",
        help="Query a value from an environment-specific state file (e.g., state/train.yml).",
    )
    query_parser.add_argument(
        "pipeline_name",
        type=str,
        help="The pipeline name used to locate the model product's 'state' directory.",
    )
    query_parser.add_argument(
        "environment",
        choices=("dev", "stage", "prod"),
        type=str,
        help="The environment file to query (e.g., 'dev' -> 'state/dev.yml').",
    )
    query_parser.add_argument(
        "key", type=str, help="The key to query from the YAML file."
    )
    query_parser.set_defaults(func=_cli_query_gcp_config)

    # Sub-command: update-git-metadata
    update_parser = subparsers.add_parser(
        "update-git-metadata",
        help="Finds all config.yml files and injects Git metadata into the model_registry section.",
    )
    update_parser.add_argument(
        "--git-repo-url",
        type=str,
        required=True,
        help="Full URL of the Git repository.",
    )
    update_parser.add_argument(
        "--git-commit-hash", type=str, required=True, help="Git commit SHA."
    )
    update_parser.add_argument(
        "--git-branch", type=str, required=True, help="Git branch name."
    )
    update_parser.add_argument(
        "--trained-by",
        type=str,
        required=True,
        help="Email of the user who triggered the build.",
    )
    update_parser.set_defaults(func=_cli_update_git_metadata)

    cron_parser = subparsers.add_parser(
        "get-cron-schedule",
        help="Get the cron schedule string for a pipeline from its config.",
    )
    cron_parser.add_argument(
        "pipeline_name",
        type=str,
        help="The unique name of the pipeline (e.g., 'xgb_training_pipeline').",
    )
    cron_parser.add_argument(
        "environment",
        choices=("dev", "stage", "prod"),
        type=str,
        help="The environment for config merging (e.g., 'dev').",
    )
    cron_parser.set_defaults(func=_cli_get_cron_schedule)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
