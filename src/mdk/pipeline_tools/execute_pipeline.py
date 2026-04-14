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

"""Run a pipeline, by either A) Submitting the pipeline to be executed on Vertex
AI Pipelines, or B) Running the pipeline locally.
"""

import mdk.config
import google.cloud.aiplatform
import google.api_core.exceptions
import kfp
import kfp.local
import argparse
import logging
import os
import pathlib
import shutil
import sys
from docker.types import Mount
import subprocess

logger = logging.getLogger(__name__)

PIPELINE_CONFIG_BASENAME = "pipeline_config.yml"
GENERAL_CONFIG_BASENAME = "config.yml"


def main():
    """This function is executed when the module is called from the command
    line.
    """
    logging.basicConfig(format="", level=logging.INFO)

    # Read our command line arguments.
    args = _parseCommandLine(sys.argv)

    executePipeline(args.pipeline_dir, args.environment, args.local)


def executePipeline(
    pipeline_dir: pathlib.Path,
    environment: str,
    is_local: bool = False,
    is_lite: bool = False,
):
    """Run a pipeline, either remotely or locally.

    Args:
        pipeline_dir (pathlib.Path): The path to the directory containing
            pipeline.py for the pipeline of interest.
        environment (str): The environment in which the pipeline will be run
            (typically dev, stage or prod)
        is_local (bool): Whether the pipeline will be run locally (True) or
            on Vertex AI Pipelines (False)  (Default: False)
    """
    # Run the pipeline.
    logger.info(f"{'Running' if is_local else 'Submitting'} pipeline: {pipeline_dir}")
    compiledPipelineFilename = _getCompiledPipelineFilename(pipeline_dir)

    if is_local:
        runPipelineLocally(compiledPipelineFilename, environment, is_lite)
    else:
        # Create a display name, and Build our labels.
        display_name = f"{pipeline_dir.stem}-{os.urandom(4).hex().upper()}"
        labels = _get_labels(pipeline_dir, environment)

        # Submit our pipeline.
        submitPipeline(
            pipeline_dir,
            compiledPipelineFilename,
            environment,
            display_name,
            labels,
        )


def runPipelineLocally(
    compiledPipelineFilename: pathlib.Path | str,
    environment: str,
    is_lite: bool = False,
):
    """Run a Kubeflow pipeline locally.

    Args:
        compiledPipelineFilename (str): The compiled spec file for the Kubeflow
            pipeline that is to be run locally.
        environment (str): Suffix to use on the end of the GCP config.yaml
            filename, such that the GCP config file is named
            config/FLAG/gcp_config.yaml, where FLAG is usually one of: (dev,
            stage, prod)
    """
    # Copy our application default credentials to a location that the container
    #   will be able to see.  (The local_outputs/ directory will get mounted
    #   inside the container.  These credentials do *not* get copied inside the
    #   image.)

    cred_dir_host = pathlib.Path("local_outputs")
    cred_dir_host.mkdir(exist_ok=True)
    cred_file_name = "application_default_credentials.json"
    cred_file_host_path = cred_dir_host / cred_file_name

    # Ensure the source file exists before attempting to copy
    source_cred_path = pathlib.Path(
        os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
    )
    if not source_cred_path.exists():
        logger.error(f"Source credentials file not found: {source_cred_path}")
        logger.error(
            "Please ensure you have run 'gcloud auth application-default login' "
            "or that the file exists at this location."
        )
        raise FileNotFoundError(
            f"Source credentials file not found: {source_cred_path}"
        )

    shutil.copy(
        source_cred_path,
        cred_file_host_path,
    )

    # Define the path where the credentials will be accessible INSIDE the container.
    #   This should be a consistent, known path, e.g., in /etc/gcp_secrets/
    cred_file_container_path = pathlib.Path("/etc/gcp_secrets") / cred_file_name

    # Generate ID token for interacting with the Expanded Model Registry
    try:
        result = subprocess.run(
            ["gcloud", "auth", "print-identity-token"],
            capture_output=True,
            text=True,
            check=True,
        )
        id_token = result.stdout.strip()
        logger.info("Successfully generated ID token.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"Failed to generate ID token using 'gcloud'. Error: {e}")
        logger.error(
            "Please ensure 'gcloud' is installed, authenticated, and in your PATH."
        )
        if isinstance(e, subprocess.CalledProcessError):
            logger.error(f"gcloud stderr: {e.stderr}")
        raise

    # Create a docker.types.Mount object for the credentials file.
    credentials_mount = Mount(
        target=str(cred_file_container_path),  # Path INSIDE the container
        source=str(cred_file_host_path.absolute()),  # ABSOLUTE path on the HOST
        type="bind",  # This is a bind mount
        read_only=True,  # Recommended for security
    )

    local_env = {
        "GOOGLE_APPLICATION_CREDENTIALS": str(cred_file_container_path),
        "ID_TOKEN": id_token,
    }
    if is_lite:
        local_env["MDK_LITE_MODE"] = "True"

    # Configure the DockerRunner with volume mounts and environment variables.
    # Configure the DockerRunner with volume mounts and environment variables.
    # See here for more details:
    # https://github.com/kubeflow/pipelines/blob/master/sdk/python/kfp/local/config.py
    # (Uses the docker pkg under the hood: https://docker-py.readthedocs.io/en/stable/containers.html)
    docker_runner_config = kfp.local.DockerRunner(
        mounts=[credentials_mount],
        environment=local_env,
    )

    # Create our runner and load our pipeline.
    kfp.local.init(runner=docker_runner_config)
    pipeline = kfp.components.load_component_from_file(str(compiledPipelineFilename))

    # Prepare parameter values for local run
    parameter_values = {
        "environment": environment,
    }

    # Run our pipeline.
    logger.info("Running local pipeline...")
    pipeline_task = pipeline(**parameter_values)
    outputs = pipeline_task.outputs
    logger.info(f"{outputs = }")
    logger.info("Task complete.")


def submitPipeline(
    pipeline_dir: pathlib.Path,
    compiledPipelineFilename: pathlib.Path,
    environment: str,
    display_name: str,
    labels: dict[str, str],
):
    """Submit our Kubeflow pipeline to be executed on Vertex AI Pipelines.

    Args:
        compiledPipelineFilename (pathlib.Path): The compiled spec file for the
            Kubeflow pipeline that is to be submitted to Vertex AI Pipelines.
        environment (str): Suffix to use on the end of the GCP config.yaml
            filename, such that the GCP config file is named
            config/FLAG/gcp_config.yaml, where FLAG is usually one of: (dev,
            stage, prod)
        display_name (str): The name of the pipeline to be displayed in Vertex
            AI Pipelines, corresponding to the display_name argument in the
            PipelineJob constructor.
        labels (dict[str, str]): Labels to attach to the pipeline when it is
            submitted.
    """
    logger.info("Initializing...")

    state_dir = pipeline_dir.parent.parent / "state"
    gcp_config_filename = state_dir / f"{environment}.yml"
    gcp_config = mdk.config.GCPConfig.from_yaml_file(str(gcp_config_filename))

    config_dir = pipeline_dir.parent.parent / "config"
    general_config_filename = config_dir / GENERAL_CONFIG_BASENAME
    general_config = mdk.config.readAndMergeYAMLConfig(
        config_filename=general_config_filename,
        environment=environment,
    )
    training_config = general_config["training"]
    inference_config = general_config["inference"]

    gcs_pipeline_staging_dir = gcp_config.pipeline_staging_dir
    if not gcs_pipeline_staging_dir.startswith("gs://"):
        gcs_pipeline_staging_dir = "gs://" + gcs_pipeline_staging_dir

    google.cloud.aiplatform.init(
        project=gcp_config.project_id,
        location=gcp_config.region,
        staging_bucket=gcs_pipeline_staging_dir,
        service_account=gcp_config.pipeline_service_account,
    )

    experiment_name = gcp_config.experiment_name
    # If we lack the permissions to create an experiment, we will proceed
    #   anyway, without an experiment.  For example, our 3/4 ID might have
    #   permissions to create an experiment in train, but not in stage.
    logger.info(f"Creating experiment: {experiment_name}")
    try:
        experiment = google.cloud.aiplatform.Experiment.get_or_create(
            experiment_name=experiment_name
        )
    except google.api_core.exceptions.PermissionDenied:
        logger.warning("WARNING: Error creating experiment:\n{e}")
        experiment = None

    parameter_values = {
        "environment": environment,
    }

    logger.info(f"Submitting pipeline job: {display_name}")
    job = google.cloud.aiplatform.PipelineJob(
        display_name=display_name,
        template_path=str(compiledPipelineFilename),
        pipeline_root=gcs_pipeline_staging_dir,
        parameter_values=parameter_values,
        enable_caching=False,
        labels=labels,
    )

    # Create schedule if specified
    # First, get the type of pipeline that is being run
    cron_schedule = None
    if "training" in display_name:
        cron_schedule = training_config.get("cron_schedule")
    elif "inference" in display_name:
        cron_schedule = inference_config.get("cron_schedule")

    if cron_schedule:
        job.create_schedule(
            display_name=f"{display_name}-schedule",
            cron=cron_schedule,
        )

    job.submit(experiment=experiment)

    logger.info("Submitted.")


def _getCompiledPipelineFilename(pipeline_dir: pathlib.Path) -> pathlib.Path:
    """Infer the name of the corresponding compiled spec file.

    Args:
        pipeline_dir (pathlib.Path): The path to the directory containing
            pipeline.py, config files, etc. for the pipeline of interest.
    """
    pipeline_name = pipeline_dir.stem
    pipeline_config_filename = (
        pipeline_dir.parent.parent / "config" / PIPELINE_CONFIG_BASENAME
    )
    pipeline_config = mdk.config.readYAMLConfig(pipeline_config_filename)
    module_path = pipeline_config["pipelines"][pipeline_name]["module_path"]
    return module_path.removesuffix(".py") + ".yml"


def _get_labels(
    pipeline_dir: pathlib.Path,
    environment: str,
) -> dict[str, str]:
    """Build the dict of labels that we want to attach to this pipeline run.

    Args:
        pipeline_dir (pathlib.Path): The path to the directory containing
            pipeline.py for the pipeline of interest.
        environment (str): The environment in which the pipeline will be run
            (typically dev, stage or prod)
    """
    # Read our various config fles etc.
    config_dir = pipeline_dir.parent.parent / "config"
    general_config_filename = config_dir / GENERAL_CONFIG_BASENAME
    general_config = mdk.config.readAndMergeYAMLConfig(
        config_filename=general_config_filename,
        environment=environment,
    )
    state_dir = pipeline_dir.parent.parent / "state"
    gcp_config_filename = state_dir / f"{environment}.yml"
    gcp_config = mdk.config.GCPConfig.from_yaml_file(str(gcp_config_filename))
    pipeline_name = pipeline_dir.stem

    # Get our finops labels from the cloud resources YAML file.
    finops_labels = _get_labels_from_cr_config(gcp_config.data_bucket)

    # Remove creation source, if it's there.
    finops_labels.pop("creation_source", None)

    # Get extra labels from the general config file.
    extra_labels = _get_extra_labels(general_config, pipeline_name)

    # Finally, create one last label indicating that this came from mdk run.
    source_label = {"pipeline_trigger_source": "mdk_run"}

    # Merge everything together.
    labels = finops_labels | extra_labels | source_label

    return labels


def _get_labels_from_cr_config(bucket_name: str) -> dict[str, str]:
    """Retrieve the list of fin ops labels to use for pipelines in this
    project.  The labels will be retrieved from a YAML file stored on GCS, via
    the top-level "tags" key in the YAML file.  The YAML file is originally
    generated during project creation by the project factory.

    Args:
        bucket_name (str): The bucket in which to search for the YAML file that
            contains the labels of interest.

    Returns:
        dict[str, str]: A dict containing a mapping of label name to label
            value.
    """
    logger.info("Retrieving finops labels...")
    labels = {}
    try:
        # Read our cloud resources config file from the project factory, on GCS.
        cloud_resources_config = mdk.config.readCloudResourcesConfig(bucket_name)

        # Read our fin ops labels.
        labels = cloud_resources_config["tags"]

    # Keep going if this fails for whatever reason.  (Failing the pipeline is
    #   not the outcome we want if e.g. this file format changes.)
    except Exception as e:
        logger.error(f"ERROR reading fin ops labels from cloud resources config: {e}")
    return labels


def _get_extra_labels(
    general_config: dict,
    pipeline_name: str,
) -> dict[str, str]:
    """Get additional labels to put on the pipeline, from the general config
    file.

    Args:
        general_config (dict): Configuration dict which includes the
            "additional_pipeline_labels" node.
        pipeline_name (str): Name of pipeline as reflected in the general_config
            as a child under the "additional_pipeline_labels" node.
    """
    labels_node = general_config.get("additional_pipeline_labels", {})

    if pipeline_name not in labels_node:
        raise RuntimeError(
            f"Error reading general config: Pipeline not given as key under"
            f' "additional_pipeline_labels": {pipeline_name}'
        )

    labels = labels_node[pipeline_name]

    return labels


def _parseCommandLine(argv) -> argparse.Namespace:
    """Get command line arguments and options."""

    parser = argparse.ArgumentParser(
        prog=pathlib.Path(__file__).name, description=__doc__
    )

    parser.add_argument(
        "pipeline_dir",
        type=pathlib.Path,
        metavar="PIPELINE_DIR",
        help=(
            "The path to the directory containing pipeline.py, config files,"
            " etc. for the pipeline of interest"
        ),
    )

    parser.add_argument(
        "--environment",
        "-e",
        choices=("dev", "stage", "prod"),
        default="dev",
        help=(
            "Suffix to use on the end of the GCP config.yaml filename, such"
            " that the GCP config file is named config/FLAG/gcp_config.yaml"
            " (where FLAG is one of: %(choices)s  (default: %(default)s)"
        ),
    )

    parser.add_argument(
        "--local",
        "-l",
        help="Run the pipeline locally, instead of submitting to Vertex AI Pipelines",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--lite",
        help="Enable lite mode (bypass expanded model registry)",
        action="store_true",
        default=False,
    )

    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    main()
