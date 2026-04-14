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
This module provides general-purpose utility functions creating Vertex AI
Custom Jobs.
"""

import logging
from typing import List, Dict, Optional, Any

from google.cloud import aiplatform
from google.cloud.aiplatform_v1.types import JobState

from mdk.custom_job.models import (
    CustomJobCommonConfig,
    DirectJobSpecificConfig,
    ScriptJobSpecificConfig,
)

logger = logging.getLogger(__name__)


def _filter_none_params(params: Dict) -> Dict:
    """Helper to remove None values from a dictionary."""
    return {k: v for k, v in params.items() if v is not None}


def _get_effective_accelerator_config(
    accelerator_type: Optional[str], accelerator_count: int
) -> Dict:
    """Helper to determine accelerator configuration, returning accelerator_type and count."""
    accelerator_config = {}
    if accelerator_type:
        accelerator_config["accelerator_type"] = accelerator_type
        accelerator_config["accelerator_count"] = (
            accelerator_count if accelerator_count > 0 else 1
        )
    elif accelerator_count > 0:
        logger.warning(
            "`accelerator_count` was specified without `accelerator_type`. "
            "Accelerators will not be requested."
        )
        accelerator_config["accelerator_count"] = 0
    else:
        accelerator_config["accelerator_count"] = 0

    return accelerator_config


def _get_full_container_uri(image: str, ar_repo: str) -> str:
    """Prepares the container URI for a custom training job."""
    # Split the image name at the first slash to get the potential registry part
    # e.g., "byoc:latest" -> "byoc:latest"
    # e.g., "us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-17.py310:latest" -> "us-docker.pkg.dev"
    registry_or_first_segment = image.split("/", 1)[0]
    is_full_uri = True if "." in registry_or_first_segment else False

    if is_full_uri:
        return image
    else:
        return f"{ar_repo}/{image}"


def _convert_config_dict_to_args(config_dict: Dict) -> List[str]:
    """Converts key-value pairs into command line arguments."""
    cli_args = []
    if config_dict:
        for k, v in config_dict.items():
            cli_flag_name = f"--{k.replace('_', '-')}"
            cli_args.extend([cli_flag_name, str(v)])
    return cli_args


def _get_custom_job_logs_url(resource_name: str) -> str:
    """Constructs the logs URL from the custom job resource name"""
    # The resource_name is typically: projects/PROJECT_ID/locations/REGION/customJobs/JOB_ID
    parts = resource_name.split("/")
    job_id = parts[-1]
    project_id = parts[1]

    logs_url = (
        f"https://console.cloud.google.com/logs/query?"
        f"project={project_id}&"
        f"query=resource.labels.job_id%3D%22{job_id}%22"
    )
    return logs_url


def _get_custom_job_console_url(resource_name: str) -> str:
    """Constructs the google cloud console URL from the custom job resource name"""
    parts = resource_name.split("/")
    job_id = parts[-1]
    location_id = parts[3]

    console_url = (
        f"https://console.cloud.google.com/vertex-ai/"
        f"locations/{location_id}/"
        f"training/{job_id}"
    )
    return console_url


def create_and_run_vertex_custom_job_direct(
    common_config: CustomJobCommonConfig,
    specific_config: DirectJobSpecificConfig,
) -> aiplatform.CustomJob:
    """
    Creates and runs a Vertex AI CustomJob by directly specifying worker_pool_specs.
    Leverages Pydantic models for validated input.
    """
    logger.info(
        f"Creating CustomJob '{common_config.display_name}' with direct worker_pool_specs."
    )

    # --- Construct machine_spec ---
    machine_spec = {"machine_type": common_config.machine_type}
    accelerator_details = _get_effective_accelerator_config(
        common_config.accelerator_type, common_config.accelerator_count
    )
    if accelerator_details["accelerator_count"] > 0:
        machine_spec.update(accelerator_details)

    # --- Construct container_spec ---
    container_spec = {"image_uri": common_config.image_uri}
    if specific_config.command is not None:
        container_spec["command"] = specific_config.command
    if common_config.args is not None:
        container_spec["args"] = common_config.args
    if common_config.env_vars is not None:
        container_spec["env"] = common_config.env_vars

    # --- Construct worker_pool_specs ---
    worker_pool_specs = [
        {
            "machine_spec": machine_spec,
            "replica_count": common_config.replica_count,
            "container_spec": container_spec,
        }
    ]

    # --- Prepare CustomJob constructor parameters ---
    constructor_params = _filter_none_params(
        common_config.model_dump(exclude_unset=True, exclude_none=True)
    )
    constructor_params.update({"worker_pool_specs": worker_pool_specs})

    # Remove items moved into worker_pool_specs
    constructor_params.pop("image_uri", None)
    constructor_params.pop("args", None)
    constructor_params.pop("env_vars", None)

    # Remove accelerator config fields if they're still there after dict conversion
    constructor_params.pop("accelerator_type", None)
    constructor_params.pop("accelerator_count", None)
    constructor_params.pop("machine_type", None)
    constructor_params.pop("replica_count", None)

    # --- Prepare CustomJob.run() / .submit() parameters ---
    run_param_names = [
        "service_account",
        "network",
        "timeout",
        "enable_web_access",
        "experiment",
        "experiment_run",
        "tensorboard",
        "restart_job_on_worker_restart",
        "create_request_timeout",
        "disable_retries",
        "scheduling_strategy",
        "max_wait_duration",
        "psc_interface_config",
    ]

    run_params = _filter_none_params(
        {k: constructor_params.pop(k, None) for k in run_param_names}
    )

    logger.info(f"Initializing CustomJob with parameters: {constructor_params}")
    custom_job = aiplatform.CustomJob(**constructor_params)

    if (
        common_config.persistent_resource_id
        and "persistent_resource_id" not in constructor_params
    ):
        run_params["persistent_resource_id"] = common_config.persistent_resource_id

    logger.info(
        f"Running CustomJob '{common_config.display_name}' with parameters: {run_params}"
    )
    custom_job.run(**run_params)

    resource_name = custom_job.resource_name
    if custom_job.state == JobState.JOB_STATE_SUCCEEDED:
        logger.info(
            f"CustomJob '{common_config.display_name}' completed successfully. Resource Name: {resource_name}"
        )
    else:
        logger.info(
            f"CustomJob '{common_config.display_name}' failed with state: {custom_job.state}. Resource Name: {resource_name}"
        )

    logger.info(
        f"*** CustomJob Console UI available at: {_get_custom_job_console_url(resource_name)} ***"
    )
    logger.info(
        f"*** CustomJob logs available at: {_get_custom_job_logs_url(resource_name)} ***"
    )
    return custom_job


def create_and_run_vertex_custom_job_from_script(
    common_config: CustomJobCommonConfig,
    specific_config: ScriptJobSpecificConfig,
) -> aiplatform.CustomJob:
    """
    Creates and runs a Vertex AI CustomJob from a local Python script
    using aiplatform.CustomJob.from_local_script.
    Leverages Pydantic models for validated input.
    """
    logger.info(
        f"Creating CustomJob '{common_config.display_name}' from local script: {specific_config.script_path}"
    )

    from_script_params: Dict[str, Any] = _filter_none_params(
        common_config.model_dump(exclude_unset=True, exclude_none=True)
    )
    from_script_params.update(
        _filter_none_params(
            specific_config.model_dump(exclude_unset=True, exclude_none=True)
        )
    )

    # aiplatform.CustomJob.from_local_script requires it to be named 'container_uri' and not 'image_uri'
    if "image_uri" in from_script_params:
        from_script_params["container_uri"] = from_script_params.pop("image_uri")
    # Similarly, rename 'env_vars' to environment_variables
    if "env_vars" in from_script_params:
        from_script_params["environment_variables"] = from_script_params.pop("env_vars")

    accelerator_details = _get_effective_accelerator_config(
        common_config.accelerator_type, common_config.accelerator_count
    )
    if accelerator_details["accelerator_count"] > 0:
        from_script_params["accelerator_type"] = accelerator_details["accelerator_type"]
        from_script_params["accelerator_count"] = accelerator_details[
            "accelerator_count"
        ]

    run_param_names = [
        "service_account",
        "network",
        "timeout",
        "enable_web_access",
        "experiment",
        "experiment_run",
        "tensorboard",
        "restart_job_on_worker_restart",
        "create_request_timeout",
        "disable_retries",
        "scheduling_strategy",
        "max_wait_duration",
        "psc_interface_config",
        "persistent_resource_id",
    ]

    run_params = _filter_none_params(
        {k: from_script_params.pop(k, None) for k in run_param_names}
    )

    logger.info(
        f"Calling CustomJob.from_local_script with parameters: {from_script_params}"
    )
    custom_job = aiplatform.CustomJob.from_local_script(**from_script_params)

    if common_config.persistent_resource_id:
        run_params["persistent_resource_id"] = common_config.persistent_resource_id

    logger.info(
        f"Running CustomJob '{common_config.display_name}' with parameters: {run_params} (created from local script)."
    )
    custom_job.run(**run_params)

    resource_name = custom_job.resource_name
    if custom_job.state == JobState.JOB_STATE_SUCCEEDED:
        logger.info(
            f"CustomJob '{common_config.display_name}' completed successfully. Resource Name: {resource_name}"
        )
    else:
        logger.info(
            f"CustomJob '{common_config.display_name}' failed with state: {custom_job.state}. Resource Name: {resource_name}"
        )

    logger.info(
        f"*** CustomJob Console UI available at: {_get_custom_job_console_url(resource_name)} ***"
    )
    logger.info(
        f"*** CustomJob logs available at: {_get_custom_job_logs_url(resource_name)} ***"
    )
    return custom_job


def handle_custom_job_if_configured(
    gcp_config: Dict[str, str],
    model_config: Dict[str, Any],
    pipeline_config: Dict[str, Any],
    component_name: str,
    staging_bucket: str,
    base_output_dir: str,
    input_artifact_map: Optional[Dict[str, str]] = None,
) -> bool:
    """
    Executes a custom Vertex AI job if 'container_specs' are defined in the pipeline config
    for the given component. If a custom job is launched, organizes the args for the CustomJob
    and calls the corresponding type of job (direct, or from script), then terminates upon
    successful completion.

    Args:
        gcp_config: Dictionary containing GCP configuration
        model_config: Dictionary containing the model configuration YAML.
        pipeline_config: Dictionary containing the pipeline configuration YAML.
        component_name: The name of the KFP component (e.g., "xgb_train").
        staging_bucket: The gcs location to store custom job metadata.
        base_output_dir: The output directory for artifacts.
        input_artifact_map: A dictionary where keys are logical input names
                             (e.g., "train_dataset") and values are the
                             corresponding KFP `dsl.Input` objects' uris.
    Return:
        boolean: Whether the job completed successfully, or not running a custom job.
    """
    # Check if custom container specs are provided for this component
    component_specs = pipeline_config.get("components", {}).get(component_name, {})
    custom_container_specs = component_specs.get("container_specs", None)

    if custom_container_specs:
        container_uri = _get_full_container_uri(
            image=component_specs["image_artifact"],
            ar_repo=gcp_config.get("artifact_registry_repo", ""),
        )

        # Prepare arguments for the custom job
        job_args = _convert_config_dict_to_args(model_config)
        artifact_args = _convert_config_dict_to_args(input_artifact_map or {})
        job_args.extend(artifact_args)
        # Pass the output dir and the project_id to the container args
        job_args.extend(["--base-output-dir", base_output_dir])
        job_args.extend(["--project", gcp_config.get("project_id", "")])

        _custom_container_specs_copy = custom_container_specs.copy()

        # Extract potential accelerator and machine parameters
        accelerator_type = _custom_container_specs_copy.pop("accelerator_type", None)
        accelerator_count = _custom_container_specs_copy.pop("accelerator_count", 0)
        machine_type = _custom_container_specs_copy.pop("machine_type", "n1-standard-4")
        replica_count = _custom_container_specs_copy.pop("replica_count", 1)

        shared_kwargs = {
            "display_name": f"{component_name}_custom_job",
            "project": gcp_config.get("project_id"),
            "location": gcp_config.get("region"),
            "image_uri": container_uri,
            "args": job_args,
            "service_account": gcp_config.get("pipeline_service_account"),
            "staging_bucket": staging_bucket,
            "base_output_dir": base_output_dir,
            "env_vars": custom_container_specs.get("env_vars", None),
            "accelerator_type": accelerator_type,
            "accelerator_count": accelerator_count,
            "machine_type": machine_type,
            "replica_count": replica_count,
        }

        if "script_path" in _custom_container_specs_copy:
            # Create a custom job from a Python script
            specific_config = ScriptJobSpecificConfig(
                script_path=_custom_container_specs_copy.pop("script_path", None),
                requirements=_custom_container_specs_copy.pop("requirements", None),
                python_module_name=_custom_container_specs_copy.pop(
                    "python_module_name", None
                ),
            )

            merged_common_config_data = shared_kwargs | _custom_container_specs_copy
            common_config = CustomJobCommonConfig(**merged_common_config_data)

            create_and_run_vertex_custom_job_from_script(
                common_config=common_config, specific_config=specific_config
            )
        else:
            specific_config = DirectJobSpecificConfig(
                command=_custom_container_specs_copy.pop("command", None)
            )

            merged_common_config_data = shared_kwargs | _custom_container_specs_copy
            common_config = CustomJobCommonConfig(**merged_common_config_data)

            create_and_run_vertex_custom_job_direct(
                common_config=common_config, specific_config=specific_config
            )
        # Job completed successfully
        return True
    # Not running a Custom Job
    return False
