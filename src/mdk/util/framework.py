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

import mdk.config
import kfp.components
import kfp.dsl.yaml_component
import logging
import pathlib
from dataclasses import dataclass

from typing import Union

logger = logging.getLogger(__name__)

# Placeholder image string, the Artifact Registry URL will be replaced during postprocessing of the spec file.
AR_REPO_URL_PLACEHOLDER = "AR_REPO_URL_PLACEHOLDER"
PIPELINE_CONFIG_BASENAME = "pipeline_config.yml"


def getTargetImage(
    component_filename: str,
    component_name: str,
) -> str:
    """Given the name of a component, this function reads the component config
    file and returns the URL of the image it is using, where the Artifact
    Registry URL is replaced with a placeholder that will later be replaced
    during postprocessing of the spec file.

    Args:
        pipeline_config_filename (str): Path to pipeline config file (which
            describes the components that comprise the pipeline, etc.)
        component_name (str): Name of the component whose image URL we would
            like to infer.

    Returns:
        str: Image URL that this component will use, where the Artifact
            Registry URL is replaced with a placeholder that will later be
            replaced during postprocessing of the spec file.
    """
    # Read our config file and build our image name.
    pipeline_config_filename = (
        pathlib.Path(component_filename).parent.parent.parent
        / "config"
        / PIPELINE_CONFIG_BASENAME
    )
    pipeline_config = mdk.config.readYAMLConfig(pipeline_config_filename)
    component_config = pipeline_config["components"][component_name]

    # If this is using BYOC, we hardcode the component to use standard.  (We
    #   need a runner container that has KFP.)
    if "container_specs" in component_config:
        image_artifact = "standard:latest"

    # Otherwise, use what the pipeline config specifies.
    else:
        image_artifact = component_config["image_artifact"]

    image_uri = f"{AR_REPO_URL_PLACEHOLDER}/{image_artifact}"

    return image_uri


def get_resource_value_from_config(
    pipeline_config_filename: str,
    resource_type: str,
    componentName: str,
) -> Union[str, int, None]:
    """Given the name of a component, this function reads the pipeline config
    file, grabs any relevant resource values if specified, and returns them
    as a dictionary suitable for KFP's resource setting methods. See more here:
    https://docs.cloud.google.com/vertex-ai/docs/pipelines/machine-types

    Args:
        pipeline_config_filename (str): Path to pipeline config file (which
            describes the components that comprise the pipeline, etc.)
        resource_type (str): Type of resource to be specified.
            Options are: "cpu", "mem", "selector_constraint", "accelerator_type",
            "accelerator_limit"
        componentName (str): Name of the component whose resources we would
        like to update.

    Returns:
        Union[str, int, None]: Formatted value of the resource, or None if not found/applicable.
            - CPU/Memory: string (e.g., "4", "16G")
            - Accelerator Limit: int (e.g., 1)
            - Other: string or object as read from config
    """
    pipeline_config = mdk.config.readYAMLConfig(pipeline_config_filename)
    component_config = pipeline_config["components"].get(componentName, {})

    if resource_type == "cpu":
        return (
            str(component_config.get(resource_type))
            if component_config.get(resource_type) is not None
            else None
        )
    elif resource_type == "mem":
        mem_value = component_config.get(resource_type, None)
        if isinstance(mem_value, (int, float)):
            return f"{int(mem_value)}G"
        elif isinstance(mem_value, str):
            # Check if it already has common units "K" (kilobyte), "M" (megabyte), or "G" (gigabyte)
            if any(unit in mem_value for unit in ["G", "M", "K"]):
                return mem_value
            else:
                try:
                    float(mem_value)
                    return f"{mem_value}G"
                except ValueError:
                    # Not a number, assume it's already properly formatted or a non-standard unit
                    return mem_value
        return mem_value
    elif resource_type == "selector_constraint":
        return component_config.get(resource_type, None)
    elif resource_type == "accelerator_type":
        return component_config.get(resource_type, None)
    elif resource_type == "accelerator_limit":
        accel_limit = component_config.get(resource_type, None)
        if accel_limit is not None:
            try:
                return int(accel_limit)
            except (ValueError, TypeError):
                print(
                    f"Warning: Accelerator limit '{accel_limit}' for component '{componentName}' could not be converted to an integer. Setting to None."
                )
                return None
        return None
    return None


def apply_resource_settings_to_task(
    task_object,
    pipeline_config_filename: str,
    component_name: str,
):
    """
    Applies resource settings from the pipeline config to a given task object
    using its individual setter methods.

    Args:
        task_object: The KFP task object (e.g., result of a component call).
        pipeline_config_filename (str): Path to the pipeline config file.
        component_name (str): The name of the component as it appears in the config.
            component_name will be in kebab-case (my-component-name), and will be
            converted to snake-case.
    """
    component_name = component_name.replace("-", "_")

    # Apply Lite Mode environment variable if enabled
    pipeline_config = mdk.config.readYAMLConfig(pipeline_config_filename)
    if pipeline_config.get("lite", False):
        task_object.set_env_variable("MDK_LITE_MODE", "True")

    # Apply CPU limit
    cpu_limit = get_resource_value_from_config(
        pipeline_config_filename, "cpu", component_name
    )
    if cpu_limit is not None:
        task_object.set_cpu_limit(cpu_limit)

    # Apply Memory limit
    mem_limit = get_resource_value_from_config(
        pipeline_config_filename, "mem", component_name
    )
    if mem_limit is not None:
        task_object.set_memory_limit(mem_limit)

    # Apply Node Selector Constraint
    selector_constraint = get_resource_value_from_config(
        pipeline_config_filename, "selector_constraint", component_name
    )
    if selector_constraint is not None:
        if isinstance(selector_constraint, dict):
            # This is a common pattern for node selectors, if your `add_node_selector_constraint`
            # supports iterating a dict or takes multiple args.
            for key, value in selector_constraint.items():
                task_object.add_node_selector_constraint(key, value)
        else:
            task_object.add_node_selector_constraint(selector_constraint)

    # Apply Accelerator Type
    accelerator_type = get_resource_value_from_config(
        pipeline_config_filename, "accelerator_type", component_name
    )
    if accelerator_type is not None:
        task_object.set_accelerator_type(accelerator_type)

    # Apply Accelerator Limit
    accelerator_limit = get_resource_value_from_config(
        pipeline_config_filename, "accelerator_limit", component_name
    )
    if accelerator_limit is not None:
        task_object.set_accelerator_limit(accelerator_limit)

    return task_object


def loadComponentSpec(
    pipeline_config_filename: str,
    componentName: str,
) -> kfp.dsl.yaml_component.YamlComponent:
    """Given the name of a component, this function reads the pipeline config
    file, infers the location of the spec file from the location of the python,
    module, and then loads that spec file.

    Args:
        comopnent_config_filename (str): Path to pipeline config file (which
            describes the components that comprise the pipeline, etc.)
        component_name (str): Name of the component whose component spec we
            would like to load.

    Returns:
        kfp.dsl.yaml_component.YamlComponent: The component loaded from the spec
            file corresponding to component_name, based on the pipeline config
            file.

    """
    # Note that the config dict is cached, so the file is not re-read every time
    #   this is called.
    pipeline_config = mdk.config.readYAMLConfig(pipeline_config_filename)
    module_path = pipeline_config["components"][componentName]["module_path"]
    # The spec file has the same name as the Python module, just with a .yml
    #   file extension instead of .py.
    spec_filename = module_path.removesuffix(".py") + ".yml"
    component = kfp.components.load_component_from_file(spec_filename)
    return component


def get_pipeline_mapping():
    """This function searches the model_product/ and examples/ directories to
    find all of the pipeline names and directories.

    Returns:
        dict[str, pathlib.Path]: A mapping giving the name of each pipeline
            (keys), with the corresponding directory with the pipeline source
            (values).
    """
    if not pathlib.Path("model_products").is_dir():
        raise RuntimeError(
            "This process is meant to be run from the top of the source"
            " directory, such that the model_products directory is a"
            " subdirectory of the current working directory"
        )

    # Find all the model product directories in the model_products/ and
    #   examples/ directories.
    pipeline_mapping = {}
    model_products_dirs = ("model_products", "examples/model_products")
    for products_dir in [pathlib.Path(d) for d in model_products_dirs]:
        # It's OK for examples/model_products/ not to exist.  We already
        #   verified model_products/ exists.
        if not products_dir.is_dir():
            continue

        # For each model product:
        for model_product in products_dir.iterdir():
            # Read the pipeline_config.yaml file for this model product.
            pipeline_config_filename = (
                model_product / "config" / PIPELINE_CONFIG_BASENAME
            )
            pipeline_config = mdk.config.readYAMLConfig(pipeline_config_filename)

            # For each pipeline in the pipeline config. get the pipeline name
            #   and pipeline directory.
            for pipeline_name, config in pipeline_config["pipelines"].items():
                module_path = config["module_path"]
                pipeline_dir = pathlib.Path(module_path).parent

                if pipeline_name in pipeline_mapping:
                    raise RuntimeError(
                        f"Error reading {PIPELINE_CONFIG_BASENAME} file(s):"
                        f" Duplicate pipeline name: {pipeline_name}\nDuplicates"
                        f" are located at:"
                        f"\n  - {pipeline_mapping[pipeline_name]}"
                        f"\n  - {pipeline_dir}"
                    )

                pipeline_mapping[pipeline_name] = pathlib.Path(pipeline_dir)

    return pipeline_mapping


def getPipelineSpecFilename(
    pipeline_name: str,
    config_filename: str | None = None,
) -> str:
    """Given the name of a pipeline, this function reads the pipeline config
    file and then infers the location of the spec file from the location of
    the python module.

    Args:
        pipeline_name (str): Name of the component whose spec filename we
            would like to infer.

    Returns:
        str: Spec filename corresponding to the component with name
            componentName.
    """
    # Note that the config dict is cached, so the file is not re-read every time
    #   this is called.
    pipeline_config = mdk.config.readYAMLConfig(config_filename)
    module_path = pipeline_config["pipelines"][pipeline_name]["module_path"]
    # The spec file has the same name as the Python module, just with a .yml
    #   file extension instead of .py.
    return module_path.removesuffix(".py") + ".yml"


def get_relative_path(path: str) -> pathlib.Path:
    """Given the absolute path of a file in the examples/ or model_products/,
    directory, return the path relative to the base of the source directory.  (A
    relative path is necessary so that the pipeline can be compiled on one
    system and then executed on a different system.)

    Example:

    >>> get_relative_path("/path/to/source_code/model_products/foo/bar.py")
    "model_products/foo/bar.py"

    >>> get_relative_path("/path/to/source_code/examples/model_products/foo/bar.py")
    "examples/model_products/foo/bar.py"

    Args:
        path (str): The absolute path of the file of interest.

    Returns:
        pathlib.Path: The input path, relative to the source directory.
    """
    path_ = pathlib.Path(path)
    dir = _get_path_from_this_onward(path_, "examples")
    if not dir:
        dir = _get_path_from_this_onward(path_, "model_products")
    if not dir:
        raise RuntimeError(
            f"Unable to find examples/ or model_products/ in path: {path}"
        )
    return dir


def _get_path_from_this_onward(
    path: pathlib.Path,
    target: str,
) -> pathlib.Path | None:
    """ """
    parts = pathlib.Path(path).parts
    try:
        where = parts.index(target)
    except ValueError:
        return None
    return pathlib.Path("/".join(parts[where:]))


@dataclass(frozen=True)
class PipelinePaths:
    """
    Centralizes all path logic for pipeline compilation and execution.

    Takes a 'base' path (usually from __file__) and derives all
    other necessary paths, providing them as POSIX strings
    ready for use in configs or external systems.
    """

    base: pathlib.Path

    @property
    def config_dir(self) -> pathlib.Path:
        """The 'config/' directory Path object."""
        return self.base.parent.parent.parent / "config"

    @property
    def state_dir(self) -> pathlib.Path:
        """The 'state/' directory Path object."""
        return self.base.parent.parent.parent / "state"

    # --- Methods that return standardized POSIX strings ---

    def get_gcp_config(self, environment: str) -> str:
        """Returns the POSIX path string for the environment-specific state file."""
        return (self.state_dir / f"{environment}.yml").as_posix()

    def get_general_config(self) -> str:
        """Returns the POSIX path string for the main config.yml."""
        return (self.config_dir / "config.yml").as_posix()

    def get_pipeline_config(self) -> str:
        """Returns the POSIX path string for the pipeline_config.yml."""
        return (self.config_dir / PIPELINE_CONFIG_BASENAME).as_posix()
