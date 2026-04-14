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

"""Compile all Kubeflow components, compile the Kubeful pipeline, and perform
any necessary post-processing on the compiled spec files.
"""

import mdk.util.framework
import mdk.config
import kfp
import kfp.compiler
import yaml
import argparse
import collections
import importlib.util
import logging
import pathlib
import sys

logger = logging.getLogger(__name__)

ImageName = collections.namedtuple("ImageName", ["base", "hash", "tag"])

STANDARD_IMAGE_ARTIFACT = "standard:latest"


def main():
    """This function is executed when the module is called from the command line."""
    logging.basicConfig(format="", level=logging.INFO)

    # Read our command line arguments and our config file.
    args = _parseCommandLine(sys.argv)
    compilePipeline(args.pipeline_dir, args.ar_repo, args.tag)


def compilePipeline(
    pipeline_dir: pathlib.Path,
    ar_repo: str,
    tags: list[str] = [],
):
    logger.info(f"Compiling components plus pipeline: {pipeline_dir}")

    # Compile all our components.
    pipeline_config_filename = (
        pipeline_dir.parent.parent
        / "config"
        / mdk.util.framework.PIPELINE_CONFIG_BASENAME
    )
    pipeline_config = mdk.config.readYAMLConfig(pipeline_config_filename)
    compileAllComponents(pipeline_config, ar_repo, tags)

    # Compile the pipeline.
    pipeline_name = pipeline_dir.stem
    pipelineModulePath = pipeline_config["pipelines"][pipeline_name]["module_path"]
    pipelineFunctionName = pipeline_config["pipelines"][pipeline_name]["function"]
    compiledPipelineFilename = pipelineModulePath.removesuffix(".py") + ".yml"
    compileSingleComponent(
        pipelineModulePath, pipelineFunctionName, compiledPipelineFilename
    )

    logger.info("Done.")


def compileAllComponents(
    config: dict,
    ar_repo: str,
    tags: list[str],
):
    """Walk thorugh the pipeline config file and compile all the
    components listed in the file.

    If a component is marked with the optional `is_user_defined: true`,
    then the compile step is skipped for that specific component

    Raises:
        - ValueError if the user specifies a container other than standard:latest
            but does not specify the container_sepcs
        - Runtime error if the component.py file is not found
    """
    # Compile each component in the pipeline config file:
    for componentName, componentConfig in config["components"].items():
        # Check for container_specs key if the user is using a non-standard image
        if (componentConfig["image_artifact"] != STANDARD_IMAGE_ARTIFACT) and (
            "container_specs" not in componentConfig
        ):
            raise ValueError(
                f"Error in {mdk.util.framework.PIPELINE_CONFIG_BASENAME}:"
                f" {componentName}: {componentConfig['image']}' is not"
                f" {STANDARD_IMAGE_ARTIFACT} => 'container_specs' must be"
                f" provided."
            )

        componentModulePath = componentConfig["module_path"]
        if not componentModulePath.endswith(".py"):
            raise RuntimeError(
                "Error reading pipeline config: Component :"
                " Expected a filename ending in .py; got "
            )
        functionName = componentConfig["function"]
        compiledFilename = componentModulePath.removesuffix(".py") + ".yml"
        compileSingleComponent(componentModulePath, functionName, compiledFilename)

        # Do post-processing: (Set UV path and add component module path)
        _postprocess(compiledFilename, componentModulePath, functionName, ar_repo, tags)


def compileSingleComponent(
    modulePath: pathlib.Path,
    functionName: str,
    compiledFilename: pathlib.Path,
):
    """Compile a Kubeflow pipeline to file."""
    logger.info(f"Compiling: {compiledFilename}")

    # Import the module.
    path = pathlib.Path(modulePath)
    module = _importFromPath(path.stem, path)

    # Get the function we want to run as the pipeline.
    try:
        function = module.__dict__[functionName]
    except KeyError as e:
        e.add_note(
            f"Error in pipeline config: Error getting function in module:"
            f" {modulePath} does not have a function named {functionName}"
        )
        raise

    kfp.compiler.Compiler().compile(
        pipeline_func=function,
        package_path=str(compiledFilename),
    )


def _importFromPath(
    module_name: str,
    file_path: pathlib.Path,
):
    """Import a module given a file path that isn't known until runtime.

    Args:
        module_name (str): The name the module will be referred to after it is
            imported.
        file_path (pathlib.Path): The path in the filesystem where the module
            being imported resides.
    """
    # This is cribbed from the Python documentation:
    #
    # https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if not spec:
        raise RuntimeError(f"Unable to get spec from file: {file_path}")
    if not spec.loader:
        raise RuntimeError(f"loader member on spec is None: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _postprocess(
    componentFilename: str,
    componentModulePath: str,
    functionName: str,
    ar_repo: str,
    images_with_tags: list[str],
):
    """This will do the following:

    - Set the Python interpreter in the spec file to the one produced by uv,
      rather than the image's system default python.  (That is, we change
      "python3" to "/app/.venv/bin/python".)

    - Tell Kubeflow where to find the Python module corresponding to this spec,
      via the --component_module_path argument being sent to the exectuor.

    - Replace a placeholder Artifact Registry URL with the correct Artifact
      Registry URL.

    - If we've been told to do so, replace the image tags in the file
       with a specific image digest.

    Args:
        componentFilename (str): The compiled component YAML file to edit.
        componentModulePath (str): The Python module corresponding to the
            spec file being edited.
        functionName (str): The name of the function that the containerized
            component will execute.  (This drives a key that we will search
            for.)
        ar_repo (str): The Artifact Registry URL
        images_with_tags (list[str]): Image digest replacement tags
    """
    # Read our config:
    with open(componentFilename, "r") as fin:
        spec = yaml.safe_load(fin)

    # Edit our config:

    # If the function name is 'foo_bar', then the key we'd need would be
    #   'exec-foo-bar' (where underscores have been replaced with dashes).
    executorLabel = f"exec-{functionName.replace('_', '-')}"
    container = spec["deploymentSpec"]["executors"][executorLabel]["container"]

    # Add --component_module_path to the executor arguments.
    args = container["args"]
    args.append("--component_module_path")
    args.append(componentModulePath)

    # Fix our command to point to our virtual environment's interpreter.
    command = container["command"]
    if command[0] != "python3":
        raise RuntimeError(
            f"{componentFilename}: Unexpected value for container command."
            f"  Expected: python3  Got: {command[0]}"
        )
    command[0] = "/app/.venv/bin/python"

    # Replace our AR repo placeholder with a specific AR URL.
    # (Note that we do this *before* we replace tags with specific digests,
    #   because images_with_tags (below) will use the full image URL, and not
    #   the placeholder string.)

    PLACEHOLDER = mdk.util.framework.AR_REPO_URL_PLACEHOLDER
    container["image"] = container["image"].replace(PLACEHOLDER, ar_repo)

    # Fix our image to point to a specific tag:

    old_image_str = container["image"]
    old_image_name = _parseImageName(old_image_str)
    # For each of the new target images with tags:
    for new_image_str in images_with_tags:
        # If the existing base image name (no tag) matches one of the target
        #   base image names (no tag) that we were given, set the image to that
        #   target image (with tag).  In other words, we're updating the tag if
        #   the base image names match.
        new_image_name = _parseImageName(new_image_str)
        if new_image_name.base == old_image_name.base:
            logger.info(f"Replacing:\n\t{old_image_str}\n\t=>\n\t{new_image_str}")
            container["image"] = new_image_str

    # Write our modified config:
    with open(componentFilename, "w") as fout:
        fout.write(yaml.dump(spec))


def _parseImageName(
    image_str: str,
) -> ImageName:
    """Parse an image into components.

    Args:
        image_str (str): The image name to be parsed, of the format IMAGE,
        IMAGE:TAG or IMAGE@sha256:HASH.

    Returns:
        An ImageName named tuple with 3 members named base, hash and tag, which
            will correspond to the base image name, the hash (if given), and the
            tag (if given).

    Examples:

    >>> kubeflow.run_pipeline._parseImageName("my_image:latest")
    ImageName(base='my_image', hash=None, tag='latest')

    >>> kubeflow.run_pipeline._parseImageName("my_image@sha256:1234567890abcdef")
    ImageName(base='my_image', hash='1234567890abcdef', tag=None)
    """
    parts = image_str.split(":")
    if len(parts) > 2:
        raise RuntimeError(
            f"Expected something like IMAGE_NAME or IMAGE_NAME:TAG or"
            f" IMAGE_NAME@sha256:HASH; got: {image_str}"
        )

    name, hash, tag = None, None, None

    # If the image name has a hash:
    if parts[0].endswith("@sha256"):
        if len(parts) == 1:
            raise RuntimeError(
                f"Image name says @sha256, but hash is missing: {image_str}"
            )
        name = parts[0].removesuffix("@sha256")
        hash = parts[1]

    # Otherwise, set the tag, if the image has one.
    else:
        name = parts[0]
        if len(parts) == 2:
            tag = parts[1]

    return ImageName(name, hash, tag)


def _parseCommandLine(argv):
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
        "ar_repo",
        metavar="ARTIFACT_REGISTRY_REPOSITORY",
        help="Artifact registry repository to use for each component",
    )

    parser.add_argument(
        "--tag",
        "-t",
        help=(
            "Replace all usages of IMAGE:OLD_TAG in the compiled spec files with"
            " IMAGE:NEW_TAG.  This argument can be supplied multiple times.  The"
            " IMAGE:NEW_TAG argument can optionally be surrounded by square"
            " brackets.  IMAGE:NEW_TAG can be a sha256 digest."
        ),
        action="append",
        metavar="IMAGE:NEW_TAG",
        default=[],
    )

    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    main()
