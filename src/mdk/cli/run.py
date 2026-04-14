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

"""Execute a Vertex Pipeline job."""

import logging
import os
import pathlib

logger = logging.getLogger(__name__)

WIDTH = 80
PIPELINE_CONFIG_BASENAME = "pipeline_config.yml"


def run(
    *,
    pipeline_name: str,
    environment: str,
    local: bool,
    lite: bool = False,
):
    """Execute a Vertex Pipeline job.

    Args:
        pipeline_name (str): The dname of the pipeline to run.
        local (bool): If this is True, the pipeline will run locally, instead of
            being submitted to Vertex AI Pipelines.
        environment (str): Suffix to use on the end of the GCP config.yaml
            filename, such that the GCP config file is named
            config/FLAG/gcp_config.yaml, where FLAG is usually one of: (dev,
            stage, prod)
    """
    # We import here to speed up mdk list.
    import mdk.config
    import mdk.util.framework

    pipeline_mapping = mdk.util.framework.get_pipeline_mapping()
    if pipeline_name not in pipeline_mapping:
        raise RuntimeError(
            f"Unable to find pipeline: {pipeline_name}  (Does it exist in the"
            f" {PIPELINE_CONFIG_BASENAME} files?)"
        )
    pipeline_dir = pipeline_mapping[pipeline_name]

    # Read our config file to get our Artifact Registry repo.
    gcp_config_filename = pipeline_dir.parent.parent / f"state/{environment}.yml"
    gcp_config = mdk.config.GCPConfig.from_yaml_file(gcp_config_filename).model_dump()
    ar_repo = gcp_config.get("artifact_registry_repo")

    # Build our Docker images, compile our pipeline, and run our pipeline.
    image_names_with_digests = buildImages(pipeline_dir, ar_repo, local)
    compilePipeline(pipeline_dir, ar_repo, image_names_with_digests)
    executePipeline(pipeline_dir, environment, local, lite)


def buildImages(
    pipeline_config_filename: pathlib.Path,
    ar_repo: str,
    is_local: bool,
) -> list[str]:
    """Build all the Docker images that are listed in pipeline_config.yml.

    Args:
        is_local (bool): Whether the pipeline is running locally.  If this is
            False, buildImages() will return an empty list.  (We don't want
            digests for local run, because the local run process cannot see
            digests, and using the :latest tag works great for local runs
            anyway.)
        ar_repo (str): Artifact Registry repository to use for Docker image
            upload.

    Returns:
        list[str]: The image with digest, for each image that gets built.
    """
    # We import here to speed up mdk list.
    import mdk.pipeline_tools.build_images

    # Print info to the terminal.
    cmd = [
        "python",
        "-m",
        "mdk.pipeline_tools.build_images",
        pipeline_config_filename,
        ar_repo,
    ]
    if is_local:
        cmd += ["--local"]
    _printCommand("Building Docker images", cmd)

    # Call buildImages().
    image_names_with_digests = mdk.pipeline_tools.build_images.buildImages(
        pipeline_config_filename, ar_repo, is_local
    )

    logger.info("Got image names with digests:")
    for image in image_names_with_digests:
        # print(), not logger, since we want this to go to stdout without
        #   embellishment.
        print(f"\t{image}")

    return image_names_with_digests


def compilePipeline(
    pipeline_dir: pathlib.Path,
    ar_repo: str,
    image_names_with_digests: list[str],
):
    """Run a pipeline by executing src/mdk/compile_pipeline.py.

    Args:
        pipeline_dir (pathlib.Path): The path to the directory containing
            pipeline.py, config files, etc. for the pipeline of interest.
        are_repo (str): Artifact registry repository URL to replace the
            placeholder value that will be in each spec file.
        image_names_with_digests (list[str]): Image names to pass to
            run_pipeline.py.
    """
    # We import here to speed up mdk list.
    import mdk.pipeline_tools.compile_pipeline

    # Print info to the terminal.
    cmd = [
        "python",
        "-m",
        "mdk.pipeline_tools.compile_pipeline",
        pipeline_dir,
        ar_repo,
    ]
    for name in image_names_with_digests:
        cmd += ["--tag", name]
    _printCommand("Compiling pipeline", cmd)

    # Call compilePipeline().
    mdk.pipeline_tools.compile_pipeline.compilePipeline(
        pipeline_dir, ar_repo, image_names_with_digests
    )


def executePipeline(
    pipeline_dir: pathlib.Path,
    environment: str,
    is_local: bool,
    is_lite: bool = False,
):
    """Run a pipeline by executing src/mdk/execute_pipeline.py.

    Args:
        pipeline_dir (pathlib.Path): The path to the directory containing
            pipeline.py, config files, etc. for the pipeline of interest.
        is_local (bool): If this is True, the pipeline will run locally, instead
            of being submitted to Vertex AI Pipelines.
        environment (str): Environment in which to run the pipeline.  This
            controls which set of config files will be used.  Usually one of:
            (dev, stage, prod)
    """
    # Import here to speed up mdk list.
    import mdk.pipeline_tools.execute_pipeline

    # Print info to the terminal.
    cmd = [
        "python", "-m", "mdk.pipeline_tools.execute_pipeline",
        pipeline_dir,
        "--environment", environment,
    ]  # fmt: skip
    if is_local:
        cmd.append("--local")
    if is_lite:
        cmd.append("--lite")
    _printCommand("Running pipeline" if is_local else "Submitting pipeline", cmd)

    # Call executePipeline().
    mdk.pipeline_tools.execute_pipeline.executePipeline(
        pipeline_dir, environment, is_local, is_lite
    )


def _printCommand(
    msg: str,
    cmd: list[object],
):
    """Print to the terminal the command that the user can use to do this without
    using mdk run, just to document what is going on and to allow the user to
    replicate this step from the terminal if they wish.

    Args:
        msg (str): Banner message to print ahead of the command line.
        cmd (list[str]): Command line to be printed to the terminal.
    """
    # If this is Windows, explicitly provide the path to the interpreter.
    cmd = cmd.copy()
    cmd = [str(x) for x in cmd]
    if (cmd[0] == "python") and (os.name == "nt"):
        cmd[0] = ".venv/Scripts/python.exe"

    padded = f" {msg} "
    logger.info(f"{padded:=^{WIDTH}}")
    logger.info(f"\n{' '.join(cmd)}\n")
    logger.info("-" * WIDTH)
