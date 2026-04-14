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

"""This module provides a command line interface to build the Docker images
necessary for our Kubeflow pipelines.
"""

import mdk.util.framework
import mdk.config
import dotenv
import argparse
import logging
import pathlib
import subprocess
import os
import sys

logger = logging.getLogger(__name__)

WIDTH = 80


def main():
    """This function is executed when the module is called from the command line."""
    logging.basicConfig(format="", level=logging.INFO)

    # Read our command line arguments.
    args = _parseCommandLine(sys.argv)

    # If we're just supposed to print out a list of image names, do so.
    if args.list_only:
        images_to_build = getImagesToBuild(
            args.ar_repo, args.pipeline_dir, args.image_name
        )
        for image in images_to_build:
            logger.info(image)
        return

    # Build our images.
    image_names_with_digests = buildImages(
        args.pipeline_dir, args.ar_repo, args.local, args.image_name, args.git_sha
    )

    # Print the image names with digests to the terminal.  (The user may want
    #   them for the --tag argument in compile_pipeline.py.)
    if args.digests:
        with open(args.digests, "w") as fout:
            for image in image_names_with_digests:
                fout.write(image)


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
        / mdk.util.framework.PIPELINE_CONFIG_BASENAME
    )
    pipeline_config = mdk.config.readYAMLConfig(pipeline_config_filename)
    images_to_build = pipeline_config["images"]
    if image_name_to_build:
        if image_name_to_build not in images_to_build:
            err_msg = f"Image '{image_name_to_build}' not found in pipeline_config.yml"
            logger.error(err_msg)
            raise ValueError(err_msg)
        images_to_build = {image_name_to_build: images_to_build[image_name_to_build]}

    return [
        f"{ar_repo}/{image_config['image']}"
        for image_config in images_to_build.values()
    ]


def buildImages(
    pipeline_dir: pathlib.Path,
    ar_repo: str,
    is_local: bool,
    image_name: str | None = None,
    git_sha: str | None = None,
) -> list[str]:
    f"""Build all docker images by looking in {mdk.util.framework.PIPELINE_CONFIG_BASENAME}.

    Args:
        pipeline_dir (pathlib.Path): The path to the directory containing
            pipeline.py, config files, etc. for the pipeline of interest.
        ar_repo (str): Artifact Registry repository to use for Docker image
            upload.
        is_local (bool): Whether the pipeline is running locally.  If this is
            False, buildImages() will return an empty list.  (We don't want
            digests for local run, because the local run process cannot see
            digests, and using the :latest tag works great for local runs
            anyway.)
        image_name (str | None): The specific image to build. If None, all
            images in the config will be built.
        git_sha (str | None): The Git commit SHA to use as an image tag. If
            provided, this will be added as a tag.

    Returns:
        list[str]: The image with digest, for each image that gets built.
    """
    # Read the .env file.
    dotenv_config = _read_dotenv()

    # Build our images.
    pipeline_config_filename = (
        pipeline_dir.parent.parent
        / "config"
        / mdk.util.framework.PIPELINE_CONFIG_BASENAME
    )
    image_urls = _read_configs_and_build_images(
        pipeline_config_filename, ar_repo, is_local, dotenv_config, image_name, git_sha
    )

    # If this is not a local pipeline, get the image names with digests of the
    #   images we just built.
    # (The reason we don't do this for local runs is because, as of this
    #   writing, KFP's local run process searches for local images using
    #   DockerClient.images.list() and only compares with the name of the image,
    #   without the digest.  If we include the digest, the match will fail, and
    #   KFP's local run process will fall back to look for the image on Artifact
    #   Registry (AR).  We don't want to upload the image to AR and then
    #   immediately download the image from AR every time for local runs, and so
    #   to prevent that we don't specify a digest for local runs, so that the
    #   spec file will just specify the :latest image, so that KFP's local run
    #   process can find the image locally without going out to AR.)
    image_names_with_digests = []
    if not is_local:
        image_names_with_digests = _getImageNamesWithDigest(image_urls)

    return image_names_with_digests


def _read_configs_and_build_images(
    pipeline_config_filename: pathlib.Path,
    artifact_registry_repo: str,
    is_local: bool,
    dotenv_config: dict,
    image_name_to_build: str | None = None,
    git_sha: str | None = None,
):
    f"""Use Docker to locally build images for each of the components listed in
    the components section of {mdk.util.framework.PIPELINE_CONFIG_BASENAME}.

    Note: This requires that the user has run the following gcloud command:

    gcloud auth configure-docker REGION-docker.pkg.dev

        ... where REGION refers to the appropriate GCP region (such as us or
        us-east4).

    Args:
        pipeline_config (dict): Config giving info about the pipeline whose
            images we are building.
        artifact_registry_repo (str): URL of Artfiact Registry repository to
            use for cacheing, etc.  (Should NOT include image name.)
        is_local (bool): Whether this build is being run locally, so that there
            is no reason to push the image after building.
        dotenv_config (dict): Configuration from a .env file.  Used for the
            MDK_HTTPS_PROXY setting.
        image_name_to_build (str | None): The specific image to build. If None,
            all images in the config will be built.
        git_sha (str | None): The Git commit SHA to use as an image tag. If
            provided, this will be added as a tag.
    """
    # Read our config files.
    pipeline_config = mdk.config.readYAMLConfig(pipeline_config_filename)

    # Get our set of images and build dirs.
    build_dir_lookup = _get_build_dir_lookup(pipeline_config)

    # Set the architecture of the Docker build depending on whether we're
    #   running locally or in Vertex AI Pipelines.
    if is_local:
        # On Mac, os.uname().machine is: arm64
        platform = f"linux/{os.uname().machine}"
    else:
        # Vertex AI Pipelines will be using linux/amd64.
        platform = "linux/amd64"

    https_proxy = dotenv_config.get("MDK_HTTPS_PROXY")

    # If we're only supposed to build a single image, modify build_dir_lookup
    #   to only include the one entry.
    if image_name_to_build:
        if image_name_to_build not in build_dir_lookup:
            err_msg = f"Image '{image_name_to_build}' not found in pipeline_config.yml"
            logger.error(err_msg)
            raise ValueError(err_msg)
        build_dir_lookup = {image_name_to_build: build_dir_lookup[image_name_to_build]}

    # For each artifact that we need:
    image_urls = set()
    for image_artifact, build_config_dir in build_dir_lookup.items():
        # Build the artifact.
        image_url = f"{artifact_registry_repo}/{image_artifact}"
        should_push_image = not is_local
        _dockerBuildImage(
            image_url,
            build_config_dir,
            https_proxy,
            platform,
            should_push_image,
            git_sha,
        )
        image_urls.add(image_url)

    return list(image_urls)


def _get_build_dir_lookup(
    pipeline_config: dict,
) -> dict[str, str]:
    """Read our two config files and get a dict describing the directories with
    the relevant Dockerfiles in them.

    Args:
        pipeline_config (dict): Config giving info about the pipeline whose
            images we are building.

    Returns:
        dict[str, str]: A dict indicating which images should be built, and
            where the relevant Dockerfiles are located.
    """
    # For each component in the pipeline:
    build_config_dir_lookup = {}
    for component_name, component_config in pipeline_config["components"].items():
        # Look up the build dir for this component's image:

        image_artifact = component_config["image_artifact"]
        # If we've already looked up the build dir for this artifact, there's no
        #   need to do it again. If container_specs are specified, the user might
        #   be trying to use an off the shelf container that we need not build.
        if image_artifact not in build_config_dir_lookup:
            # Look up the build config dir in the images section:
            build_config_dirs = []
            for image_config in pipeline_config["images"].values():
                if image_config["artifact"] == image_artifact:
                    build_config_dirs.append(image_config["build_config_dir"])

            # Check that we found exactly one matching artifact name in the
            #   images section.
            if len(build_config_dirs) > 1:
                raise RuntimeError(
                    f"Error reading {mdk.util.framework.PIPELINE_CONFIG_BASENAME}:"
                    f" {len(build_config_dirs)} different images in images"
                    f" section have artifact = {image_artifact}"
                )
            # Check for container_specs in case the user provided a full container uri
            # e.g. us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-17.py310:latest
            if len(build_config_dirs) == 0 and "container_specs" in component_config:
                break
            elif len(build_config_dirs) == 0:
                raise RuntimeError(
                    f"Error reading {mdk.util.framework.PIPELINE_CONFIG_BASENAME}: No matching"
                    f" images found in images section with artifact ="
                    f" {image_artifact} (required from component"
                    f" {component_name})"
                )
            # Now that we know we have one unique artifact in the "images"
            #   section corresponding to this component's artifact, persist the
            #   build dir we'll need for building that artifact.
            build_config_dir_lookup[image_artifact] = build_config_dirs[0]

    return build_config_dir_lookup


def _dockerBuildImage(
    image_url: str,
    build_config_dir: str,
    https_proxy: str | None,
    platform: str,
    should_push_image: bool,
    git_sha: str | None = None,
):
    """Run Docker Build to build an image.

    Args:
        image_artifact_url (str): The Artifact Registry URL to which the built
            images sshould be uploaded.
        build_config_dir (str): The directory in which the relevant Dockerfile
            resides.
        https_proxy (str | None): The proxy that uv/pip should used.  Ignored if
            this is None or the empty string.
        platform (str): The platform for which this should be built (e.g.
            linux/arm64 or linux/amd64)
        should_push_image (bool): Whether we should call docker push to push
            the image to Artfiact Regisry after it is built.
        git_sha (str | None): The Git commit SHA to use as an image tag. If
            provided, this will be added as a tag.
    """
    msg = f" Building image: {image_url.split('/')[-1]}"
    logger.info(f"{msg:=^{WIDTH}}")

    # Build the image:
    dockerfile = str(pathlib.Path(build_config_dir) / "Dockerfile")
    cmd = [
        "docker", "build", ".",
        "--tag", image_url,
        "--cache-from", image_url,
        "--platform", platform,
        "--file", dockerfile,
        "--progress=plain",
    ]  # fmt: skip

    # If a git_sha is provided, add it as another tag.
    bare_image_url = image_url.split(":")[0]
    if git_sha:
        sha_tag = f"{bare_image_url}:{git_sha}"
        cmd += ["--tag", sha_tag]

    # If we need a proxy for uv/pip, pass that info to the Dockerfile.
    if https_proxy:
        cmd += [
            "--build-arg",
            f"HTTPS_PROXY={https_proxy}",
        ]

    logger.info(f"\n{' '.join(cmd)}\n")
    logger.info("-" * WIDTH)

    # Do the docker build.
    # (Note: We don't capture stdout beause the output of Docker doesn't go to
    #   stdout, but instead is magically sent directly to the terminal.)
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise RuntimeError(f"{' '.join(cmd[:2])} returned a nonzero exit code: {rc}")

    # Push the image.
    if should_push_image:
        cmd = ["docker", "push", "--all-tags", bare_image_url]

        logger.info("=" * WIDTH)
        logger.info(f"\n{' '.join(cmd)}\n")
        logger.info("-" * WIDTH)

        rc = subprocess.run(cmd).returncode
        if rc != 0:
            raise RuntimeError(
                f"{' '.join(cmd[:2])} returned a nonzero exit code: {rc}"
            )

        logger.info("=" * WIDTH)

    return image_url


def _getImageNamesWithDigest(image_urls: list[str]) -> list[str]:
    """Get the image names with the digests from Docker for each image in
    image_urls.

    Args:
        image_urls (list[str]): List of image names (without digest).

    Returns:
        list[str]: The image with digest, for each image in image_urls.
    """
    image_names_with_digests: list[str] = []
    for image_url in image_urls:
        cmd = [
            "docker", "inspect", f"{image_url}",
            "--format={{.RepoDigests}}",
        ]  # fmt: skip

        logger.info(f"\n{' '.join(cmd)}\n")
        logger.info("-" * WIDTH)

        # Run docker inspect.
        proc = subprocess.run(cmd, capture_output=True)
        if proc.stdout is None:
            err_msg = (
                f"Error running docker inspect. stdout is None. stderr: {proc.stderr}"
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        # For each line of the output, collect the image names with digests.
        names = []
        for bytes_line in proc.stdout.split(b"\n"):
            if not bytes_line:
                continue
            line = bytes_line.decode()
            logger.info(line)

            # Deal with the starting and ending square brackets in the output.
            if not line.startswith("["):
                err_msg = f"Expected output to start with '['. Got: {line}"
                logger.error(err_msg)
                raise RuntimeError(err_msg)
            if not line.endswith("]"):
                err_msg = f"Expected output to end with ']'. Got: {line}"
                logger.error(err_msg)
                raise RuntimeError(err_msg)
            image_name_with_digest = line.lstrip("[").rstrip("]")

            # Save our image name.
            names.append(image_name_with_digest)

        # We ask for the :latest image, so we expect to only get a single image.
        if not len(names) == 1:
            err_msg = f"Expected a single image name. Got: {names}"
            logger.error(err_msg)
            raise RuntimeError(err_msg)
        image_names_with_digests += names

    # Check for empty output.
    if not image_names_with_digests:
        err_msg = "No image names found using docker inspect"
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    return image_names_with_digests


def _read_dotenv() -> dict:
    """Read a .env configuration file, if it exists.  Print information to the
    terminal for transparency.
    """
    dotenv_file = pathlib.Path(".env")
    if dotenv_file.exists():
        dotenv_config = dotenv.dotenv_values(".env")
        logger.info(f".env file found.  Using configuration:\n{dotenv_config}")
    else:
        logger.info(".env file not found.")
        dotenv_config = {}
    return dotenv_config


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
        help="Artifact registry repository to which the images should be uploaded",
    )

    parser.add_argument(
        "--image-name",
        help="The specific image to build. If not provided, all images will be built.",
        metavar="IMAGE_NAME",
        default=None,
    )

    parser.add_argument(
        "--git-sha",
        help="Git commit SHA to use as an image tag.",
        metavar="GIT_SHA",
        default=None,
    )

    parser.add_argument(
        "--local",
        "-l",
        help="Run the pipeline locally, instead of submitting to Vertex AI Pipelines",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--list-only",
        help="Only list the images that would be built; do not build them.",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--digests",
        "-d",
        metavar="FILE",
        help="Write image names with digests to FILE",
        default=None,
    )

    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    main()
