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
MDK CI/CD Docker Build Utility (ci_build_utils.py)

This script is a simple CLI wrapper for building a single Docker image.
It is designed to be called from a GHA matrix build strategy.
It reuses the core build logic from 'mdk.pipeline_tools.build_images'.
"""

import argparse
import logging
import sys
from mdk.pipeline_tools.build_images import _dockerBuildImage, _getImageNamesWithDigest

logger = logging.getLogger(__name__)

# --- CLI Wrapper ---


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="MDK CI/CD Single Image Builder.")

    parser.add_argument(
        "--image-url",
        type=str,
        required=True,
        help="Full URL of the image to build (e.g., 'ar-repo/my-image').",
    )
    parser.add_argument(
        "--build-config-dir",
        type=str,
        required=True,
        help="Path to the directory containing the Dockerfile (e.g., 'model_products/xgb/components/training').",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="linux/amd64",
        help="Target platform (e.g., 'linux/amd64').",
    )
    parser.add_argument(
        "--git-sha", type=str, help="Git commit SHA to use as an additional tag."
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the image to the registry after building.",
    )
    parser.add_argument(
        "--digest-file",
        type=str,
        required=True,
        help="File path to write the final image URI (with digest) to.",
    )
    args = parser.parse_args()

    https_proxy = None

    logger.info("--- Building Image ---")
    logger.info(f"Image URL: {args.image_url}")
    logger.info(f"Dockerfile Dir: {args.build_config_dir}")
    logger.info(f"Platform: {args.platform}")
    logger.info(f"Push: {args.push}")
    logger.info(f"Git SHA Tag: {args.git_sha or 'None'}")
    logger.info(f"HTTPS Proxy: {https_proxy or 'None'}")

    try:
        # Build docker image
        _dockerBuildImage(
            image_url=args.image_url,
            build_config_dir=args.build_config_dir,
            https_proxy=None,
            platform=args.platform,
            should_push_image=args.push,
            git_sha=args.git_sha,
        )
        logger.info(f"Successfully built and pushed {args.image_url}")

        image_name_with_digest = _getImageNamesWithDigest([args.image_url])

        with open(args.digest_file, "w") as fout:
            for image in image_name_with_digest:
                fout.write(image)

    except Exception as e:
        logger.error(f"Failed to build image {args.image_url}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
