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

"""Generate a skeleton of template files to implement various ML Ops processes."""

import copier
import pathlib
import shutil
import os
import logging

logger = logging.getLogger(__name__)


def init(
    *,
    skip_answered: bool,
    overwrite: bool,
    verbose: bool,
    lite: bool = False,
):
    """Generate a skeleton of template files to implement various ML Ops processes.

    Args:
        skip_answered (bool): Skip asking questions that have already been
            answered in the .copier-answers.yaml file
        overwrite (bool): If a target file already exists, then instead of
            asking whether it should be overwritten, just overwrite it without
            asking.
        verbose (bool): Controls the verbosity. If True, displays all logs
            from the template generation process.
        lite (bool): Enable lite mode (dev environment only, disabled extended registry).
    """
    # Find our templates:
    templates_dir = pathlib.Path(__file__).parent.parent.parent.parent / "templates"
    if not templates_dir.is_dir():
        # Fallback to searching upwards for 'templates' directory
        current = pathlib.Path(__file__).parent
        while current != current.parent:
            if (current / "templates").is_dir():
                templates_dir = current / "templates"
                break
            current = current.parent

    # Run Copier.
    dest_dir = "."
    exclude_list = [".github/workflows"] if lite else []
    copier.run_copy(
        src_path=str(templates_dir),
        dst_path=dest_dir,
        defaults=skip_answered,
        overwrite=overwrite,
        quiet=not verbose,
        data={"lite": lite},
        exclude=exclude_list,
    )

    # Copy the MDK library (without the command line) into the target project.
    #   This is a temporary measure; eventually, the MDK will be distributed
    #   via a dependency in pyproject.toml, but that requires the MDK to be
    #   available on Nexus.
    copy_mdk_lib()

    logger.info("Initialization complete.")


def copy_mdk_lib():
    """This function copies the MDK library into the target project.

    This is a temporary measure; eventually, the MDK will be distributed via a
    dependency in pyproject.toml.  The reason we don't do that yet is because,
    for pip to find the MDK, it would require the MDK to be available on Nexus.
    """
    logger.info("Copying MDK...")

    src_mdk_dir = pathlib.Path(__file__).parent.parent
    dest_mdk_dir = pathlib.Path("src/mdk")
    dest_mdk_dir.mkdir(exist_ok=True)

    exclude = ["__pycache__"]

    for dirpath, _, filenames in os.walk(src_mdk_dir):
        for filename in filenames:
            src = pathlib.Path(dirpath) / filename
            dest = dest_mdk_dir / src.relative_to(src_mdk_dir)

            # Skip this file if any part of the directory has a name that is in
            #   our exclude list.
            if any(part in exclude for part in dirpath.split("/")):
                continue

            # Create the destination directory if it does not exist.
            if not dest.parent.is_dir():
                logger.info(f"Creating: {dest.parent}")
                dest.parent.mkdir(parents=True)

            # Copy the file.
            try:
                shutil.copy(src, dest)
            except shutil.SameFileError as e:
                e.add_note(
                    "This error can happen if the source mdk and the"
                    " destination mdk are the same (i.e. the source mdk is"
                    " being told to install itself on top of itself). If you"
                    " would like to re-run mdk init. you should not be trying"
                    " to overwrite the MDK with itself.  Use a separate"
                    " destination directory."
                )
                raise
