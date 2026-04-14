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


#!/usr/bin/env python

"""This script converts all templates in a given directory from Jinja format
to Backstage format.
"""

import logging
import argparse
import os
import re
import shutil
import sys
import pathlib
from abc import ABC, abstractmethod
from pydantic import BaseModel, DirectoryPath

JINJA_EXT = ".jinja"

IGNORE = [
    ".DS_Store",
    ".copier-answers.yml",
    "copier.yml",
    "{{_copier_conf.answers_file}}.jinja",
]


class PathResolver(ABC):
    """Abstract base class for determining a file's destination path."""

    @abstractmethod
    def resolve(
        self, base_dest_dir: pathlib.Path, relative_path: pathlib.Path
    ) -> pathlib.Path:
        """Given a base destination, returns the final path for a file."""
        pass

    @staticmethod
    @abstractmethod
    def applies_to(relative_path: pathlib.Path) -> bool:
        """Returns True if this resolver should be used for the given path."""
        pass


class OptionalPathResolver(PathResolver):
    """Places files into an 'optional' subdirectory."""

    def resolve(
        self, base_dest_dir: pathlib.Path, relative_path: pathlib.Path
    ) -> pathlib.Path:
        return base_dest_dir / "optional" / relative_path

    @staticmethod
    def applies_to(relative_path: pathlib.Path) -> bool:
        path_str = str(relative_path)
        return (
            "configs/examples" in path_str
            or "src/examples" in path_str
            or "tests/unit_tests" in path_str
            or path_str.startswith("examples")
            or path_str.startswith("scripts")
        )


class DefaultPathResolver(PathResolver):
    """The default resolver that places files in the root of the destination."""

    def resolve(
        self, base_dest_dir: pathlib.Path, relative_path: pathlib.Path
    ) -> pathlib.Path:
        return base_dest_dir / "skeleton" / relative_path

    @staticmethod
    def applies_to(relative_path: pathlib.Path) -> bool:
        return True


class ScriptArgs(BaseModel):
    """Defines the command-line arguments for the script."""

    src_dir: DirectoryPath
    dest_dir: pathlib.Path
    force: bool = False
    copy_mdk: bool = True
    verbose: bool = False


class JinjaToBackstageConverter:
    """Converts a directory of Jinja templates to Backstage format."""

    def __init__(self, settings: ScriptArgs):
        self.settings = settings
        self.project_root = pathlib.Path(__file__).parent.parent
        self.path_resolvers = [OptionalPathResolver(), DefaultPathResolver()]

    def run(self):
        """Executes the conversion process."""
        self._process_template_directory()
        if self.settings.copy_mdk:
            self._copy_auxiliary_files()

    def _process_template_directory(self):
        """Walks the source directory and processes each file."""
        logging.info("Processing templates from '%s'...", self.settings.src_dir)
        for dirpath, _, filenames in os.walk(self.settings.src_dir):
            for filename in filenames:
                if filename in IGNORE:
                    continue

                src = pathlib.Path(dirpath) / filename
                dest = self._get_destination_path(src)
                if dest.exists() and not self.settings.force:
                    self._print_status("SKIP (exists)", dest)
                    continue

                self._migrate("create", src, dest, force=self.settings.force)

    def _copy_auxiliary_files(self):
        """Copies auxiliary source files into the skeleton."""
        logging.info("Copying MDK and other auxiliary files...")
        skeleton_dir = self.settings.dest_dir / "skeleton"

        # Copy all .py files from the src/mdk directory.
        mdk_src_dir = self.project_root / "src" / "mdk"
        for src_file in mdk_src_dir.rglob("*.py"):
            relative_path = src_file.relative_to(mdk_src_dir)
            dest_file = skeleton_dir / "src" / "mdk" / relative_path
            self._migrate_with_force_check(src_file, dest_file)

        # Copy src/model/registry directory
        src_registry = self.project_root / "src" / "model" / "registry"
        dest_registry = skeleton_dir / "src" / "model" / "registry"
        self._migrate_with_force_check(src_registry, dest_registry)

    def _get_destination_path(self, src_path: pathlib.Path) -> pathlib.Path:
        """Determines the destination path for a source file using resolvers."""
        relative_path = src_path.relative_to(self.settings.src_dir)
        for resolver in self.path_resolvers:
            if resolver.applies_to(relative_path):
                return resolver.resolve(self.settings.dest_dir, relative_path)
        raise RuntimeError(f"Could not determine destination for {src_path}")

    def _migrate_with_force_check(self, src: pathlib.Path, dest: pathlib.Path):
        """Wrapper for _migrate that handles the force flag check."""
        if dest.exists() and not self.settings.force:
            self._print_status("SKIP (exists)", dest)
            return
        self._migrate("create", src, dest, force=self.settings.force)

    def _migrate(
        self, msg: str, src: pathlib.Path, dest: pathlib.Path, force: bool = False
    ):
        """Migrates a file or directory from a source to a destination."""
        if dest.exists() and force:
            msg = "replace"
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()

        dest.parent.mkdir(parents=True, exist_ok=True)

        if str(src).endswith(JINJA_EXT):
            dest = pathlib.Path(str(dest).removesuffix(JINJA_EXT) + ".njk")
            self._print_status(msg, dest)
            self._convert_template(src, dest)
        elif src.is_file():
            self._print_status(msg, dest)
            shutil.copyfile(src, dest)
        elif src.is_dir():
            self._print_status(msg, dest)
            shutil.copytree(src, dest)

    def _convert_template(self, src: pathlib.Path, dest: pathlib.Path):
        """Reads a Jinja template and converts it to Backstage format."""
        with open(src, "r") as fin:
            content = fin.read()

        # Replace {{ value }} with ${{values.value}}, removing extra whitespace.
        content = re.sub(r"\{\{\s*(.*?)\s*\}\}", r"${{values.\1}}", content)

        with open(dest, "w") as fout:
            fout.write(content)

    def _print_status(self, msg: str, dest: pathlib.Path | str):
        """Prints a formatted status message to the terminal."""
        logging.info("%s  %s", msg.rjust(14), dest)


def parse_command_line(argv):
    """Get command line arguments and options."""

    parser = argparse.ArgumentParser(
        prog="mdk",
        description=__doc__,
    )
    parser.add_argument(
        "src_dir",
        metavar="INPUT_JINJA_TEMPLATE_DIR",
        help="Input directory containing Jinja templates",
    )
    parser.add_argument(
        "dest_dir",
        metavar="OUTPUT_BACKSTAGE_SKELETON_DIR",
        help=(
            "Output directory containing Backstage templates.  If this"
            " directory does not exist, it will be created."
        ),
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help=(
            "If the target file in the output directory already exists, overwrite it."
        ),
    )
    parser.add_argument(
        "--no-copy-mdk",
        dest="copy_mdk",
        action="store_false",
        help=("Do not copy the MDK source into the backstage directory."),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help=("Enable verbose output for debugging."),
    )

    args = parser.parse_args(argv[1:])
    return ScriptArgs(**vars(args))


def main():
    """This is invoked when the script is called from the command line."""
    settings = parse_command_line(sys.argv)

    log_level = logging.DEBUG if settings.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s", stream=sys.stdout)

    converter = JinjaToBackstageConverter(settings)
    converter.run()


if __name__ == "__main__":
    main()
