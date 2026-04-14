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

"""Manage various parts of the MLOps pipeline, including generating a skaffolding of
template files and running a pipeline job.
"""

import mdk.cli

# Ensure you import other modules if 'init', 'list', 'run' are in different files
import mdk.cli.init
import mdk.cli.list
import mdk.cli.run

import argparse
import logging
import sys

PIPELINE_CONFIG_BASENAME = "pipeline_config.yml"

logger = logging.getLogger(__name__)


def main():
    """This is invoked when the script is called from the command line."""

    logging.basicConfig(format="", level=logging.INFO)

    # Get our command line arguments.
    clargs = parseCommandLine(sys.argv)

    # Convert to dict for legacy support
    args_dict = vars(clargs)

    # ---------------------------------------------------------
    # CORRECTED DISPATCH LOGIC
    # ---------------------------------------------------------
    # We simply check if a function was assigned to the parser.
    # This works for run, list, and init uniformly.
    if "func" in args_dict:
        func = args_dict.pop("func")
        func(**args_dict)
    else:
        # No subcommand provided
        print("Error: No command specified (use: init, list, run)")
        return 1

    return 0


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parseCommandLine(argv):

    """Get command line arguments and options."""

    # Top-level parser:
    parser = argparse.ArgumentParser(
        prog="mdk",
        description=__doc__,
    )
    subparsers = parser.add_subparsers(required=True)

    # mdk init:
    parser_init = subparsers.add_parser(
        "init",
        help="Generate a skeleton of template files to implement various ML Ops processes.",
    )
    parser_init.set_defaults(func=mdk.cli.init.init)
    parser_init.add_argument(
        "--skip-answered",
        help="Skip asking questions that have already been answered in the .copier-answers.yaml file",
        action="store_true",
        default=False,
    )
    parser_init.add_argument(
        "--lite",
        type=str_to_bool,
        nargs='?',
        const=True,
        default=True,
        help="Enable lite mode (dev environment only, disabled extended registry). Default is True. Pass False to disable.",
    )
    parser_init.add_argument(
        "--overwrite",
        help="Overwrite existing files without asking.",
        action="store_true",
        default=False,
    )
    parser_init.add_argument(
        "--verbose",
        help="Controls the verbosity.",
        action="store_true",
        default=False,
    )

    # 3. mdk list:
    parser_list = subparsers.add_parser(
        "list",
        help="List the pipelines that are available to run.",
    )
    parser_list.set_defaults(func=mdk.cli.list.list)

    # 4. mdk run:
    parser_run = subparsers.add_parser(
        "run",
        help="Run a Vertex Pipeline job, using the mdk_run.yml file",
    )
    parser_run.set_defaults(func=mdk.cli.run.run)

    parser_run.add_argument(
        "pipeline_name",
        help=f"The name of the pipeline to run (from {PIPELINE_CONFIG_BASENAME}).",
        nargs="?",
        default=None,
    )
    parser_run.add_argument(
        "--environment",
        "-e",
        choices=("dev", "stage", "prod"),
        default="dev",
        help="Suffix to use on the end of the GCP config.yaml filename.",
    )
    parser_run.add_argument(
        "--local",
        "-l",
        help="Run the pipeline locally, instead of submitting to Vertex AI Pipelines",
        action="store_true",
        default=False,
    )

    parser_run.add_argument(
        "--lite",
        help="Enable lite mode (bypass expanded model registry)",
        action="store_true",
        default=False,
    )

    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    sys.exit(main())
