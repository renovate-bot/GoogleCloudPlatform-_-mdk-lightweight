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

import sys
from ._util import get_config_value_by_key


def main():
    """
    Command-line interface to extract a single value from a YAML configuration file.
    Usage: python -m mdk.config.getconfigvalue <config_filename> <key>
    """
    # Check for exactly 3 arguments (script name, filename, key)
    if len(sys.argv) != 3:
        sys.stderr.write(
            "Usage: python -m mdk.config.getconfigvalue <config_filename> <key>\n"
        )
        # Print an empty string to stdout for reliable bash capture
        print("", file=sys.stdout)
        sys.exit(1)

    config_filename = sys.argv[1]
    key = sys.argv[2]

    # Call the core logic and print the result to stdout for bash capture
    result = get_config_value_by_key(config_filename, key)

    # If result is empty, check for file/key errors before printing
    if not result:
        # Suppress error messages from the core function and handle them here,
        # or rely on the core function's simple return of "".
        # Since the core function handles the exception and returns "", we can
        # rely on that for the bash script to capture an empty value.
        pass

    print(result)


if __name__ == "__main__":
    main()
