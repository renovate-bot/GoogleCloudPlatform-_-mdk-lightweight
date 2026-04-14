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

import os

def check_files(root_dir):
    missing_license = []
    # Extensions to check
    extensions = ('.py', '.sh', '.yml.jinja', '.tf')
    # Directories to skip
    skip_dirs = {'.git', '.github', '__pycache__', '.pytest_cache', '.venv', 'venv', 'env', '.gemini', 'tmp'}
    
    for root, dirs, files in os.walk(root_dir):
        # Filter directories in place
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            file_path = os.path.join(root, file)
            # Skip the script itself if it's in the root
            if file == "check_license.py" or file_path.endswith("check_license.py"):
                continue
                
            if file.endswith(extensions):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        header = ''
                        for _ in range(20):
                            line = f.readline()
                            if not line:
                                break
                            header += line
                        
                        if "Copyright 2026 Google LLC." not in header:
                            missing_license.append(file_path)
                except Exception as e:
                    pass
    return missing_license

if __name__ == "__main__":
    missing = check_files('.')
    for path in sorted(missing):
        print(path)
