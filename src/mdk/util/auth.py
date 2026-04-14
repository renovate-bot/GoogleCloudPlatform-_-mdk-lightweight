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

import google.auth.credentials
import google.auth.transport.requests
import google.oauth2.id_token


def generate_gcp_jwt(
    audience: str,
    credentials: google.auth.credentials.Credentials | None = None,
) -> str:
    """Generates a GCP JWT using a predefined credential.
    Args:
        audience: (str) The URL of the service.
        credentials: (google.auth.credentials.Credentials) Credentials to use
            to generate JWT.  If None, a JWT will use Application Default
            Credentials.

    Returns:
        str: GCP JWT associated with the credentials.
    """

    # If we didn't get credentials, use application default credentials.
    if not credentials:
        credentials, _ = google.auth.default()

    session = google.auth.transport.requests.AuthorizedSession(credentials)
    request = google.auth.transport.requests.Request(session)

    return google.oauth2.id_token.fetch_id_token(request, audience)
