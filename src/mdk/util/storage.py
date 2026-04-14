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

"""This module provides general-purpose utility functions related to Google
Cloud Storage.
"""

import google.cloud.storage
import pathlib
import shutil
import os
import logging

logger = logging.getLogger(__name__)


def upload(
    src: str,
    dest: str,
    *,
    mkdir: bool = False,
):
    """Copy a file to either GCS or the local filesystem.  It is convenient for
    this function to accept either a local path or a remote GCS URL because,
    for remote pipelines, Kubeflow will provide a GCS URL, and for local
    pipelines, Kubeflow will provide a local file path.

    Note: This assumes that aiplaform.init() has already been called to set the
    GCP project.

    Args:
        src (str): Path to the file on the local filesystem that needs to be
            copied to *dest*.
        dest (str): If *dest* is a GCS URL (starting with "gs://"), then *src*
            will be uploaded to *dest*.  If *dest* is the path to a location on
            the local filesystem, *src* will be copied to *dest*.
        mkdir (bool): (Optional) If *mkdir* is True, then if *dest* is a local
            path, then any non-existing parent directories will be created for
            *dest*.  If *dest* is a GCS URL, the *mkdir* flag is ignored (since
            GCS will automatically create parent folders as necessary anyway).
    """

    if dest.startswith("gs://"):
        # with open("config/gcp_config.yml", "r") as fin:
        #     gcp_config = yaml.safe_load(fin)
        # project_id = gcp_config["project_id"]

        # client = google.cloud.storage.Client(project=project_id)
        client = google.cloud.storage.Client()
        upload_to_gcs_uri(client, src, dest)
    else:
        if mkdir:
            # If mkdir == True, make any parent directories for the destination.
            pathlib.Path(dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dest)


def download(
    src: str,
    dest: str,
):
    """Copy a file to either GCS or the local filesystem.  It is convenient for
    this function to accept either a local path or a remote GCS URL because,
    for remote pipelines, Kubeflow will provide a GCS URL, and for local
    pipelines, Kubeflow will provide a local file path.

    Note: This assumes that aiplaform.init() has already been called to set the
    GCP project.

    Args:
        src (str): If *src* is a GCS URL (starting with "gs://"), then *src*
            will be downloaded to *dest*.  If *src* is the path to a location on
            the local filesystem, *src* will be copied to *dest*.
        dest (str): Destination path to the local fileystem to which *src* will
            be copied.
    """
    if src.startswith("gs://"):
        # with open("config/gcp_config.yml", "r") as fin:
        #     gcp_config = yaml.safe_load(fin)
        # project_id = gcp_config["project_id"]

        # client = google.cloud.storage.Client(project=project_id)
        client = google.cloud.storage.Client()
        download_from_gcs_uri(client, src, dest)
    else:
        shutil.copy(src, dest)


def upload_to_gcs_uri(
    client: google.cloud.storage.Client,
    local_filename: str,
    gcs_uri: str,
):
    """Upload a file from local storage to GCS, given a GCS URI.

    Args:
        client: Storage Client used to upload files
        local_filename: Filename of the file on the local filesystem which will
            be uploaded to gcs_uri.
        gcs_uri: gs_file_path: Full destination GCS file location (including gs://)
    """
    logger.info(f"Uploading to: {gcs_uri}")

    bucket_name, remote_path = _parse_gcs_uri(gcs_uri)

    # Do the upload.
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(remote_path)
    blob.upload_from_filename(local_filename)


def download_from_gcs_uri(
    client: google.cloud.storage.Client,
    gcs_uri: str,
    local_filename: str,
):
    """Download a file from local storage to GCS, given a GCS URI.

    Args:
        client: Storage Client used to download files
        gcs_uri: gs_file_path: Full source GCS file location (including gs://)
        local_filename: Destination path on the local fileystem to which the
            file at gcs_uri will be downloaded.
    """
    logger.info(f"Downloading from: {gcs_uri}")

    bucket_name, remote_path = _parse_gcs_uri(gcs_uri)

    # Do the download.
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(remote_path)
    blob.download_to_filename(local_filename)


def _parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    """Decompose a GCS URI into a bucket and folder.  We verify that the GCS URI
    begins with the gs:// scheme.

    Args:
      gcs_uri (str): URI to be decomposed

    Returns:
      tuple[str, str]: First element is the bucket name, second element is the
          remote path.
    """
    # We strip off the gs://, but we verify its presence first just as a sanity
    #   check.  (Secifying a URI as bucket_name/foo/bar, without the gs://
    #   scheme, is a weird construction and so we'll assume it's probably a
    #   mistake.)
    if not gcs_uri.startswith("gs://"):
        raise RuntimeError(f"URI {gcs_uri}: gcs_uri should start with gs://")
    gcs_uri = gcs_uri.removeprefix("gs://")
    parts = gcs_uri.split("/")
    bucket_name = parts[0]
    path = "/".join(parts[1:])

    return bucket_name, path


def get_parent_path_intelligent(uri: str) -> str:
    """
    Returns the parent path if the URI appears to be a file,
    otherwise returns the URI normalized (no trailing slash) if
    it appears to be a directory.
    """
    if not uri:
        return uri

    # Determine if it's likely a file by checking for an extension
    # os.path.splitext works on basename, so we need basename first.
    basename = os.path.basename(uri)
    _, ext = os.path.splitext(basename)

    if ext:  # Has an extension, so it's a file
        return os.path.dirname(uri)
    else:  # No extension, treat as a directory
        # Normalize by removing trailing slash if present, for consistency
        if uri.endswith("/") and uri != "gs://":  # Check if it's not the GCS root
            return uri.rstrip("/")
        return uri
