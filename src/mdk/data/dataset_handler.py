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

"""This module provides the DatasetHandler class."""

import mdk.util.storage
import logging

logger = logging.getLogger(__name__)


class DatasetHandler:
    """This class encapsulates a dataset in such a way that it is easy to use
    with Vertex AI Pipelines.

    Specifically: Vertex AI Pipelines uses URIs as its abstraction for
    datasets, and so this class handles the process of constructing URIs and, if
    necessary, copying flat files to GCS where they can be exposed via a URI.
    """

    def __init__(
        self,
        uri: str,
    ):
        """Constructor.

        Args:
            uri (str): URI from Kubeflow, denoting a temporary storage location
                that is appropriate for storing datasets, if the user so
                desires.
        """
        self._uri = uri

    def set_bigquery_table(
        self,
        project_id: str,
        dataset: str,
        table: str,
    ):
        """Configure the DatasetHandler to return a URI pointing to a dataset
        in the form of a BigQuery table.

        Args:
            project_id (str): GCP Project ID in which this table resides.
            dataset (str): The name of the BigQuery dataset in which this table
                resides.
            table (str): The name of the table with the data of interest.
        """
        self._uri = f"bq://{project_id}.{dataset}.{table}"

    def set_local_file(
        self,
        filename: str,
    ):
        """Configure the DatasetHandler to provide a URI pointing to a dataset
        in the form of a BigQuery table.  This will have the effect of copying
        the file at the path *filename* to the temporary GCS location provided
        to the DatasetHandler's constructor.

        Args:
            filename (str): Local file that will be copied to the uri specified
                during this object's construction.
        """
        mdk.util.storage.upload(filename, self._uri)

    @property
    def uri(self) -> str:
        """The URI of the dataset of interest, whether that is a BigQuery table
        or a file.
        """
        return self._uri
