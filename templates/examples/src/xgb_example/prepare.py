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

"""This serves as a template for new projects, so that data scientists can fill
in the below methods for the purpose of getting up and running with MLOps
infrastructure.

This module is for implementing data preprocessing.
"""

import google.cloud.bigquery
import mdk.config
import collections
import logging

logger = logging.getLogger(__name__)

PrepareOutput = collections.namedtuple(
    "PrepareOutput", ["train_uri", "val_uri", "test_uri"]
)

RANDOM_STATE = 0


def prepare(
    general_config_filename: str,
    project_id: str,
    environment: str,
) -> PrepareOutput:
    """Prepares sample data.
    Args:
        general_config_filename (str): Filename of config file with model-
            related configuration info.
        project_id: GCP project ID
        environment: The environment to use (e.g., 'prod', 'stage', 'train').

    Returns:
        PrepareOutput: The train_uri, val_uri and test_uri members give the
            URIs of the BigQuery tables for the train, validation and test data,
            respectively.
    """
    logger.info("Preparing data...")

    # Load and validate our config files.
    general_config = mdk.config.readAndMergeYAMLConfig(
        config_filename=general_config_filename, environment=environment
    )
    training_config = general_config["training"]
    dataset = training_config["dataset"]

    seed = "MyArbitrarySeed"

    # Training dataset, with a random sample of 60% of the data:
    sql = f"""
        create or replace table
            `{project_id}.{dataset}.train`
        as
        select
            samp.* except(id, Class),
            map.ClassIndex
        from
            `{project_id}.{dataset}.samples` samp
        join
            `{project_id}.{dataset}.class_mapping` map on samp.Class=map.Class
        where
            abs(mod(farm_fingerprint(concat(cast(id as string), '{seed}')), 5)) < 3;
    """
    logger.info(sql)
    # Note: This will fail with a permissions error without the project keyword arg.
    client = google.cloud.bigquery.Client(project=project_id)
    client.query_and_wait(sql).to_dataframe()

    # Validation dataset, with a random sample of 20% of the data:
    sql = f"""
        create or replace table
            `{project_id}.{dataset}.val`
        as
        select
            samp.* except(id, Class),
            map.ClassIndex
        from
            `{project_id}.{dataset}.samples` samp
        join
            `{project_id}.{dataset}.class_mapping` map on samp.Class=map.Class
        where
            abs(mod(farm_fingerprint(concat(cast(id as string), '{seed}')), 5)) = 3;
    """
    logger.info(sql)
    client.query_and_wait(sql)

    # Test dataset, with a random sample of the remaining 20% of the data:
    sql = f"""
        create or replace table
            `{project_id}.{dataset}.test`
        as
        select
            samp.* except(id, Class),
            map.ClassIndex
        from
            `{project_id}.{dataset}.samples` samp
        join
            `{project_id}.{dataset}.class_mapping` map on samp.Class=map.Class
        where
            abs(mod(farm_fingerprint(concat(cast(id as string), '{seed}')), 5)) = 4;
    """
    logger.info(sql)
    client.query_and_wait(sql)

    # Return the URIs of the BigQuery tables with our train, validation and test
    #   datasets.
    filenames = PrepareOutput(
        f"bq://{project_id}.{dataset}.train",
        f"bq://{project_id}.{dataset}.val",
        f"bq://{project_id}.{dataset}.test",
    )

    return filenames
