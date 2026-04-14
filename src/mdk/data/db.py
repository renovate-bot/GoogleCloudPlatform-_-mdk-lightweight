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

"""This module provides database-related convenience functions."""

import bigframes.pandas as bpd
import pandas as pd
from vertexai.resources.preview.feature_store import offline_store, FeatureGroup
import logging

logger = logging.getLogger(__name__)


def getDataframeFromBigQuery(
    client,
    uri,
):
    """Given a URI beginning with bq://, select all rows from this table and
    return it as a dataframe.

    Args:
        client (google.cloud.bigquery.Client):
        uri (str):

    Returns:
        pandas.DataFrame:
    """
    SCHEME = "bq://"
    if not uri.startswith(SCHEME):
        raise ValueError(f"Expected uri to begin with {SCHEME}")
    table = uri.removeprefix(SCHEME)

    sql = f"select * from `{table}`"
    logger.info(sql)
    df = client.query_and_wait(sql).to_dataframe()

    return df


def getDataFrameFromFeatureStore(
    feature_group_id: str,
    read_instances_uri: str,
    project_id: str,
    region: str,
) -> pd.DataFrame:
    """
    Fetches feature data from Vertex AI Feature Store using a Point-in-Time lookup.

    This function joins features from a specified feature group to a "read instances"
    table at specific timestamps, preventing data leakage.

    Args:
        feature_group_id: The ID of the Vertex AI Feature Group to pull features from.
        read_instances_uri: The BigQuery URI (bq://...) to the "read instances" table.
                            This table must contain entity IDs and a 'feature_timestamp' column.
        project_id: The GCP project ID.
        region: The location/region to run the BigQuery job (e.g., "us", "us-east4").

    Returns:
        A pandas DataFrame containing the joined historical features.
    """
    logger.info(
        f"Starting historical feature fetch from Feature Group: {feature_group_id}"
    )

    # Configure bigframes to use the correct project and location for the query job.
    bpd.options.bigquery.project = project_id
    bpd.options.bigquery.location = region

    # Step 1: Load the "read instances" table.
    logger.info(f"Loading entity lookup data from: {read_instances_uri}")
    table_id = read_instances_uri.replace("bq://", "")
    entity_df = bpd.read_gbq(table_id)

    # The function requires a timestamp column of datetime dtype for the point-in-time join.
    entity_df["feature_timestamp"] = bpd.to_datetime(entity_df["feature_timestamp"])

    # Step 2: Dynamically build the list of all features from the Feature Group.
    fg = FeatureGroup(feature_group_id)
    features_to_fetch = [f.name for f in fg.list_features()]

    # Format each feature name into the required fully-qualified format:
    feature_string_list = [
        f"{project_id}.{feature_group_id}.{feature_name}"
        for feature_name in features_to_fetch
    ]
    logger.info(
        f"Fetching {len(feature_string_list)} features from project '{project_id}'..."
    )

    # Step 3: Execute the point-in-time lookup.
    training_bdf = offline_store.fetch_historical_feature_values(
        entity_df=entity_df, features=feature_string_list, location=region
    )

    # Step 4: Convert the bigframes DataFrame to an in-memory pandas DataFrame.
    logger.info("Converting result to pandas DataFrame...")
    df = training_bdf.to_pandas()

    logger.info("Successfully fetched historical features.")
    return df
