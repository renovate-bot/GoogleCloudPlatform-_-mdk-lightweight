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

"""
A script to automate the creation and population of Vertex AI Feature Groups.

This script streamlines the process of setting up a Vertex AI Feature Store
by connecting to an existing BigQuery table and registering its columns as features.
It handles both the initial creation of the Feature Group and the subsequent
registration of individual features, skipping those that already exist.

Prerequisites:
1. Install required libraries:
   pip install google-cloud-aiplatform

2. Authenticate with Google Cloud:
   gcloud auth application-default login
   (Or ensure your environment has a service account with 'Vertex AI Administrator'
    and 'BigQuery Data Viewer' roles).

3. Ensure the Vertex AI Service Agent has 'BigQuery Data Viewer' permissions
   on the source BigQuery project if it is different from the Vertex AI project.
"""

import logging
from typing import List, Optional
from google.cloud import aiplatform
from google.api_core import exceptions
from vertexai.resources.preview import feature_store

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def initialize_vertex(
    project_id: str, location: str, service_account: Optional[str] = None
):
    """Initializes the Vertex AI SDK."""
    aiplatform.init(
        project=project_id, location=location, service_account=service_account
    )


def get_or_create_feature_group(
    feature_group_id: str,
    bq_table_uri: str,
    entity_id_columns: List[str],
    description: str = "",
) -> feature_store.FeatureGroup:
    """
    Retrieves an existing Feature Group or creates it if it does not exist.
    """
    try:
        # Attempt to retrieve existing group
        fg = feature_store.FeatureGroup(name=feature_group_id)
        logging.info(f"Found existing Feature Group: {fg.name}")
        return fg
    except exceptions.NotFound:
        logging.info(
            f"Feature Group '{feature_group_id}' not found. Creating new one..."
        )
        try:
            fg = feature_store.FeatureGroup.create(
                name=feature_group_id,
                description=description,
                source=feature_store.utils.FeatureGroupBigQuerySource(
                    uri=bq_table_uri, entity_id_columns=entity_id_columns
                ),
            )
            logging.info(f"Successfully created Feature Group: {fg.name}")
            return fg
        except Exception as e:
            logging.error(f"Failed to create Feature Group: {e}")
            raise


def register_features(
    feature_group: feature_store.FeatureGroup, feature_ids: List[str]
):
    """
    Loops through a list of feature IDs and registers them to the Feature Group.
    """
    logging.info(
        f"Starting registration of {len(feature_ids)} features to '{feature_group.name}'..."
    )

    for feature_id in feature_ids:
        # Basic validation to ensure feature ID is valid for Vertex AI

        try:
            feature_group.create_feature(name=feature_id)
            logging.info(f" - OK: registered '{feature_id}'")
        except exceptions.AlreadyExists:
            logging.info(f" - SKIP: '{feature_id}' already exists.")
        except Exception as e:
            logging.error(f" - FAIL: could not register '{feature_id}': {e}")


# =========================================
# Execution Block
# =========================================
if __name__ == "__main__":
    # 1. Configuration
    # TODO: Replace these with your actual project and service account details.
    PROJECT_ID = ""  # e.g. "my-sample-project"
    LOCATION = "us-east4"
    SERVICE_ACCOUNT = (
        ""  # e.g. "pipelines-sa@my-sample-project.iam.gserviceaccount.com"
    )

    # Define source for feature group
    FG_ID = "dry_beans_fg"
    # TODO: Replace with your actual BigQuery source URI
    BQ_SOURCE_URI = "bq://shared-project-480017.ml_data_dev.dry_beans"
    ENTITY_ID_COLS = ["entity_id"]

    # 2. Define your features
    FEATURE_LIST = [
        "Area",
        "Perimeter",
        "MajorAxisLength",
        "MinorAxisLength",
        "AspectRation",
        "Eccentricity",
        "ConvexArea",
        "EquivDiameter",
        "Extent",
        "Solidity",
        "roundness",
        "Compactness",
        "ShapeFactor1",
        "ShapeFactor2",
        "ShapeFactor3",
        "ShapeFactor4",
        "Class",
    ]

    # 3. Run Workflow
    initialize_vertex(PROJECT_ID, LOCATION, SERVICE_ACCOUNT)

    my_fg = get_or_create_feature_group(
        feature_group_id=FG_ID,
        bq_table_uri=BQ_SOURCE_URI,
        entity_id_columns=ENTITY_ID_COLS,
        description="Feature Group for Dry Beans",
    )

    register_features(feature_group=my_fg, feature_ids=FEATURE_LIST)
