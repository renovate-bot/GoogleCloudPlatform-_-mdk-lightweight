#!/bin/bash
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


set -x

environment=dev
PROJECT_ID=""
DATASET_ID="ml_data_dev"
DATA_BUCKET="$PROJECT_ID-data"

gcp_conf_yaml=examples/model_products/xgb_example/state/$environment.yml
csv=examples/src/xgb_example/data/Dry_Beans_Dataset.csv
csv_with_id="$(dirname $csv)/numbered_$(basename $csv)"

# Feature and Read Instances Table Names
DRY_BEANS_TABLE="dry_beans"
READ_INSTANCES_TABLE="dry_beans_read_instances"
TEMP_TABLE="dry_beans_temp_data" # New temporary table for initial load

# Read config for region, or set a default if not found
region="us"

# In the below, we assume there is no scheme, so we strip it off here, if it exists.
data_bucket=${DATA_BUCKET#gs://}

echo "Checking whether dataset exists: ${PROJECT_ID}:${DATASET_ID}"
bq --location $region ls --dataset_id ${PROJECT_ID}:${DATASET_ID} \
    && dataset_exists=true || dataset_exists=false

# Create a dataset, if it does not exist.
if [ $dataset_exists == true ] ; then
    echo "${PROJECT_ID}:${DATASET_ID} exists => skipping"
else
    echo "${PROJECT_ID}:${DATASET_ID} does not exist => creating"
    bq \
        --location=$region \
        mk \
        --dataset \
        ${PROJECT_ID}:${DATASET_ID} \
        || exit $?
fi

# Add a row number to the CSV.
echo "Adding 'entity_id' to CSV: $csv_with_id"
# Rename the injected column to 'entity_id' for Feature Store compatibility.
awk -F',' 'BEGIN {OFS=FS} NR==1 {print "entity_id,"$0; next} {print NR-1","$0}' $csv > $csv_with_id

# Upload our CSV.
echo "Uploading to GCS: gs://${DATA_BUCKET}/$(basename $csv_with_id)"
gcloud storage cp "$csv_with_id" gs://${DATA_BUCKET}/ || exit $?


## --- 1. Load Data into a temporary table ---

echo "(Re-) creating temporary load table: ${PROJECT_ID}:${DATASET_ID}.${TEMP_TABLE}"
# Remove the old main table and create a temporary table with the CSV's raw schema
bq rm --force --table "${PROJECT_ID}:${DATASET_ID}.${DRY_BEANS_TABLE}" || exit $?
bq rm --force --table "${PROJECT_ID}:${DATASET_ID}.${TEMP_TABLE}" || exit $?
bq mk --table "${PROJECT_ID}:${DATASET_ID}.${TEMP_TABLE}" \
entity_id:integer,\
Area:integer,\
Perimeter:float,\
MajorAxisLength:float,\
MinorAxisLength:float,\
AspectRation:float,\
Eccentricity:float,\
ConvexArea:float,\
EquivDiameter:float,\
Extent:float,\
Solidity:float,\
roundness:float,\
Compactness:float,\
ShapeFactor1:float,\
ShapeFactor2:float,\
ShapeFactor3:float,\
ShapeFactor4:float,\
Class:string \
    || exit $?

echo "Loading data into temporary table using bq load..."
bq \
    --location=$region \
    load \
    --source_format=CSV \
    --skip_leading_rows=1 \
    --replace \
    "${PROJECT_ID}:${DATASET_ID}.${TEMP_TABLE}" \
    "gs://${DATA_BUCKET}/$(basename $csv_with_id)" \
    || exit $?
echo "Temporary table load complete."

## --- 2. Create Final Feature Table with 'feature_timestamp' ---

echo "Creating final feature table: ${PROJECT_ID}:${DATASET_ID}.${DRY_BEANS_TABLE} with feature_timestamp."

read -r -d "" create_final_query << EOF
CREATE OR REPLACE TABLE \`${PROJECT_ID}.${DATASET_ID}.${DRY_BEANS_TABLE}\` AS
SELECT
    entity_id,
    -- Add the required feature_timestamp column using current timestamp
    CURRENT_TIMESTAMP() as feature_timestamp,
    Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea,
    EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1,
    ShapeFactor2, ShapeFactor3, ShapeFactor4, Class
FROM
    \`${PROJECT_ID}.${DATASET_ID}.${TEMP_TABLE}\`
EOF

bq query --use_legacy_sql=false "$create_final_query" || exit $?
echo "Feature table load complete: bq://${PROJECT_ID}.${DATASET_ID}.${DRY_BEANS_TABLE}"


## --- 3. Create the dry_beans_read_instances Table ---

echo "Creating read instances table: ${PROJECT_ID}:${DATASET_ID}.${READ_INSTANCES_TABLE}"

read -r -d "" read_instances_query << EOF
CREATE OR REPLACE TABLE \`${PROJECT_ID}.${DATASET_ID}.${READ_INSTANCES_TABLE}\` AS
SELECT
  t.entity_id,
  -- Column named 'request_time' for batch feature retrieval
  t.feature_timestamp
FROM
  \`${PROJECT_ID}.${DATASET_ID}.${DRY_BEANS_TABLE}\` as t
-- This selects all rows from the feature table for a full feature retrieval dataset
EOF

bq query --use_legacy_sql=false "$read_instances_query" || exit $?
echo "Read instances table creation complete: bq://${PROJECT_ID}.${DATASET_ID}.${READ_INSTANCES_TABLE}"

# Clean up the temporary table
bq rm --force --table "${PROJECT_ID}:${DATASET_ID}.${TEMP_TABLE}"

# --- Script Ends Here ---
