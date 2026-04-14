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

# Set some constants:
environment=dev
dataset_id=ml_dataset
gcp_conf_yaml=../model_products/xgb_example/state/$environment.yml
csv=../src/xgb_example/data/Dry_Beans_Dataset.csv
csv_with_id="$(dirname $csv)/numbered_$(basename $csv)"
raw_data_table=samples
mapping_csv=mapping.csv

# Read our config file.

project_id=$(python -m mdk.config.get_config_value $gcp_conf_yaml project_id)
region="us"
data_bucket=$(python -m mdk.config.get_config_value $gcp_conf_yaml data_bucket)

if [ -z $project_id ] ; then
    echo "Error reading config: $gcp_conf_yaml"
    exit -1
fi

# In the below, we assume there is no scheme, so we strip it off here, if it exists.
data_bucket=${data_bucket#gs://}

echo "Checking whether dataset exists: $project_id:$dataset_id"
bq --location $region ls --dataset_id $project_id:$dataset_id \
    && dataset_exists=true || dataset_exists=false

# Create a dataset, if it does not exist.
if [ $dataset_exists == true ] ; then
    echo "$project_id:$dataset_id exists => skipping"
else
    echo "$project_id:$dataset_id does not exist => creating"
    bq \
        --location=$region \
        mk \
        --dataset \
        ${project_id}:${dataset_id} \
        || exit $?
fi

# Add a row number to the CSV.
echo $csv
echo $csv_with_id
awk -F',' 'BEGIN {OFS=FS} NR==1 {print "RowNumber,"$0; next} {print NR-1","$0}' $csv > $csv_with_id

# Upload our CSV.
echo "Uploading: ${data_bucket}/$csv_with_id"
gcloud storage cp "$csv_with_id" gs://$data_bucket/ || exit $?

# (Re-) create our table.
echo "(Re-) creating: $project_id:$dataset_id.$raw_data_table"
bq rm --force --table "$project_id:$dataset_id.$raw_data_table" || exit $?
bq mk --table "$project_id:$dataset_id.$raw_data_table" \
id:integer,\
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

# Load our CSV.
echo "Loading: gs://$data_bucket/$(basename $csv_with_id)"
bq \
    --location=$region \
    load \
    --source_format=CSV \
    --skip_leading_rows=1 \
    "$project_id:$dataset_id.$raw_data_table" \
    gs://$data_bucket/$(basename $csv_with_id) \
    || exit $?

# echo "Load complete."

# Create a table to map the classes to integers.
read -r -d "" query << EOF
create or replace table \`$project_id.$dataset_id.class_mapping\` as
with T as (
  select
    distinct Class
  from
    \`$project_id.$dataset_id.samples\`
  order by
    Class
)
select
  T.Class as Class,
  row_number() over () - 1 as ClassIndex
from
  T
EOF
bq query --use_legacy_sql=false "$query" || exit $?

# Create a view with all the data, without the id column.
query="select * except(id) from \`$project_id.$dataset_id.$raw_data_table\`"
bq query --use_legacy_sql=false "create or replace view \`$project_id.$dataset_id.all_samples\` as $query;" || exit $?

# Create a view for model monitoring target, without the id or Class columns.
query="select * except(id, Class) from \`$project_id.$dataset_id.$raw_data_table\` limit 1000"
bq query --use_legacy_sql=false "create or replace view \`$project_id.$dataset_id.subset_no_labels\` as $query;" || exit $?
