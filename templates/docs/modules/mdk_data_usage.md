# How to use the `mdk.data` library

This document outlines how to use the funcionality provided by the `mdk.data`
subpackage.

## Example Usage of `mdk.data.getDataframeFromBigQuery()`

This is a convenience function to fetch all the data in a BigQuery table and
return it as a Pandas `DataFrame`.

```python
    gcp_project_id = "my-project"
    bq_dataset = "my_dataset"
    table_name = "my_table"
    uri = f"bq://{gcp_project_id}.{bq_dataset}.{table_name}"

    client = google.cloud.bigquery.Client()
    df = mdk.data.getDataframeFromBigQuery(client, uri)

    print("Here are the first few lines of the table we have queried:")
    print(df.head())
```

## Example Usage of `mdk.data.getDataFrameFromFeatureStore()`

This is a convenience function to fetch all the data in a Feature Store feature
group and return it as a Pandas `DataFrame`.

```python
    import mdk.data

    gcp_bigquery_project_id = "my-bigquery-project"
    bq_dataset = "my_dataset"
    table_name = "my_table"
    # This is a BQ table with entity_id and timestamp columns for point-in-time
    #   lookup.
    fs_read_instances_uri = f"bq://{gcp_bigquery_project_id}.{bq_dataset}.{table_name}"

    fs_feature_group_id = "my_feature_group_id"
    fs_project_id = "my-project"
    fs_region = "us"

    df = mdk.data.getDataFrameFromFeatureStore(
        fs_feature_group_id, fs_read_instances_uri, fs_project_id, fs_region
    )

    print("Here are the first few lines of the data we have queried:")
    print(df.head())
```

If you experience an error saying that the resource is not found, even though
you are sure it exists, one place to check is to make sure the specified region
matches the region on the Feature Store instance.

## Example Usage of `mdk.data.DatasetHandler`

This class is a convenience wrapper to aid in passing datasets to Kubeflow as a
URI.  It makes sense the context of a Kubeflow component, which uses URIs as
its abstraction for passing data to other components in a Kubeflow pipeline.

### Example for using a BigQuery table as a dataset:

```python
import mdk.data
from kfp import dsl


@dsl.component(target_image="us-docker.pkg.dev/my-project/my-repo/my-image")
def my_component(
    my_kfp_dataset: dsl.Output[dsl.Dataset],
):
    # Instantiate a DatasetHandler:
    my_dataset_handler = mdk.data.DatasetHandler(my_kfp_dataset.uri)

    # Set up the DatasetHandler for use with a BigQuery table:
    gcp_project_id = "my-project"
    bq_dataset = "my_dataset"
    table_name = "my_table"
    my_dataset_handler.set_bigquery_table(project_id, bq_dataset, train_table)

    # Pass the appropriate BigQuery URI to Kubeflow:
    my_kfp_dataset.uri = my_dataset_handler.uri
```

### Example for using a flat file as a dataset:

```python
import mdk.data
from kfp import dsl


@dsl.component(target_image="us-docker.pkg.dev/my-project/my-repo/my-image")
def my_component(
    my_kfp_dataset: dsl.Output[dsl.Dataset],
):
    # Instantiate a DatasetHandler:
    my_dataset_handler = mdk.data.DatasetHandler(my_kfp_dataset.uri)

    # Set up the DatasetHandler for use with a BigQuery table:
    csv_filename = "example.csv"
    train_dataset_handler.set_local_file(csv_filename)

    # Pass the appropriate BigQuery URI to Kubeflow:
    my_kfp_dataset.uri = my_dataset_handler.uri
```

Please note: The above example uses a CSV file, but it will work equally well
for parquet files, pickle files, etc.
