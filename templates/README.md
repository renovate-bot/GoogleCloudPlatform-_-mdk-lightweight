# Getting started with the MLOps Development Kit (MDK)

This document contains a short overview of how to get started using the MDK,
including running an example pipeline, and editing the `model_workflow` and
its associated configuration. More in-depth documentation can be found under
the `docs/` directory.

## Prerequisites

- Please make sure you have already run

```bash
gcloud auth login
```

and

```bash
gcloud auth application-default login
```

- Please also make sure you have already configured Docker authentication for
Artifact Registry by running:

```bash
gcloud auth configure-docker ${REGION}-docker.pkg.dev
```

where `${REGION}` is the region you will use (probably `us`, for default terraform values).

## Running an Example

To run an example pipeline, follow the following steps

1. Create a new virtual
environment in the target directory:

```bash
uv venv
uv sync
uv pip install -e .
```

2. Activate the target directory's new virtual environment:

```bash
# For Linux/Mac
source .venv/bin/activate

# For Windows
.venv\Scripts\activate
```

3. Run the data upload script to make the example BigQuery tables available:
```bash
# For Linux/Mac
cd examples/scripts
chmod +x load-dry-beans-data-to-bq.sh
./load-dry-beans-data-to-bq.sh

# For Windows
dir examples/scripts
bash load-dry-beans-data-to-bq.sh
```

4. Get a list of available pipelines.

```bash
cd ../..
mdk list
```

5. Submit a pipeline for execution

```bash
mdk run xgb_training_pipeline
```

## Iterative development

In the `src/model_workflow` directory, there are several files that correspond to
different steps in an ML pipeline (train, evaluate, batch_predict, etc). These
files can be updated with your specific workloads. Any external variables that
are necessary for your workloads (e.g. hyperparameters, GCS file URIs, dataset
paths, etc) can be placed in the general config file
[model_products/main_product/config/config.yml](model_products/main_product/config/config.yml).
This config file and the `src/model_workflow` directory are the two primary
areas for your active development.

Please see [docs/configs/mdk_general_config_usage.md](docs/configs/mdk_general_config_usage.md)
for more guidelines on how to use the general config.

There are a number of pre-set pipelines that are available for your use, including:
- `training_pipeline`: Runs the following files in containerized components,
    as well as an upload model operation that saves the model to model registry:
    - prepare.py
    - optimize_hyperparameters.py
    - train.py
    - evaluate.py
- `batch_inference_pipeline`: Runs a batch prediction on the model specified,
    as well as an optional monitoring operation:
    - batch_predict.py
- `deployment_pipeline`: Deploys a model to an endpoint, as well as an optional
    monitoring operation.

Update the [pyproject.toml](pyproject.toml) file with any new python package
dependencies you need to add.

If you wish to add additional components, images, or change the image that a
specific component is using, follow the following instructions:
1. To create a new component, go into the `model_products/main_product/components`
directory and copy one of the components that is currently there. Rename the
directory to the name you wish to name your new component, and then update the
`components/my_new_component/component.py` file. Change the name of the function
within this file as well.
2. To create a new image, go into the `model_products/main_product/images`
directory. There, you will see 2 available images. You can either edit the files
under the `byoc` directory, or copy `byoc` and name it rename it, then edit the
files.
3. To create a new pipeline, go into the `model_products/main_product/pipelines`
directory and copy one of the pipelines that is currently there. Rename the
directory to the name you wish to name your new pipeline, and then update the
`pipelines/my_new_pipeline/pipeline.py` file. Change the name of the function
within this file as well.
4. You must register any new components, images, or pipelines within the pipeline
config file [model_products/main_product/config/pipeline_config.yml](model_products/main_product/config/pipeline_config.yml).
To do this, you can add your specific components/images/pipelines in the specific
section within this file. Be sure to specify the correct `function` and
`module_path`.

Please see [docs/configs/mdk_pipeline_config_usage.md](docs/configs/mdk_pipeline_config_usage.md)
for more guidelines on how to use the pipeline config.

If you wish to use the bring-your-own-container (BYOC) functionality, please see
[docs/modules/mdk_custom_job_usage.md](docs/modules/mdk_custom_job_usage.md)
for more guidelines.


Any other operations are documented in detail under the `docs/` directory.
