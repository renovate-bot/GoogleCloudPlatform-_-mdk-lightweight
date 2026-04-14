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

import yaml
import mdk.util.storage
import mdk.util.framework
import mdk.pipeline_tools.compile_pipeline
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

RETRAINING_CONFIG_FILENAME = "retraining_config.yml"


def set_up_retraining_via_model_monitoring(
    monitor_id: str,
    gcs_bucket_root: str,
    pipeline_root: str,
    pipeline_runner_sa: str,
    training_pipeline_name: str,
    inference_pipeline_name: str,
    experiment_name: str,
    region: str,
    ar_repo: str,
    environment: str,
    app_root: str,
    job_id: Optional[str] = None,
    schedule_id: Optional[str] = None,
):
    """
    Compiles, configures, and uploads artifacts needed for an automatic
    retraining pipeline trigger. If both job_id and schedule_id are provided,
    it uploads artifacts for both.

    Args:
        monitor_id: The ModelMonitor resource id.
        job_id: The ModelMonitoringJob resource id.
        schedule_id: The ModelMonitoring Schedule resource id.
        gcs_bucket_root: The root GCS bucket for storing artifacts.
        pipeline_root: GCS location for storing pipeline artifacts.
        pipeline_runner_sa: Service Account to be used for running Vertex Pipeline.
        training_pipeline_name: The name of the training pipeline to compile.
        inference_pipeline_name: The name of the inference pipeline to compile.
        experiment_name: The name of the Vertex Experiment to be attached with the pipeline.
        region: The GCP region.
        ar_repo: Artifact registry repo to use for compiling the image
        environment: The environment pipeline parameter.
        app_root: The parent directory of the config directory (e.g. /app)
    """
    logger.info("Setting up artifacts for automatic retraining trigger...")

    if not job_id and not schedule_id:
        raise ValueError(
            "Either job_id or schedule_id must be provided for setting up retraining."
        )

    pipeline_mapping = mdk.util.framework.get_pipeline_mapping()

    # # 2. Compile both pipelines
    for pipeline_name in (training_pipeline_name, inference_pipeline_name):
        logger.info(f"Compiling: {pipeline_name}")
        if pipeline_name not in pipeline_mapping:
            raise RuntimeError(
                f"Unable to find pipeline: {pipeline_name}  (Does it exist in the"
                f" in the pipeline base dir files?)"
            )
        pipeline_dir = pipeline_mapping[pipeline_name]
        # TODO: Note that this will use the image with the :latest tag.  This
        #   could lead to a problem if someone builds an image after this
        #   completes but before the pipeline starts.  Maybe pass the current
        #   image digest as a parameter to the pipeline, so that it can be
        #   passed here via the tags argument?
        mdk.pipeline_tools.compile_pipeline.compilePipeline(pipeline_dir, ar_repo)

    # Define the local paths where the compiler is expected to save the specs
    pipeline_dir = pipeline_mapping[training_pipeline_name]
    pipeline_config_filename_relative_path = (
        pipeline_dir.parent.parent
        / "config"
        / mdk.util.framework.PIPELINE_CONFIG_BASENAME
    )
    pipeline_config_filename_abs_path = os.path.join(
        app_root, pipeline_config_filename_relative_path
    )
    training_pipeline_spec_path = mdk.util.framework.getPipelineSpecFilename(
        training_pipeline_name, pipeline_config_filename_abs_path
    )

    pipeline_dir = pipeline_mapping[inference_pipeline_name]
    pipeline_config_filename_relative_path = (
        pipeline_dir.parent.parent
        / "config"
        / mdk.util.framework.PIPELINE_CONFIG_BASENAME
    )
    pipeline_config_filename_abs_path = os.path.join(
        app_root, pipeline_config_filename_relative_path
    )
    inference_pipeline_spec_path = mdk.util.framework.getPipelineSpecFilename(
        inference_pipeline_name, pipeline_config_filename_abs_path
    )

    # 2. Determine all GCS locations to upload to.
    upload_targets = []
    gcs_monitor_folder_path = os.path.join(
        gcs_bucket_root, "retraining-configs", f"monitor-{monitor_id}"
    )

    if job_id:
        upload_targets.append(os.path.join(gcs_monitor_folder_path, f"job-{job_id}"))
    if schedule_id:
        upload_targets.append(
            os.path.join(gcs_monitor_folder_path, f"schedule-{schedule_id}")
        )

    # 3. Create the retraining config file locally
    # This config is generic and doesn't contain the GCS paths for the specs yet.
    retraining_config = {
        "gcs_bucket_root": gcs_bucket_root,
        "region": region,
        "pipeline_root": pipeline_root,
        "pipeline_runner_sa": pipeline_runner_sa,
        "experiment_name": experiment_name,
        "parameters": {"environment": environment},
    }

    # 4. Loop through each target, update the config with specific URIs, and upload everything.
    for gcs_target_path in upload_targets:
        logger.info(f"Uploading retraining artifacts to: {gcs_target_path}")

        # Define GCS URIs for the artifacts for this specific target path
        gcs_training_spec_uri = os.path.join(
            gcs_target_path, "training_pipeline_spec.yml"
        )
        gcs_inference_spec_uri = os.path.join(
            gcs_target_path, "inference_pipeline_spec.yml"
        )
        gcs_retraining_config_uri = os.path.join(
            gcs_target_path, RETRAINING_CONFIG_FILENAME
        )

        # Add the specific spec URIs to the config for this target
        retraining_config["inference_pipeline_spec_uri"] = gcs_inference_spec_uri
        retraining_config["training_pipeline_spec_uri"] = gcs_training_spec_uri

        # Write the finalized config to a local file
        with open(RETRAINING_CONFIG_FILENAME, "w") as fout:
            fout.write(yaml.safe_dump(retraining_config, indent=4))

        try:
            mdk.util.storage.upload(
                RETRAINING_CONFIG_FILENAME, gcs_retraining_config_uri
            )
            mdk.util.storage.upload(training_pipeline_spec_path, gcs_training_spec_uri)
            mdk.util.storage.upload(
                inference_pipeline_spec_path, gcs_inference_spec_uri
            )
            logger.info(f"Successfully uploaded artifacts for {gcs_target_path}")
        except Exception as e:
            e.add_note(f"Failed to upload retraining artifacts to {gcs_target_path}")
            raise
