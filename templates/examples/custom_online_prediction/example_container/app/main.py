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

from fastapi import FastAPI, Request
from google.cloud import storage
import joblib
import tensorflow as tf
import torch
import xgboost as xgb
import numpy as np
import logging
import os
import pickle

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="{levelname}:{asctime}:{name}:{lineno}:{message}",
    level=logging.INFO,
    style="{",
)

app = FastAPI()
gcs_client = storage.Client()


# Using the follow naming conventions:
# https://cloud.google.com/vertex-ai/docs/model-registry/import-model#upload-model-artifacts
MODEL_FILENAMES = [
    "model.pkl",
    "model.joblib",
    "model.bst",
    "model.pth",
    "model.pt",
    "saved_model.pb",
]
PREPROCESSOR_FILENAME = "preprocessor.pkl"

_model = None
_preprocessor = None


try:
    with open(PREPROCESSOR_FILENAME, "wb") as preprocessor_f:
        gcs_client.download_blob_to_file(
            f"{os.environ['AIP_STORAGE_URI']}/{PREPROCESSOR_FILENAME}", preprocessor_f
        )
    with open(PREPROCESSOR_FILENAME, "rb") as f:
        _preprocessor = pickle.load(f)
except Exception as e:
    logger.error(f"Error downloading or loading preprocessor.pkl: {e}")

downloaded_model_filename = None
for filename in MODEL_FILENAMES:
    model_uri = f"{os.environ['AIP_STORAGE_URI']}/{filename}"
    try:
        with open(filename, "wb") as model_f:
            gcs_client.download_blob_to_file(model_uri, model_f)
        logger.info(f"Successfully downloaded model file: {filename}")
        downloaded_model_filename = filename
        break

    except FileNotFoundError:
        logger.error(f"Model file not found at: {model_uri}")
        continue
    except Exception as e:
        logger.error(
            f"Error downloading or opening model file {filename} from {model_uri}: {e}"
        )
        continue

if downloaded_model_filename:
    if downloaded_model_filename.endswith(
        ".joblib"
    ) or downloaded_model_filename.endswith(".bst"):
        try:
            _model = joblib.load(downloaded_model_filename)
            logger.info(f"Successfully loaded model from {downloaded_model_filename}")
        except Exception as e:
            logger.error(f"Error loading model from {downloaded_model_filename}: {e}")
    elif downloaded_model_filename.endswith(".pkl"):
        try:
            with open(downloaded_model_filename, "rb") as f:
                _model = pickle.load(f)
            logger.info(f"Successfully loaded model from {downloaded_model_filename}")
        except Exception as e:
            logger.error(f"Error loading model from {downloaded_model_filename}: {e}")
    elif downloaded_model_filename.endswith(
        ".pth"
    ) or downloaded_model_filename.endswith(".pt"):
        try:
            _model = torch.load(downloaded_model_filename)
            logger.info(f"Successfully loaded model from {downloaded_model_filename}")
        except Exception as e:
            logger.error(f"Error loading model from {downloaded_model_filename}: {e}")
    elif downloaded_model_filename == "saved_model.pb":
        # load model directly from GCS:
        try:
            _model = tf.keras.models.load_model(os.environ["AIP_STORAGE_URI"])
            logger.info(f"Successfully loaded model from {downloaded_model_filename}")
        except Exception as e:
            logger.error(f"Error loading model from {downloaded_model_filename}: {e}")
else:
    logger.error("No model file was successfully downloaded.")

if _model:
    logger.info("Model is loaded and ready to be used.")


@app.get(os.environ["AIP_HEALTH_ROUTE"], status_code=200)
def health():
    return {"status": "healthy"}


@app.post(os.environ["AIP_PREDICT_ROUTE"])
async def predict(request: Request):
    body = await request.json()
    instances = body["instances"]
    inputs = np.asarray(instances)
    if _preprocessor:
        inputs = _preprocessor.preprocess(inputs)
    if isinstance(_model, xgb.core.Booster):
        inputs = xgb.DMatrix(inputs, feature_names=None)
    outputs = _model.predict(inputs)
    logger.info(f"outputs: {outputs}")
    return {"predictions": outputs.tolist()}
