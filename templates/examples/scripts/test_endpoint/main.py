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
A script to send multiple concurrent prediction requests to a Google Cloud Vertex AI Endpoint.

This script is designed to help you load-test your deployed models on Vertex AI.
It uses a ThreadPoolExecutor to send requests in parallel, allowing you to simulate
multiple users or a high-volume application.

Prerequisites:
1.  Install the required libraries:
    pip install google-cloud-aiplatform pandas

2.  Authenticate with Google Cloud:
    gcloud auth application-default login

3.  Have a deployed model on a Vertex AI Endpoint.
"""

import time
import concurrent.futures
from typing import Dict, List, Any
import pandas as pd
from google.cloud import aiplatform

# --- Configuration ---
# TODO: Replace these with your actual project and endpoint details.
PROJECT_ID = ""  # e.g. "my-sample-project"
LOCATION = "us-east4"  # e.g., "us-central1"
ENDPOINT_ID = (
    ""  # e.g. "projects/889810617340/locations/us-east4/endpoints/939418336725303296"
)

# --- Test Parameters ---
# The CSV file containing a list of prediction instances.
# Each item in the list will be sent as a separate request.
INPUT_DATA_FILE = "dry_beans_inference_data.csv"

# Total number of requests to send. The script will cycle through the instances
# in the input file until this number is reached.
TOTAL_REQUESTS = 50

# Number of concurrent requests to send at a time.
MAX_WORKERS = 10
# --------------------


def send_prediction_request(
    endpoint_instance: aiplatform.Endpoint, instance: List[Any], request_id: int
) -> Dict[str, Any]:
    """
    Sends a single prediction request to the Vertex AI endpoint.

    Args:
        endpoint_instance: The initialized Vertex AI Endpoint object.
        instance: A list representing a single prediction instance.
        request_id: A unique identifier for the request for logging.

    Returns:
        A dictionary containing the request status, response, and latency.
    """
    start_time = time.time()
    try:
        # The predict method sends the request and waits for the response.
        response = endpoint_instance.predict(instances=[instance])
        end_time = time.time()
        return {
            "request_id": request_id,
            "status": "success",
            "latency_seconds": round(end_time - start_time, 4),
            "prediction": response.predictions[0],
        }
    except Exception as e:
        end_time = time.time()
        print(f"--- Request {request_id} failed: {e} ---")
        return {
            "request_id": request_id,
            "status": "failure",
            "latency_seconds": round(end_time - start_time, 4),
            "error": str(e),
        }


def main():
    """
    Main function to initialize the client and run the load test.
    """
    print("--- Vertex AI Endpoint Load Test ---")
    print(f"Project: {PROJECT_ID}, Location: {LOCATION}")
    print(f"Endpoint: {ENDPOINT_ID}")
    print("-" * 35)
    print(f"Total Requests: {TOTAL_REQUESTS}")
    print(f"Concurrency Level (Max Workers): {MAX_WORKERS}")
    print("-" * 35)

    # Initialize the Vertex AI client
    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    # Get the endpoint resource
    try:
        endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)
        print("Successfully initialized endpoint.")
    except Exception as e:
        print(f"Error initializing endpoint: {e}")
        print("Please ensure your Project ID, Location, and Endpoint ID are correct.")
        return

    # Load instances from the CSV file
    try:
        df = pd.read_csv(INPUT_DATA_FILE, header=None)

        # --- FIX: Convert all data to a numeric format ---
        # The XGBoost error indicates string values are being sent to the model, which expects numbers.
        # This block ensures all data is converted to a numeric type before sending.

        # 1. Convert all columns to numeric types. Any values that can't be converted will become NaN (Not a Number).
        numeric_df = df.apply(pd.to_numeric, errors="coerce")

        # 2. Replace any resulting NaN values with 0.
        # Note: Depending on your model, filling with the mean or median might be a better strategy.
        numeric_df.fillna(0, inplace=True)

        # 3. Convert the cleaned DataFrame to the list of lists format for the API call.
        instances = numeric_df.astype(float).values.tolist()
        # --- END FIX ---

        if not instances:
            print(f"Error: {INPUT_DATA_FILE} is empty or could not be read properly.")
            return
        print(f"Loaded {len(instances)} unique instance(s) from {INPUT_DATA_FILE}.")
    except FileNotFoundError:
        print(f"Error: Input data file not found at '{INPUT_DATA_FILE}'")
        return
    except Exception as e:
        print(f"Error reading or processing CSV file: {e}")
        return

    success_count = 0
    failure_count = 0
    total_latency = 0
    results = []

    # Use a ThreadPoolExecutor for concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create a future for each request
        futures = [
            executor.submit(
                send_prediction_request,
                endpoint,
                # Cycle through the instances if TOTAL_REQUESTS > len(instances)
                instances[i % len(instances)],
                i + 1,
            )
            for i in range(TOTAL_REQUESTS)
        ]

        print(f"\nSubmitting {len(futures)} requests...")

        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            if result["status"] == "success":
                success_count += 1
                total_latency += result["latency_seconds"]
                print(
                    f"Request {result['request_id']}: Success ({result['latency_seconds']}s)"
                )
            else:
                failure_count += 1
                print(
                    f"Request {result['request_id']}: Failure ({result['latency_seconds']}s)"
                )

    print("\n--- Test Summary ---")
    print(f"Total Requests Sent: {TOTAL_REQUESTS}")
    print(f"Successful Requests: {success_count}")
    print(f"Failed Requests: {failure_count}")

    if success_count > 0:
        average_latency = total_latency / success_count
        print(f"Average Latency for Successful Requests: {average_latency:.4f} seconds")

    print("-" * 20)


if __name__ == "__main__":
    main()
