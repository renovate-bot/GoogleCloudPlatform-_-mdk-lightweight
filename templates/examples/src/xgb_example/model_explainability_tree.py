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

from typing import Tuple, Union, List
import mdk.data
import matplotlib.pyplot as plt
import shap
import base64
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import google.cloud.bigquery
from PIL import Image
import numpy as np
import logging
import math
import os

logger = logging.getLogger(__name__)


def convert_png_to_embedded_html(
    local_png_path: str, title: str = "Pipeline Image Artifact"
) -> str:
    """
    Reads a local PNG file, encodes it in Base64, and returns a string
    containing a full HTML page with the image embedded.

    Args:
        local_png_path (str): The local file path to the PNG image.
        title (str): The title to display on the HTML page.

    Returns:
        str: A complete HTML string with the image embedded.
    """
    try:
        # 1. Read the PNG file in binary mode
        with open(local_png_path, "rb") as f:
            png_bytes = f.read()

        # 2. Encode the binary data to a Base64 string
        encoded_string = base64.b64encode(png_bytes).decode("utf-8")

        # 3. Construct the HTML content with the Base64 data URI
        html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                text-align: center;
            }}
            img {{
                max-width: 90%; /* Ensures image fits on screen */
                height: auto;
                border: 1px solid #ccc;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            }}
        </style>
    </head>
    <body>
        <h1></h1>
        <h2></h2>
        <img src="data:image/png;base64,{encoded_string}" alt="{title} Plot">
        <p>Image embedded using Base64 Data URI.</p>
    </body>
    </html>
    """
        return html_content
    except FileNotFoundError:
        return f"<h1>Error: PNG file not found at {local_png_path}</h1>"
    except Exception as e:
        return f"<h1>An error occurred during encoding: {e}</h1>"


def _create_beeswarm_plot(i: int, shap_values: shap.Explanation, beeswarm_path: str):
    """
    Creates a beeswarm plot for each class based on pre-calculated shap values and writes to a PNG file

    Args:
        i (int): ith class in the precalculated shap values
        shap_values (ndarray): The array with the shap values
        beeswarm_path (str): the local file path to save the beeswarm plots.

    Returns:
        Does not return anything. Implicitly saves a PNG file to beeswarm_path
    """
    plt.figure(figsize=(4, 3))
    shap.plots.beeswarm(shap_values[:, :, i], show=False)
    plt.tight_layout()
    plt.savefig(beeswarm_path, bbox_inches="tight")
    plt.close()


def _create_plot_list_of_classes(
    classes_list: List[str], shap_values: shap.Explanation
) -> List[str]:
    """
    Creates a beeswarm plot for each class and adds the image file to a list

    Args:
        classes_list (List[str]): The list of classes for which the plot is to be generated
        shap_values (nd array): The pre-calculated shap values array

    Returns:
        List[str] : List of plot images, one per class.
    """
    plot_list = []
    for i, c in enumerate(classes_list):
        plot_path = f"Class_{c}.png"
        _create_beeswarm_plot(i, shap_values, plot_path)
        plot_list.append(plot_path)
    return plot_list


def combine_pngs_with_matplotlib(
    png_files: List[str],
    output_filename: str = "combined_plots_mpl_stitch.png",
    ncols: int = 2,
) -> str:
    """
    Reads a list of PNG files (using Pillow), combines them into a single
    Matplotlib figure using subplots, adds the filename as a title to each
    subplot, and saves the final image.

    Args:
        png_files (List[str]): A list of file paths to the individual PNG plots.
        output_filename (str): The path to save the final combined image.
        ncols (int): The number of columns for the grid layout. Defaults to 3.

    Returns:
        str: The path to the saved combined image file.
    """
    if not png_files:
        logger.error("Error: The list of PNG files is empty.")
        return ""

    try:
        # Load the images into a list of NumPy arrays (Matplotlib's preferred format)
        # Pillow is used here just for loading the image data.
        image_arrays = [np.asarray(Image.open(f)) for f in png_files]
    except Exception as e:
        logger.error(f"Error loading images with Pillow: {e}")
        return ""

    n_plots = len(image_arrays)
    nrows = int(math.ceil(n_plots / ncols))

    # 1. Create the Matplotlib Figure and Axes grid
    # Increased figsize to give more room for titles and layout
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 5.5, nrows * 5),  # Adjusted figure size
        constrained_layout=True,
    )

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # 2. Iterate and display each image in its subplot
    for i in range(n_plots):
        ax = axes[i]

        # Use Matplotlib's imshow to display the image array
        ax.imshow(image_arrays[i])

        # Add the title using the filename, making sure it's visible
        title_text = os.path.basename(png_files[i])
        ax.set_title(
            title_text,
            fontsize=12,  # Larger font size for better visibility
            color="darkblue",  # Distinct color
            fontweight="bold",  # Bold for emphasis
        )

        # IMPORTANT: Turn off axes, ticks, and labels for a clean stitch
        ax.axis("off")

    # 3. Remove any unused subplots
    for i in range(n_plots, nrows * ncols):
        fig.delaxes(axes[i])

    # Add a main title for the entire figure (optional, but good practice)
    fig.suptitle(
        f"Combined SHAP Beeswarm Plots ({n_plots} Classes)",
        fontsize=16,
        y=1.02,  # Position slightly above the top subplots
    )

    # 4. Save the combined figure as a single PNG file
    plt.savefig(output_filename, bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info(
        f"Matplotlib successfully stitched {n_plots} image files with titles into: {output_filename}"
    )
    return output_filename


def _plot_to_markdown(file_path: str, title: str) -> str:
    """Helper function to encode a plot image into a Markdown string.

    Args:
        file_path : the plot image file to be shown in pipeline console markdown
        title: title of the plot

    Returns:
        str: Markdown string string containing the plot image
    """
    with open(file_path, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode("utf-8")
    return f"### {title}\n![{title}](data:image/png;base64,{encoded_string})"


def _create_summary_plot(
    shap_values: shap.Explanation, shap_dtest: pd.DataFrame
) -> str:
    """
    Creates summary plot based on shap values and a test dataframe

    Args:
        shap_values : pre-calculated shap values
        shap_dtest: test dataframe corespodning to the shap values

    Returns:
        str: plot image filepath

    """
    logger.info("Generating SHAP summary plot...")
    plt.figure()
    shap.summary_plot(shap_values, shap_dtest, plot_type="bar", show=False)
    plt.tight_layout()
    summary_plot_file = "summary_plot.png"  # file for saving the shap plot
    plt.savefig(summary_plot_file, bbox_inches="tight")
    plt.close()
    return summary_plot_file


def model_explainability_tree(
    model: Union[
        xgb.XGBClassifier,
        xgb.XGBRegressor,
        xgb.Booster,
        RandomForestClassifier,
        # lgb.LGBMClassifier,
        # lgb.Booster
    ],
    test_uri: str,
    target_class: str,
    shap_sample_size: int,
) -> Tuple[shap.Explanation, pd.DataFrame, List[str]]:
    """
    Run shap explainability for tree models

    Args:
        model: the trained tree model - this can be an xgb, lgb or a sklearn classifier model
        test_uri: the uri to the test dataset resource
        target_class: the column of the dataset that has teh target class values
        shap_sample_size: the number of rows for which shap explainability wil be run
    """

    client = google.cloud.bigquery.Client()
    # read the test dataset from its uri which points to a BQ table
    df = mdk.data.getDataframeFromBigQuery(client, test_uri).sample(
        n=shap_sample_size, random_state=42
    )
    # Get list of classes to show the class values in the plot
    classes_list = list(df[target_class].unique())
    # Get features on which to run shap values
    shap_dtest = df.drop(columns=[target_class])
    explainer = shap.TreeExplainer(
        model
    )  # use TreeExplainer which is optimized for tree models
    shap_values = explainer(shap_dtest)
    return shap_values, shap_dtest, classes_list
