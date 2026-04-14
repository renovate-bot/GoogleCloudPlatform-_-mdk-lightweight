# How to Use the Model Serialization Library

This document outlines the two primary ways to use the model serialization functions:

1. As a library in your Python code (e.g., within a pipeline component)
2. As a standalone command-line tool for quick tests

---

## 1. Using it from Another Python Script *(The Intended Way)*

This is the most common and flexible approach. The library exposes simple `load` and `save` functions that can be imported directly from the `mdk.model` package.

### Example: `your_component.py`

```python
import logging
import numpy as np
import xgboost as xgb
import os

# Import the public facade functions from the top-level model package
from mdk.model import load, save

# It's good practice for the application using the library to configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

try:
    # --- Create and Write a Model ---
    print("Creating a sample XGBoost model...")
    # Create a simple, dummy model for demonstration
    model_to_save = xgb.XGBClassifier(n_estimators=5, use_label_encoder=False)
    model_to_save.fit(np.array([[0], [1]]), np.array([0, 1]), eval_metric='logloss')

    # Use the library's save function. It automatically infers the best
    # format and returns the filename it used.
    output_filename = save(model=model_to_save, filename="my_xgboost_model.ubj")

    print(f"Model successfully written to: {output_filename}")

    # --- Read the Model Back ---
    print("\nReading the model back from the file...")

    # Use the load function, specifying the class for clarity
    loaded_model = load(filename=output_filename, model_class=xgb.XGBClassifier)

    print("Model successfully loaded.")
    print(f"Loaded model type: {type(loaded_model)}")

except ImportError:
    print("Error: This example requires numpy and xgboost. Please install them.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

finally:
    # Clean up the created file
    if 'output_filename' in locals() and os.path.exists(output_filename):
        os.remove(output_filename)
```

## 2. Using it from the Command Line *(For Quick Tests)*

The package includes a command-line interface (`cli.py`) for easy testing of the serialization logic from a shell. This is ideal for verifying that different model types can be written and load correctly without creating a full Python script.

The script is invoked using:

```bash
python -m mdk.model.io.cli
```

The CLI has two main commands: `save` and `load`.

---

## ✅ Writing a Test Model

The `save` command creates a small, in-memory dummy model of a specified framework and saves it to a file.

### Example

```bash
# This command creates a dummy XGBoost model and saves it to a file.
python -m mdk.model.io.cli save xgboost --output my_test_model.ubj
```

Expected Output
```
INFO: Creating a dummy 'xgboost' model for save command...
INFO: Write successful.

--- Model Write Succeeded ---
Framework: xgboost
Dummy model written to: my_test_model.ubj
-----------------------------
```

## 📥 Reading a Model

The `load` command loads a serialized model from a file and prints its Python type to confirm successful deserialization.

### Example

```bash
# This command loads the file we just created.
python -m mdk.model.io.cli load my_test_model.ubj
```
Expected Output
```
INFO: Attempting to load model from: my_test_model.ubj
INFO: Read successful.

--- Model Read Succeeded ---
File: my_test_model.ubj
Loaded Object Type: <class 'xgboost.sklearn.XGBClassifier'>
```
