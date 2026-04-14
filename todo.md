# 📌 MDK Development Todo List

## 🚀 Upgrades & Maintenance
- [ ] **Upgrade MDK to Python 3.12**
  - Update `pyproject.toml` and `.python-version` targets.
  - Test and synchronize dependencies inside local environments.
  - Update Dockerfile base images (`python:3.11-slim` ➔ `python:3.12-slim`).
- [ ] **Test Makefile compatibility with Artifact Registry**
  - Verify if make targets function correctly when targeting Artifact Registry vs. the original Nexus setup.
## 🧪 Testing & Validation
- [ ] **Test all remaining continuous training models**
  - Verify execution in **Lite Mode** (`lite: true` or `--lite`).
  - Verify execution in **Standard Mode** (Connected to Expanded Model Registry).
  - Scope models: `sklearn_example`, `xgb_example`.

## 🛠 `mdk init` Improvements
- [ ] **Handle `--skip-answered` gracefully when answers file is missing**
  - Currently raises `ValueError: Question "..." is required` if `.copier-answers.yml` doesn't exist.
  - *Recommendation:* Fall back to interactive prompts for missing values with a warning, rather than crashing immediately.

## 🌍 Terraform & FinOps
- [ ] **Revise original Terraform templates**
  - Include labels on all resources (The FinOps setup/mapping wasn't handled properly during execution).
- [ ] **Harden FinOps label fetching in Python**
  - Ensure that when labels do not exist in the GCS bucket config, the code gracefully skips without any residual logging noise or defaults failures.
- [ ] **Create non-Docker Artifact Registry**
  - Add configuration in the shared services Terraform to provision a non-docker AR (e.g., Python package index).

---

## 💡 Recommended Changes & Fixes

### 1. Fix `submitPipeline()` missing `is_lite` forwarding
In `src/mdk/pipeline_tools/execute_pipeline.py`, the `submitPipeline` setup does **not** take or pass along the `is_lite` flag when executing remotely on Vertex AI. Adding this will create a better symmetry with the local execution layout setup.

### 2. Add `--rebuild` or `--no-cache` flag to `mdk run`
To avoid potential local Docker caching discrepancy during speed builds, introducing a toggle to force Docker layer build invalidation or `--no-cache` in `build_images.py` will make debugging internal file sync problems much safer.

### 3. Move display_name underscore replacements upstream
Currently, `vertex.py` converts underscores to dashes during Model ID assignment on line 231 to satisfy Vertex AI structure:
```python
model_id_for_new_model = None if parent_model_resource_name else cfg.model_name.lower().replace("_", "-")
```
Creating a common standard validator inside `RegistryAppConfig` or configuration parsing models earlier on would guarantee absolute safety design layout.
