# miniML

miniML is a lightweight, custom-built Python library for machine learning, data science, and AI experimentation. It provides modular tools for preprocessing, scaling, and building pipelines, especially for time-series and trading data.  

> ⚠️ This library is a work in progress. APIs and features may change as development continues.

## Features

- **DynamicScaler** – Automatically applies the appropriate scaling method (MinMax, Robust, Standard) for each feature, detecting outliers and bounds dynamically.
- **MachLearnTools** – Provides simple pipelines for:
  - Standard ML tasks (dense/ANN-based workflows)
  - Time-series data (sliding windows, flattening, train/test splitting)
- Utilities for preprocessing, encoding, evaluation, and prediction (modules under development).

## Installation

```bash
git clone https://github.com/yourusername/miniML.git
cd miniML
pip install -r requirements.txt  # if any dependencies
```

## Usage

```python
import pandas as pd
from miniML.dynamicScaler import DynamicScaler
from miniML.machLearnTools import MachLearnTools

# Load your data
X = pd.read_csv("features.csv")
y = pd.read_csv("labels.csv")

# Initialize ML tools
ml_tools = MachLearnTools(X, y)

# Run a standard ML pipeline
X_train, X_test, y_train, y_test = ml_tools.standard_pipeline()

# Run a timeseries pipeline
X_train, X_test, y_train, y_test = ml_tools.timeseries_pipeline(window=10)
```

## Project Structure

miniML/
├─ dynamicScaler.py # Automatic feature scaling
├─ encoder.py # Encoding utilities
├─ evaluate.py # Evaluation metrics and helpers
├─ machLearnTools.py # Pipeline and preprocessing tools
├─ predictor.py # Prediction utilities

## License

This project is licensed under the MIT License.
