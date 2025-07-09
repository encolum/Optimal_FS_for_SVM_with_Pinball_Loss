# Pin-FS-SVM: Pinball Loss Feature Selection Support Vector Machine

## Overview

This repository implements **Pin-FS-SVM (Pinball Loss Feature Selection Support Vector Machine)**, a robust feature selection model that combines pinball loss function with support vector machines for binary classification tasks. The project includes comprehensive experiments comparing Pin-FS-SVM against several baseline methods across multiple datasets with different noise conditions.

## Key Features

- **Robust Feature Selection**: Uses pinball loss function for improved robustness to outliers and noise
- **Multiple SVM Variants**: Implements 7 different SVM-based models for comparison
- **Comprehensive Evaluation**: Tests across 6 datasets with 4 different noise conditions
- **Statistical Analysis**: Includes Wilcoxon signed-rank tests with Benjamini-Hochberg correction
- **Automated Experiments**: Grid search with cross-validation for hyperparameter optimization

## Models Implemented

### 1. Pin-FS-SVM (Proposed Method)
- **File**: `src/models/pin_fs_svm.py`
- **Description**: Feature selection SVM using pinball loss function
- **Key Parameters**: 
  - `B`: Maximum number of features to select
  - `C`: Regularization parameter
  - `tau`: Pinball loss parameter (0 < tau ≤ 1)

### 2. Baseline Models

#### MILP1 (`src/models/milp1_svm.py`)
- Mixed Integer Linear Programming SVM with L1-norm feature selection
- Uses binary variables for explicit feature selection

#### L1-SVM (`src/models/l1_svm.py`)
- Support Vector Machine with L1 regularization
- Promotes sparsity in feature weights

#### L2-SVM (`src/models/l2_svm.py`)
- Standard L2-regularized Support Vector Machine
- Baseline comparison without feature selection

#### Pinball SVM (`src/models/pinball_svm.py`)
- SVM with pinball loss but without feature selection
- Tests the effect of pinball loss alone

#### Fisher-SVM (`src/models/fisher_svm.py`)
- Uses Fisher score for feature selection
- Selects features based on F-score thresholds

#### RFE-SVM (`src/models/rfe_svm.py`)
- Recursive Feature Elimination SVM
- Eliminates features iteratively based on weight ranking

## Datasets

The project evaluates performance on 6 benchmark datasets:

1. **WDBC (Breast Cancer Wisconsin Diagnostic)**: 569 samples, 30 features
2. **Sonar**: 208 samples, 60 features  
3. **Ionosphere**: 351 samples, 34 features
4. **Diabetes (PIMA Indian)**: 768 samples, 8 features
5. **Cleveland (Heart Disease)**: 303 samples, 13 features
6. **Colon**: 62 samples, 2000 features

Each dataset is tested under 4 conditions:
- **Original**: Clean data
- **Noise**: Label and feature noise added
- **Outlier**: Outliers introduced
- **Both**: Both noise and outliers present

## Project Structure

```
Pin_FS_SVM/
├── src/
│   ├── models/                    # SVM model implementations
│   │   ├── pin_fs_svm.py         # Main proposed method
│   │   ├── milp1_svm.py          # MILP1 baseline
│   │   ├── l1_svm.py             # L1-SVM baseline
│   │   ├── l2_svm.py             # L2-SVM baseline
│   │   ├── pinball_svm.py        # Pinball SVM baseline
│   │   ├── fisher_svm.py         # Fisher-SVM baseline
│   │   └── rfe_svm.py            # RFE-SVM baseline
│   ├── utils/                     # Utility functions
│   │   ├── data_loader.py        # Dataset loading utilities
│   │   ├── preprocessing.py      # Data preprocessing
│   │   └── metrics.py            # Evaluation metrics
│   └── experiment/               # Experiment scripts
│       ├── run_experiment.py     # Main experiment runner
│       ├── hyperparameter_plot.py # Parameter visualization
│       └── results/              # Experiment results
│           └── wilcoxon/         # Statistical analysis
├── Dataset/                      # All datasets
│   └── Dataset/                  # Original and modified datasets
└── requirements.txt              # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Pin_FS_SVM
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Key Dependencies
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `scikit-learn`: Machine learning utilities
- `docplex`: IBM CPLEX optimization
- `matplotlib`: Visualization
- `scipy`: Statistical analysis
- `statsmodels`: Statistical modeling

## Usage

### Running Experiments

1. **Single Model Experiment**:
```python
from src.experiment.run_experiment import run_experiment
from src.models.pin_fs_svm import PinFSSVM

# Define model configuration
models_config = [{
    'class': PinFSSVM,
    'param_grid': {
        'B': [5, 10, 15],
        'C': [0.1, 1.0, 10.0],
        'tau': [0.1, 0.5, 0.9]
    }
}]

# Define dataset configuration
datasets_config = [{
    'dataset_name': 'wdbc',
    'dataset_types': ['original', 'noise', 'outlier', 'both']
}]

# Run experiment
run_experiment(models_config, datasets_config)
```

2. **Hyperparameter Visualization**:
```python
from src.experiment.hyperparameter_plot import main_compare_dataset_types
main_compare_dataset_types()
```

3. **Statistical Analysis**:
   - Open `src/experiment/results/wilcoxon/wilcoxon.ipynb`
   - Run all cells to perform Wilcoxon signed-rank tests
   - Results saved to `statistical_summary_all_datasets.xlsx`

### Model Usage Example

```python
from src.models.pin_fs_svm import PinFSSVM
from src.utils.data_loader import load_dataset
from sklearn.model_selection import train_test_split

# Load dataset
X, y = load_dataset('wdbc', 'original')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = PinFSSVM(B=10, C=1.0, tau=0.5)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Get selected features
selected_features = model.get_selected_features()
print(f"Selected features: {selected_features}")
```

## Evaluation Metrics

The project evaluates models using:
- **AUC (Area Under ROC Curve)**: Primary metric for model comparison
- **Accuracy**: Classification accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **G-Mean**: Geometric mean of sensitivity and specificity
- **Number of Selected Features**: Feature selection effectiveness
- **Training Time**: Computational efficiency

## Statistical Analysis

- **Cross-Validation**: 10-fold stratified cross-validation
- **Statistical Tests**: Wilcoxon signed-rank tests for pairwise comparisons
- **Multiple Comparisons**: Benjamini-Hochberg procedure for FDR control
- **Confidence Intervals**: Bootstrap confidence intervals for effect sizes

## Results

Experimental results are saved in:
- `src/experiment/results/`: Individual experiment results
- `src/experiment/results/wilcoxon/`: Statistical analysis results
- Excel files with detailed metrics for each model and dataset combination

## Key Findings

1. **Robustness**: Pin-FS-SVM shows improved robustness to noise and outliers compared to baseline methods
2. **Feature Selection**: Effectively selects relevant features while maintaining classification performance
3. **Computational Efficiency**: Reasonable training times compared to other optimization-based methods

## Publication

This work is part of research on robust feature selection methods for machine learning. Please cite appropriately if using this code in your research.

## License

[Add appropriate license information]

## Contributors

[Add contributor information]

## Contact

[Add contact information for questions or collaborations]

---

**Note**: This project requires IBM CPLEX solver for optimization-based models. Ensure proper licensing and installation of CPLEX before running experiments.
