# Machine Learning Engineering

Project containing implementations and experiments in machine learning engineering, with automated GitHub Actions workflow for model training and evaluation.

## Quick Start

### Local Development
1. Create virtual environment: `python -m venv .venv`
2. Activate: `source .venv/bin/activate` (Linux/Mac) or `.venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -e .`
4. Run preprocessing: `python preprocess_data.py`
5. Train model: `python train_torch_model.py`
6. Make predictions: `python predict_torch_model.py`

### GitHub Actions Workflow

The project includes automated training and evaluation pipeline configured in `.github/workflows/ml_training.yml`

#### Running the Workflow

1. Push code to GitHub repository
2. Go to **Actions** tab in GitHub
3. Select **ML Model Training & Evaluation** workflow
4. Click **Run workflow** (can be triggered manually via workflow_dispatch)
5. Provide optional parameters:
   - **num_epochs**: Number of training epochs (default: 10)
   - **batch_size**: Training batch size (default: 32)
   - **learning_rate**: Learning rate for optimizer (default: 0.001)

#### Workflow Jobs

**Train Model Job:**
- Checks out repository code
- Sets up Python 3.12 environment
- Installs project dependencies from pyproject.toml
- Preprocesses training data
- Trains PyTorch model with specified parameters
- Uploads artifacts (models, metrics, MLflow runs)

**Evaluate Model Job:**
- Runs after training job completes
- Downloads training artifacts from previous job
- Generates predictions on test data
- Computes evaluation metrics
- Uploads evaluation results

## Project Structure

```
├── .github/workflows/
│   └── ml_training.yml         # GitHub Actions workflow
├── artifacts/                  # Generated models and processed data
├── train_torch_model.py        # Model training script
├── predict_torch_model.py      # Model prediction script
├── preprocess_data.py          # Data preprocessing
├── compute_stats.py            # Evaluation metrics
├── pyproject.toml              # Dependencies
└── README.md                   # This file
```

## Dependencies

- Python 3.12+
- PyTorch 2.2.0+
- scikit-learn 1.8.0+
- pandas 3.0.1+
- numpy 2.4.3+
- MLflow 2.18.0+
- matplotlib 3.10.8+
