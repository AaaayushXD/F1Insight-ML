#!/bin/bash
# F1Insight ML training: install deps (if needed) and run training.
set -e
cd "$(dirname "$0")"
if ! python -c "import sklearn" 2>/dev/null; then
  echo "Installing ML dependencies (scikit-learn, xgboost, joblib, matplotlib, seaborn)..."
  pip install -r requirements.txt
fi
echo "Running ML training (temporal split: train 2014-2021, val 2022-2023, test 2024-2025)..."
python -m app.ml.train
echo "Outputs: app/ml/outputs/ (models, preprocessor, evaluation_report.json)"
