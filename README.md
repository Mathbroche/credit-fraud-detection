# Credit Fraud Detection

Anomaly detection system for credit card fraud using a PyTorch Autoencoder trained exclusively on normal transactions.

## Results

AUC-ROC : 0.95
Best F1  : 0.40
Recall   : 0.84

## Dataset

Kaggle Credit Card Fraud Detection — 284,807 transactions, 492 frauds (0.17%).
Features V1-V28 are anonymized via PCA by the original authors.

## Architecture

Autoencoder trained on normal transactions only.
The reconstruction error is used as an anomaly score.
Transactions with error above a learned threshold are flagged as fraud.

Input (30) -> Encoder -> Latent space (10) -> Decoder -> Output (30)

## Project Structure

credit-fraud-detection/
    config.py
    main.py
    src/
        data.py
        dataset.py
        model.py
        train.py
        evaluate.py
    notebooks/
        exploration.ipynb

## Setup

git clone https://github.com/mathbroche/credit-fraud-detection
cd credit-fraud-detection
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Download the dataset from Kaggle and place creditcard.csv in the data/ folder.
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip

python main.py

## Key Concepts

Imbalanced data — 0.17% fraud rate, accuracy is irrelevant
Anomaly detection — model learns normal behavior, flags deviations
Threshold tuning — automatic threshold selection to maximize F1
AUC-ROC — main evaluation metric, independent of threshold

## Limitations

V1-V28 features are anonymized and uninterpretable.
The model cannot be deployed on real transactions without the original PCA matrix.
Threshold is optimized on the test set — slightly optimistic F1.
