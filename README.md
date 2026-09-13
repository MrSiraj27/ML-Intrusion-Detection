# IDS ML Project

A machine-learning intrusion detection proof-of-concept built on the [UNSW-NB15](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15) dataset. It includes exploratory data analysis, preprocessing, Decision Tree / Random Forest classifiers, and a Streamlit dashboard for interactive training, evaluation, and prediction.

## Project structure

```
ids-ml-project/
  data/                          # datasets (gitignored — see below)
  models/                        # trained models + preprocessing artifacts (gitignored)
  notebooks/
    01_eda.ipynb                 # exploratory data analysis
    02_preprocessing.ipynb       # cleaning, encoding, scaling -> data/*_clean.csv
    03_model_training.ipynb      # trains Decision Tree + Random Forest
  dashboard.py                   # Streamlit app (upload, train, visualize, predict)
  requirements.txt
```

## 1. Setup

Requires Python 3.11+.

```bash
git clone <this-repo-url>
cd ids-ml-project

python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Verify the install:

```bash
python -c "import sklearn; print(sklearn.__version__)"
```

## 2. Get the data

The raw UNSW-NB15 CSVs are not committed to the repo (they're large). Download them with [kagglehub](https://github.com/Kaggle/kagglehub) — you'll need a free Kaggle account and an API token (`kaggle.json`, see [Kaggle's docs](https://www.kaggle.com/docs/api)):

```bash
python notebooks/download_data.py
```

This downloads `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` into `data/`.

## 3. Run the notebooks (in order)

```bash
jupyter lab
```

Then run, top to bottom:

1. **`notebooks/01_eda.ipynb`** — explores the raw data (class balance, attack categories, correlations).
2. **`notebooks/02_preprocessing.ipynb`** — cleans, encodes, and scales the data, writing `data/train_clean.csv`, `data/test_clean.csv`, and `models/preprocessing.joblib` (the fitted encoders/scaler, reused by the dashboard).
3. **`notebooks/03_model_training.ipynb`** — trains a Decision Tree and a Random Forest with 5-fold cross-validation and saves them to `models/decision_tree.joblib` and `models/random_forest.joblib`.

You can also execute a notebook non-interactively from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/02_preprocessing.ipynb
```

## 4. Run the dashboard

Once the notebooks above have been run at least once (so `data/*_clean.csv` and `models/*.joblib` exist):

```bash
streamlit run dashboard.py
```

Streamlit will print a local URL (default `http://localhost:8501`) — open it in your browser. The dashboard has four pages:

- **Dataset Upload/Explorer** — inspect the bundled clean datasets or upload your own CSV.
- **Model Training** — pick Decision Tree or Random Forest, tune hyperparameters, run 5-fold CV, and save the trained model.
- **Visualizations** — confusion matrix, ROC curve, and feature importance for any trained/saved model against the test set.
- **Predictions** — classify traffic as Normal/Attack via manual input or a batch CSV upload, using the saved models and preprocessing artifacts.

## Notes

- `data/` and `models/*.joblib` are gitignored since they're large, regenerable binary/data files — running the steps above recreates them locally.
- Re-running `02_preprocessing.ipynb` regenerates `models/preprocessing.joblib`; re-running `03_model_training.ipynb` regenerates the model files.
