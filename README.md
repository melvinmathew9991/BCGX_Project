# PowerCo Customer Churn & Price-Sensitivity Analysis

A BCG X Data Science job-simulation project. PowerCo, a fictional SME gas/electricity
utility, is losing customers to competitors. This project tests the hypothesis that
**price sensitivity** drives churn, and builds a model to predict which customers are
at risk.

## Repository structure

```
BCGX_Project/
├── data/
│   ├── raw/                       # Client-supplied source data
│   │   ├── client_data.csv        # Customer profile, consumption, contract dates, churn label
│   │   └── price_data.csv         # Monthly tariff prices per customer (off-peak/peak/mid-peak)
│   └── processed/                 # Outputs of the notebooks, in pipeline order
│       ├── clean_data_after_eda.csv     # Cleaned client data (output of notebook 01)
│       └── data_for_predictions.csv     # Fully engineered, model-ready dataset (output of notebook 02)
├── notebooks/
│   ├── 01_eda_visualisation.ipynb       # Data profiling, distributions, churn-split analysis
│   ├── 02_feature_engineering.ipynb     # Price-sensitivity features, date/categorical encoding, skew correction
│   ├── 03_modeling.ipynb                # Random Forest churn classifier + feature importances
│   └── 04_findings_and_recommendation.ipynb  # Second modelling pass + hyperparameter search
├── docs/
│   ├── briefs/                    # Client/task briefs handed down for each stage
│   │   ├── task1_brief.pdf
│   │   ├── task2_brief.pdf
│   │   ├── task3_brief.pdf
│   │   └── data_description.pdf   # Data dictionary for client_data.csv / price_data.csv
│   └── deliverables/              # Work submitted back to the client
│       ├── task1_email_response.docx
│       └── executive_summary.pptx
├── requirements.txt
└── .gitignore
```

## Setup

A virtual environment (`.venv/`, gitignored) keeps dependencies isolated from the
system Python install:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (use `source .venv/bin/activate` on macOS/Linux)
pip install -r requirements.txt
python -m ipykernel install --user --name=bcgx_project --display-name "Python (BCGX_Project)"
```

Open the notebooks in Jupyter and select the **"Python (BCGX_Project)"** kernel, then
run them in order (01 → 04); each one reads its input relative to the `notebooks/`
folder (e.g. `../data/raw/client_data.csv`) and writes its output into
`../data/processed/`.

## Pipeline

1. **Business understanding** (`docs/briefs/task1_brief.pdf`, answered in
   `docs/deliverables/task1_email_response.docx`) — scoped the data and approach needed
   to test whether price sensitivity drives churn.
2. **EDA** (`notebooks/01_eda_visualisation.ipynb`) — profiles both raw datasets and
   compares churned vs. retained customers across consumption, margin, tenure, and
   sales channel.
3. **Feature engineering** (`notebooks/02_feature_engineering.ipynb`) — builds
   price-sensitivity features (Dec–Jan off-peak price deltas, cross-period price
   swings), tenure/contract-timing features, encodes categoricals, and corrects skew.
4. **Modelling** (`notebooks/03_modeling.ipynb`, `notebooks/04_findings_and_recommendation.ipynb`) —
   trains Random Forest classifiers, evaluates them, and ranks feature importance.

## Key result

Churn rate: **9.7%** across 14,606 customers. Price sensitivity is a **minor**
contributor to churn — consumption, forecasted consumption, and net margin are the
dominant drivers. Recommendation: target the 20% retention discount only at
high-value customers with a high predicted churn probability, rather than offering it
broadly.

**Caveat:** the delivered Random Forest has ~90% accuracy but only ~5% recall on
actual churners, a symptom of the ~90/10 class imbalance rather than a well-tuned
model — see notebook 04 for the (currently unexecuted) hyperparameter search intended
to address this before the recommendation above is actioned on live customers.
