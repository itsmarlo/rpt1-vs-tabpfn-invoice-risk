# RPT-1 vs TabPFN: Invoice Payment Risk

An interview-ready machine-learning project that predicts whether an accounts
receivable invoice will be paid late using realistic synthetic SAP-style data.

## Business problem

Late customer payments increase working-capital requirements and collections
costs. An early risk score can help accounts receivable teams prioritize
disputes, dunning activity, and customer outreach before an invoice becomes
seriously overdue.

This project is SAP-relevant because the features mirror common ERP and
accounts receivable concepts: company codes, customer groups, payment terms,
open items, disputes, purchase orders, credit scores, and dunning levels.

## Model comparison

| Model | Role | Runs locally? |
|---|---|---|
| Logistic Regression | Interpretable classic baseline | Yes |
| Random Forest | Nonlinear classic baseline | Yes |
| Gradient Boosting | Sequential tree ensemble baseline | Yes |
| TabPFN | Executable tabular foundation-model baseline | Yes, when its package and model weights are available |
| SAP-RPT-1 OSS | Hugging Face research checkpoint | Optionally, with gated access and substantial hardware |
| SAP-RPT-1 AI Core | Enterprise integration boundary | Only with SAP infrastructure, credentials, and an implemented API contract |

This project compares a general tabular foundation model, TabPFN, with an
SAP-specific relational foundation model concept, SAP-RPT-1. Since SAP-RPT-1
access requires SAP infrastructure and credentials, this repository provides a
clean integration interface and keeps the full local experiment runnable with
TabPFN and classic ML baselines.

## Setup

Python 3.10 or newer is recommended.

```bash
cd rpt1-vs-tabpfn-invoice-risk
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TabPFN may download model weights on first use and can have platform-specific
PyTorch requirements. If it is unavailable, the experiment still completes and
records TabPFN as skipped. TabPFN uses low-memory mode, small inference batches,
and progressively smaller retry profiles to reduce peak memory use on Apple
Silicon.

### Optional SAP-RPT-1 OSS

SAP publishes a gated, Apache-2.0 model repository at
[`SAP/sap-rpt-1-oss`](https://huggingface.co/SAP/sap-rpt-1-oss). Its checkpoints
are intended for research use and are not served by a Hugging Face Inference
Provider.

1. Sign in to Hugging Face and accept the model repository conditions.
2. Authenticate locally with a read token:

```bash
hf auth login
```

3. Install the optional SAP package:

```bash
pip install -r requirements-rpt1-oss.txt
```

The normal experiment command then attempts the OSS model automatically:

```bash
python -m src.run_experiment
```

The local configuration uses a 256-row context and bagging factor of 1. This is
far below SAP's best-performance recommendation but is more realistic for a
developer machine. Download, gated-access, dependency, and memory failures are
reported as skipped results without stopping the other models.

## Run

Run the complete experiment from the project directory:

```bash
python -m src.run_experiment
```

Run tests:

```bash
pytest
```

Open the optional exploration notebook:

```bash
jupyter notebook notebooks/01_exploration.ipynb
```

The runner creates the synthetic dataset when it is missing, uses one
stratified train/test split for every executable model, and fits preprocessing
only on the training partition. The unique `invoice_id` is retained for
traceability but excluded from model features. The synthetic `customer_id` is
also excluded because one-hot encoding an arbitrary identifier adds substantial
dimensionality without useful signal.

## Expected output

The `results/` directory contains:

- `metrics_comparison.csv`: metrics and execution status for every model
- `predictions_logistic_regression.csv`: row-level baseline predictions
- `predictions_random_forest.csv`: row-level baseline predictions
- `predictions_gradient_boosting.csv`: row-level baseline predictions
- `predictions_tabpfn.csv`: TabPFN predictions, or an empty schema if skipped
- `predictions_rpt1_oss.csv`: local RPT-1 OSS predictions, or an empty schema
- `confusion_matrix_tabpfn.png`: TabPFN matrix, or a clear skipped placeholder

Metrics include accuracy, precision, recall, F1, ROC AUC, and measured
prediction time. In a collections workflow, recall indicates how many truly
late invoices were found; precision indicates how much analyst attention would
be spent on genuinely risky invoices. ROC AUC is useful for comparing ranking
quality independently of a single decision threshold.

## Optional SAP-RPT-1 configuration

Copy `.env.example` to `.env` and populate it only with credentials for an
authorized SAP environment. `.env` is ignored by Git.

The AI Core integration intentionally does not invent an SAP API payload or
make a fake request. `RPT1Client` validates configuration and exposes a clean
`predict` interface, while the environment-specific authentication and
inference call remains explicitly marked as a TODO. Without credentials, the
final comparison contains a `SAP-RPT-1 AI Core` row with null metrics and
`skipped_if_not_configured` status.

## Limitations

- The dataset is synthetic and does not reproduce customer behavior, data
  quality issues, leakage patterns, or process variation from a real SAP system.
- The target is generated from a known formula, so results are illustrative
  rather than evidence of production performance.
- One-hot encoding does not model relationships between customers, invoices,
  payments, and company entities as a relational model could.
- TabPFN training is capped and may be reduced automatically after a memory
  failure, so its metrics may use less training data than the classic baselines.
- SAP recommends an 80 GB GPU for the strongest RPT-1 OSS configuration. The
  deliberately small local context may still exceed laptop memory or perform
  below the recommended configuration.
- No threshold tuning, probability calibration, temporal validation, fairness
  analysis, or cost-sensitive decision policy is included.
- SAP-RPT-1 inference requires a verified SAP AI Core API contract and access.

## Next steps

1. Replace the synthetic CSV with de-identified, time-stamped ERP extracts.
2. Use a temporal holdout and monitor drift by company code and customer group.
3. Add calibration and choose a threshold based on collections capacity and cost.
4. Add relational context such as customer history, payment events, and disputes.
5. Implement `RPT1Client.predict` against verified SAP documentation and compare
   all models on the same held-out invoice set.

## Interview talking points

- Why a simple logistic model remains valuable for explainability and governance.
- Why a shared split and leakage-safe preprocessing matter for a fair comparison.
- How foundation models change the tradeoff between task-specific training and
  in-context prediction.
- Why `invoice_id` is excluded and why real deployments need temporal validation.
- How the optional integration boundary keeps credentials and cloud access out
  of the local developer workflow.
- Which metric should drive collections decisions and how business costs affect
  the operating threshold.
