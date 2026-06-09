"""Generate realistic synthetic SAP-style accounts receivable invoices."""

from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = [
    "invoice_id",
    "company_code",
    "customer_id",
    "customer_group",
    "region",
    "industry",
    "invoice_amount",
    "currency",
    "payment_terms_days",
    "days_since_invoice",
    "dispute_flag",
    "previous_late_payments",
    "open_items_count",
    "credit_score",
    "dunning_level",
    "has_purchase_order",
    "invoice_channel",
    "month",
    "target_paid_late",
]


def generate_synthetic_invoices(
    n_rows: int = 3_000,
    seed: int = 42,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Create a reproducible invoice dataset with a learnable late-payment target."""
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")

    rng = np.random.default_rng(seed)
    customer_ids = np.array([f"CUST{number:04d}" for number in range(1, 181)])

    df = pd.DataFrame(
        {
            "invoice_id": [f"INV{number:07d}" for number in range(1, n_rows + 1)],
            "company_code": rng.choice(
                ["1000", "1100", "2000", "3000"], n_rows, p=[0.38, 0.22, 0.25, 0.15]
            ),
            "customer_id": rng.choice(customer_ids, n_rows),
            "customer_group": rng.choice(
                ["Enterprise", "Mid-Market", "SMB", "Public Sector"],
                n_rows,
                p=[0.25, 0.32, 0.33, 0.10],
            ),
            "region": rng.choice(
                ["DACH", "EMEA", "Americas", "APJ"], n_rows, p=[0.32, 0.27, 0.26, 0.15]
            ),
            "industry": rng.choice(
                ["Manufacturing", "Retail", "Technology", "Healthcare", "Utilities"],
                n_rows,
            ),
            "invoice_amount": np.round(rng.lognormal(mean=8.35, sigma=1.05, size=n_rows), 2),
            "currency": rng.choice(["EUR", "USD", "GBP", "CHF"], n_rows, p=[0.55, 0.27, 0.10, 0.08]),
            "payment_terms_days": rng.choice(
                [14, 30, 45, 60, 90], n_rows, p=[0.08, 0.47, 0.20, 0.20, 0.05]
            ),
            "days_since_invoice": rng.integers(1, 121, n_rows),
            "dispute_flag": rng.binomial(1, 0.12, n_rows),
            "previous_late_payments": np.clip(rng.poisson(1.8, n_rows), 0, 10),
            "open_items_count": np.clip(rng.poisson(4.2, n_rows) + 1, 1, 25),
            "credit_score": np.clip(np.round(rng.normal(680, 75, n_rows)), 350, 850).astype(int),
            "dunning_level": rng.choice([0, 1, 2, 3], n_rows, p=[0.60, 0.24, 0.11, 0.05]),
            "has_purchase_order": rng.binomial(1, 0.78, n_rows),
            "invoice_channel": rng.choice(
                ["EDI", "Portal", "Email", "Paper"], n_rows, p=[0.34, 0.30, 0.29, 0.07]
            ),
            "month": rng.integers(1, 13, n_rows),
        }
    )

    log_amount = np.log1p(df["invoice_amount"])
    risk_score = (
        -3.0
        + 0.20 * (log_amount - log_amount.mean())
        + 1.25 * df["dispute_flag"]
        + 0.28 * df["previous_late_payments"]
        + 0.11 * df["open_items_count"]
        - 0.008 * (df["credit_score"] - 650)
        + 0.55 * df["dunning_level"]
        + 0.010 * (df["payment_terms_days"] - 30)
        + 0.65 * (1 - df["has_purchase_order"])
        + 0.25 * (df["days_since_invoice"] > df["payment_terms_days"]).astype(int)
        + 0.18 * df["month"].isin([1, 7, 12]).astype(int)
        + rng.normal(0, 0.45, n_rows)
    )
    late_probability = 1 / (1 + np.exp(-risk_score))
    df["target_paid_late"] = rng.binomial(1, late_probability)
    df = df[EXPECTED_COLUMNS]

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(destination, index=False)

    return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    path = project_root / "data" / "synthetic_invoices.csv"
    generated = generate_synthetic_invoices(output_path=path)
    print(f"Generated {len(generated):,} invoices at {path}")
    print(f"Late-payment rate: {generated['target_paid_late'].mean():.1%}")

