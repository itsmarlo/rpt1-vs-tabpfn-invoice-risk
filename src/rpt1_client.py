"""Integration boundary for SAP-RPT-1.

No SAP endpoint or payload is assumed here. Implement the exact API contract
only after confirming it against the target SAP AI Core environment.
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


class RPT1Client:
    """Configuration-aware placeholder for an SAP-RPT-1 integration."""

    REQUIRED_VARIABLES = (
        "SAP_RPT1_ENDPOINT",
        "SAP_RPT1_CLIENT_ID",
        "SAP_RPT1_CLIENT_SECRET",
        "SAP_RPT1_TOKEN_URL",
        "SAP_RPT1_RESOURCE_GROUP",
    )

    def __init__(self, env_path: str | Path | None = None) -> None:
        load_dotenv(dotenv_path=env_path)
        self.config = {name: os.getenv(name, "").strip() for name in self.REQUIRED_VARIABLES}

    def is_configured(self) -> bool:
        """Return true only when every required environment variable is present."""
        return all(self.config.values())

    def predict(
        self,
        context_df: pd.DataFrame,
        query_df: pd.DataFrame,
        target_column: str,
    ) -> pd.DataFrame:
        """Predict through SAP-RPT-1 once the environment-specific API is implemented."""
        if not self.is_configured():
            missing = [name for name, value in self.config.items() if not value]
            raise RuntimeError(
                "SAP-RPT-1 is not configured. Missing environment variables: "
                + ", ".join(missing)
            )

        # TODO: Authenticate with the configured SAP AI Core tenant and call
        # SAP-RPT-1 using its verified, environment-specific request schema.
        raise NotImplementedError(
            "SAP-RPT-1 credentials are configured, but the API call is intentionally "
            "left as a TODO until the exact SAP AI Core contract is known."
        )

