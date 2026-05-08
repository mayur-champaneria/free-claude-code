"""Amazon Bedrock provider settings."""

from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict


class BedrockSettings(BaseModel):
    """Provider-specific Bedrock settings passed to the adapter constructor."""

    region: str = ""
    profile: str = ""
    access_key_id: str = ""
    secret_access_key: str = ""
    session_token: str = ""
    control_base_url: str = ""

    model_config = ConfigDict(extra="forbid")

    def resolved_region(self) -> str:
        """Return the AWS region for Bedrock runtime/control-plane requests."""
        return (
            self.region.strip()
            or os.getenv("AWS_REGION", "").strip()
            or os.getenv("AWS_DEFAULT_REGION", "").strip()
            or "us-east-1"
        )
