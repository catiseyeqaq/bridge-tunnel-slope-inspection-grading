# -*- coding: utf-8 -*-
"""Shared official API credential loading (never log secrets)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

from track4_contract_agent.assets import source_root

_PROJECT_ROOT = source_root() or Path.cwd()
_SECRETS_FILE = _PROJECT_ROOT / ".secrets" / "official_api.env"


def load_official_credentials() -> Tuple[str, str]:
    """Return (api_key, api_base). Key/base from env or private secrets file."""
    api_key = os.environ.get("TRACK4_OFFICIAL_API_KEY", "")
    api_base = os.environ.get("TRACK4_OFFICIAL_API_BASE", "")
    secrets_disabled = os.environ.get("TRACK4_DISABLE_SECRETS", "").lower() in {
        "1",
        "true",
        "yes",
    }
    if (not api_key or not api_base) and not secrets_disabled and _SECRETS_FILE.is_file():
        for line in _SECRETS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("TRACK4_OFFICIAL_API_KEY="):
                api_key = api_key or line.split("=", 1)[1].strip().strip('"').strip("'")
            elif line.startswith("TRACK4_OFFICIAL_API_BASE="):
                api_base = api_base or line.split("=", 1)[1].strip().strip('"').strip("'")
    return api_key, api_base.rstrip("/")
