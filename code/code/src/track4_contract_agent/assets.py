"""Locate runtime assets in a source checkout or an installed distribution."""

from __future__ import annotations

import sys
from pathlib import Path


_DISTRIBUTION_SHARE = Path(sys.prefix) / "share" / "track4-contract-agent"


def source_root() -> Path | None:
    """Return the repository root when running from a source checkout."""

    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").is_file():
        return candidate
    return None


def asset_path(relative_path: str | Path) -> Path:
    """Resolve an asset from the checkout first, then the installed share tree."""

    relative = Path(relative_path)
    roots = [root for root in (source_root(), _DISTRIBUTION_SHARE) if root is not None]
    for root in roots:
        candidate = root / relative
        if candidate.exists():
            return candidate
    searched = ", ".join(str(root / relative) for root in roots)
    raise FileNotFoundError(f"Track4 runtime asset not found: {relative}; searched: {searched}")


def runtime_root() -> Path:
    """Return a stable root for manifests and local secret discovery."""

    return source_root() or _DISTRIBUTION_SHARE
