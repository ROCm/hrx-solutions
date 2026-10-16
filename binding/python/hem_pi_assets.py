"""Read the fixed, reviewed Pi resources from either source or an installed wheel."""

from __future__ import annotations

import hashlib
from importlib.resources import files
from typing import Dict

from .hem_pi_contracts import EXTENSION_PATH


ASSET_NAMES = ("index.ts", "discovery.ts", "budgets.ts", "stream.ts",
               "profile.json", "package.json")


def bundled_assets() -> Dict[str, bytes]:
    resource = files(__package__).joinpath("pi_extension")
    return {EXTENSION_PATH + "/" + name: resource.joinpath(name).read_bytes()
            for name in ASSET_NAMES}


def asset_digests(assets: Dict[str, bytes]) -> Dict[str, str]:
    return {name: hashlib.sha256(data).hexdigest()
            for name, data in sorted(assets.items())}
