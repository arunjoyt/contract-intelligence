"""Loads and validates the demo-data YAML fixture.

Validation happens once at load time so a fixture-authoring mistake fails loudly and
immediately, rather than silently causing a Contract idempotency check to skip a distinct
entry or an ingest to choke on a bad status value later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

VALID_STATUSES = {"Unsigned", "Active", "Inactive"}
VALID_DOCSTATUSES = {0, 1, 2}


class FixtureValidationError(Exception):
    """Raised when the demo data fixture fails validation."""


@dataclass
class Supplier:
    name: str
    supplier_group: str


@dataclass
class TermsAndConditions:
    title: str
    terms_html: str
    disabled: bool = False


@dataclass
class Contract:
    key: str
    party_type: str
    party_name: str
    status: str
    docstatus: int
    start_date: str
    end_date: str
    contract_terms_html: str
    is_signed: bool = False
    pdf_attachment: dict[str, Any] | None = None


@dataclass
class DemoData:
    suppliers: list[Supplier] = field(default_factory=list)
    terms_and_conditions: list[TermsAndConditions] = field(default_factory=list)
    contracts: list[Contract] = field(default_factory=list)


def load_fixture(path: Path) -> DemoData:
    """Load and validate `path` (a YAML file matching the DemoData shape)."""
    raw = yaml.safe_load(path.read_text())
    suppliers = [Supplier(**s) for s in raw.get("suppliers", [])]
    terms_and_conditions = [
        TermsAndConditions(**t) for t in raw.get("terms_and_conditions", [])
    ]
    contracts = [Contract(**c) for c in raw.get("contracts", [])]
    data = DemoData(
        suppliers=suppliers, terms_and_conditions=terms_and_conditions, contracts=contracts
    )
    _validate(data)
    return data


def _validate(data: DemoData) -> None:
    errors: list[str] = []

    for supplier in data.suppliers:
        if not supplier.supplier_group:
            errors.append(f"Supplier {supplier.name!r} has a null/empty supplier_group")

    supplier_names = {s.name for s in data.suppliers}

    natural_keys: dict[tuple[str, str, str], str] = {}
    for contract in data.contracts:
        if contract.status not in VALID_STATUSES:
            errors.append(
                f"Contract {contract.key!r} has invalid status {contract.status!r} "
                f"(expected one of {sorted(VALID_STATUSES)})"
            )
        if contract.docstatus not in VALID_DOCSTATUSES:
            errors.append(
                f"Contract {contract.key!r} has invalid docstatus {contract.docstatus!r} "
                f"(expected one of {sorted(VALID_DOCSTATUSES)})"
            )
        if contract.party_type == "Supplier" and contract.party_name not in supplier_names:
            errors.append(
                f"Contract {contract.key!r} references supplier {contract.party_name!r} "
                "which is not defined in this fixture's suppliers list"
            )

        natural_key = (contract.party_name, contract.start_date, contract.end_date)
        if natural_key in natural_keys:
            errors.append(
                f"Contract {contract.key!r} has the same (party_name, start_date, end_date) "
                f"as {natural_keys[natural_key]!r} — natural key must be unique: {natural_key}"
            )
        else:
            natural_keys[natural_key] = contract.key

    if errors:
        raise FixtureValidationError(
            "Demo data fixture failed validation:\n" + "\n".join(f"  - {e}" for e in errors)
        )
