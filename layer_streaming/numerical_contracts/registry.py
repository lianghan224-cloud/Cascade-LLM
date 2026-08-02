"""Architecture- and ABI-specific numerical contract registry."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from ..qualification_status import qualification_status


@dataclass(frozen=True)
class NumericalContractKey:
    architecture: str
    provider_name: str
    provider_abi: int
    model_geometry_id: str
    weight_format: str
    activation_dtype: str
    scale_dtype: str
    physical_layout: str

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class NumericalContractRecord:
    key: NumericalContractKey
    contract_path: str
    qualification_status: str

    def __post_init__(self):
        object.__setattr__(
            self,
            "qualification_status",
            qualification_status(self.qualification_status),
        )


class NumericalContractRegistry:
    def __init__(self):
        self._contracts = {}

    def register(self, record, replace=False):
        if not isinstance(record, NumericalContractRecord):
            raise TypeError("record must be NumericalContractRecord")
        if record.key in self._contracts and not replace:
            raise ValueError("numerical contract key is already registered")
        self._contracts[record.key] = record
        return record

    def resolve(self, key):
        """Resolve only an exact key; contracts never cross architecture/ABI."""

        return self._contracts.get(key)

    def list_contracts(self):
        return tuple(
            self._contracts[key]
            for key in sorted(
                self._contracts,
                key=lambda item: (
                    item.architecture,
                    item.provider_name,
                    item.provider_abi,
                    item.model_geometry_id,
                    item.weight_format,
                    item.activation_dtype,
                    item.scale_dtype,
                    item.physical_layout,
                ),
            )
        )


def default_numerical_contract_registry(root=None):
    registry = NumericalContractRegistry()
    project_root = Path(root) if root else Path(__file__).resolve().parents[2]
    contract = project_root / "tests/fixtures/fused_w8a16_sm86_golden_v1.json"
    if contract.is_file():
        registry.register(
            NumericalContractRecord(
                key=NumericalContractKey(
                    architecture="sm86",
                    provider_name="cutlass_w8a16_sm86_abi2",
                    provider_abi=2,
                    model_geometry_id="llama31_8b",
                    weight_format="int8_symmetric_per_channel",
                    activation_dtype="bf16",
                    scale_dtype="bf16",
                    physical_layout="row_major",
                ),
                contract_path=str(contract),
                qualification_status="numerically_qualified",
            )
        )
    return registry
