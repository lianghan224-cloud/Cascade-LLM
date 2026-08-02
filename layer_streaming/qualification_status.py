"""One qualification vocabulary shared by KV reports and providers."""

from enum import Enum


class QualificationStatus(str, Enum):
    DECLARED = "declared"
    COMPILED = "compiled"
    SMOKE_PASSED = "smoke_passed"
    NUMERICALLY_QUALIFIED = "numerically_qualified"
    PERFORMANCE_QUALIFIED = "performance_qualified"
    PRODUCTION = "production"
    EXPERIMENTAL = "experimental"
    UNSUPPORTED = "unsupported"


QUALIFICATION_STATUSES = tuple(item.value for item in QualificationStatus)


def qualification_status(value):
    """Return a normalized status and reject the former ambiguous vocabulary."""

    if isinstance(value, QualificationStatus):
        return value.value
    try:
        return QualificationStatus(str(value)).value
    except ValueError:
        raise ValueError(
            "invalid qualification status {!r}; expected one of {}".format(
                value,
                ", ".join(QUALIFICATION_STATUSES),
            )
        )
