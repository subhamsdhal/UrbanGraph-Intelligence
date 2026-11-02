"""Narrow, reviewed compatibility for recorded UG-003R source provenance.

This is not a general permission to ignore source hashes. Legacy plans still need
byte integrity, identical dependencies and complete semantic/scaler reconstruction.
"""

from urbangraph.datasets.serialization import object_dict
from urbangraph.evaluation.dataset import identity

UG003R_SOURCE_INVENTORY = "735b07a88b06b0181d6bc07c16917c36343089b8849ae2721b1332ac416a0507"
UG003S_SOURCE_INVENTORY = "40c971b9ec1ee7495f21e5d4b6db27bedca9719c64996c1550b00e21b34c9943"
LEGACY_PROTOCOLS = (
    "dcrnn-sample-first-v1",
    "strict-chronological-v1",
    "strict-rolling-origin-v1",
)


def compatible_provenance(recorded: object, current: object, protocol: str) -> bool:
    old, now = object_dict(recorded), object_dict(current)
    if old == now:
        return True
    return (
        set(old) == {"source_sha256", "dependencies"}
        and (
            (
                identity(object_dict(old["source_sha256"])) == UG003R_SOURCE_INVENTORY
                and protocol in LEGACY_PROTOCOLS
            )
            or (
                identity(object_dict(old["source_sha256"])) == UG003S_SOURCE_INVENTORY
                and protocol in (*LEGACY_PROTOCOLS, "dcrnn-positional-v1")
            )
        )
        and old["dependencies"] == now.get("dependencies")
    )
