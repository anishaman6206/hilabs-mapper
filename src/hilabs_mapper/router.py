"""
Routing logic: decide which coding system to use
based on the entity type column.
"""

from typing import Literal

System = Literal["SNOMEDCT_US", "RXNORM"]

def route_system(entity_type: str) -> System:
    """
    Map entity type -> target coding system.
    - "Medicine" => RXNORM
    - "Diagnosis", "Procedure", "Lab" => SNOMEDCT_US
    - Default: SNOMEDCT_US
    """
    et = (entity_type or "").strip().lower()
    if et == "medicine":
        return "RXNORM"
    return "SNOMEDCT_US"
