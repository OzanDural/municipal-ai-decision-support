from typing import Any, Literal

from pydantic import BaseModel


class ServiceAreaProperties(
    BaseModel
):
    code: str
    name: str
    source: str


class ServiceAreaFeatureResponse(
    BaseModel
):
    type: Literal["Feature"] = "Feature"

    properties: ServiceAreaProperties

    geometry: dict[str, Any]