from dataclasses import dataclass
from pydantic import BaseModel
from enum import Enum


class NodeState(Enum):
    IDLE = 1
    READY = 2
    AWAITING_MEASUREMENTS = 3
    DONE = 4


class BatchSizeMethod(Enum):
    CONSTANT_VAL = 1
    VARY_ON_SUCCESS = 2
    NESTED_RL = 3


class BatchSelectionMethod(Enum):
    TOP_K_FROM_ARM = 1
    UNIFORM_SPREAD = 2
    WEIGHTED_SPREAD = 3

class PropagationMethod(Enum):
    ON_RECEIPT = 1
    IN_ORDER = 2

class ResponseData(BaseModel):
    matches_template: bool
    start_time: str
    end_time: str


class LocationData(BaseModel):
    country_name: str
    country_code: str


class RequestPayload(BaseModel):
    vp: str
    location: LocationData
    service: str
    test_url: str
    response: list[ResponseData]
    anomaly: bool
    controls_failed: bool
    stateful_block: bool


@dataclass
class Tag:
    tag: str
    result_output_file: str
    eval_output_file: str
    mongo_uri: str = None
    database: str = None
    result_collection: str = None
    eval_collection: str = None


@dataclass
class Job:
    keyword: str
    services: list[str]
    vantage_point_predicate: str
    tag: Tag

@dataclass
class MeasurementResponse:
    target: str
    blocked: bool
