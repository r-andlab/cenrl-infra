from dataclasses import dataclass
from pydantic import BaseModel
from enum import Enum
from typing import List, Dict


# For pending measurements from API within the BatchUCB Model
@dataclass
class Pending:
    node_id: str  # action space node key
    arm_key: str
    arm_seq: List[str]
    blocked: bool = False
    reward: float = 0.0
    observed_value: float | None = None


# Represents Different States of Regional Nodes
class NodeState(Enum):
    IDLE = 1
    READY = 2
    AWAITING_MEASUREMENTS = 3
    DONE = 4

# Each Method Class represents different ways that the model can operate
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


# Classes for requests and responses from Go API -----------------

class TestResponseData(BaseModel):
    matches_template: bool
    start_time: str
    end_time: str

class LocationData(BaseModel):
    country_name: str
    country_code: str

class EvalPayload(BaseModel):
    vp: str
    service: str
    response: List[Dict]
    issue: str | None = None
    template: Dict | None = None

class TestPayload(BaseModel):
    vp: str
    location: LocationData
    service: str
    test_url: str
    response: List[TestResponseData]
    anomaly: bool
    controls_failed: bool
    stateful_block: bool
    tag: str | None = None


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
    services: List[str]
    vantage_point_predicate: str
    tag: Tag

@dataclass
class MeasurementResponse:
    target: str
    blocked: bool
