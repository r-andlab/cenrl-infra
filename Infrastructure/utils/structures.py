from dataclasses import dataclass, asdict
from pydantic import BaseModel


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
