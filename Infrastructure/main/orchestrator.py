import os

from Infrastructure.examples.vps import vantage_points
from pathlib import Path


from models.ucb.ucb_naive import UCBNaiveParserOptions
from typing import Dict, List, Any
from Infrastructure.apis.funneler import HyperQuackAPI
from Infrastructure.models.BatchUCB import BatchUCB
from Infrastructure.main.node import RegionalNode
from Infrastructure.utils.structures import NodeState
from time import sleep


class Orchestrator:
    def __init__(
        self,
        params: Dict,
        vantage_point_map: Dict[str, str],
        go_api_endpoint: str,
        services: List[str],
        previous_values_folder: str = None
    ):
        self.params = params
        self.output_folder = params.get("outfile_csv", None)
        if previous_values_folder:
            path = Path(previous_values_folder)
            if not path.exists() or not path.is_dir():
                previous_values_folder = None
        if self.output_folder:
            os.makedirs(os.path.dirname(self.output_folder), exist_ok=True)
        self.agents: dict[str, RegionalNode] = {
            key: RegionalNode(
                params=self.params,
                country_name=key,
                vps=[value],
                model_klass=BatchUCB,
                output_folder=self.output_folder,
                action_space_folder=previous_values_folder

            )
            for key, value in vantage_point_map.items()
        }
        self.api = HyperQuackAPI(go_api_endpoint)
        self.api.update_vps(new_vps=list(vantage_point_map.values()), services=services)
        self.services = services

    def run_forever(self):
        while True:
            if len(self.agents) == 0:
                print("All countries completed, exiting program...")
                return
            for country, n in self.agents.items():
                if n.state == NodeState.IDLE or n.state == NodeState.READY:
                    targets = n.maybe_request_more()
                    if targets is not None:
                        # print(f"{country} requesting {targets}\n")
                        self.funnel_measurements(country, targets)
                    #if n.state == NodeState.AWAITING_MEASUREMENTS:
                    n.maybe_update_model(self.api.try_get_results(country))
                if n.state == NodeState.DONE:
                    print(
                        f"{country} had completed {n.model.num_episodes} and is finished."
                    )
                    self.agents.pop(country)
                    break
            sleep(0.5)

    def funnel_measurements(self, country, targets):
        vps = self.agents[country].get_vps()
        self.api.schedule_measurements(
            vps=list(vps), services=self.services, targets=targets
        )

    def add_vp(self, country, vp):
        if country not in self.agents:
            self.agents[country] = RegionalNode(
                params=self.params, country=country, vps=set(), model_klass=BatchUCB
            )
        self.agents[country].active_vps.add(vp)

    def deactivate_vp(self, country, vp):
        if country not in self.agents:
            raise KeyError(f"Country {country} not present in network")
        self.agents[country].deactivate_vp(vp)

    def delete_vp(self, country, vp):
        if country not in self.agents:
            raise KeyError(f"Country {country} not present in network")
        self.agents[country].delete_vp(vp)


if __name__ == "__main__":
    parser = UCBNaiveParserOptions()
    params = parser.parse()
    m = Orchestrator(params, vantage_points, "http://127.0.0.1:8888", ["https"], )  # previous_values_folder="outputs/outtest5")
    m.run_forever()
    # python3 Infrastructure/main/orchestrator.py -E 1 -m 1000 -v -f "categories" -a inputs/tranco/tranco_categories_subdomain_tld_entities_top10k.csv -f "categories" -s 0.0 -c 0.03 -V 0.0 -o outputs/outtest
