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
        self.output_folder = params.get("output_directory", None)
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
        self.api = HyperQuackAPI(go_api_endpoint, debug=False)
        self.api.update_vps(new_vps=list(vantage_point_map.values()), services=services)
        self.services = services

    def tick(self) -> None:
        finished_nodes: List[str] = []

        for country, n in self.agents.items():
            # Attempt to read new results from funneler
            results = self.api.try_get_results(country)
            if results:
                n.maybe_update_model(results)

            # Remove node if it has completed
            if n.state is NodeState.DONE:
                print(
                    f"{country} had completed {n.model.num_episodes} and is finished."
                )
                finished_nodes.append(country)
                continue
            
            # Try to get more targets to request
            vps = list(n.get_vps())
            if vps:
                targets = n.maybe_request_more()
                if not targets:
                    continue
                self.api.schedule_measurements(
                    vps=vps, services=self.services, targets=targets, country=country
                )

        for country in finished_nodes:
            self.agents.pop(country, None)

    def run_forever(self):
        while self.agents:
            self.tick()
            sleep(0.5)
        print("All countries completed, exiting program...")

    def funnel_measurements(self, country, targets):
        vps = self.agents[country].get_vps()
        self.api.schedule_measurements(
            vps=list(vps), services=self.services, targets=targets
        )

    def add_vp(self, country, vp):
        if country not in self.agents:
            self.agents[country] = RegionalNode(
                params=self.params, country_name=country, vps=set(), model_klass=BatchUCB
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


class OrchestrationParser(UCBNaiveParserOptions):

    def set_params(self, args):
        if args.outfile[-1] != "/":
            args.outfile+="/"
        super().set_params(args)

    # def parse(self):
    #     self.add_arguments()
    #     args = self.parser.parse_args()
    #     self.set_params(args)

    #     return self.params
    

if __name__ == "__main__":
    parser = OrchestrationParser()
    params = parser.parse()
    m = Orchestrator(params, vantage_points, "http://127.0.0.1:8888", ["https"], )  # previous_values_folder="outputs/outtest5")
    m.run_forever()
    # python3 Infrastructure/main/orchestrator.py -E 1 -m 1000 -v -f "categories" -a inputs/tranco/tranco_categories_subdomain_tld_entities_top10k.csv -f "categories" -s 0.0 -c 0.03 -V 0.0 --output-folder outputs/outtest
