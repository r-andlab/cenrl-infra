import typing

from models.base.model import run_multiprocessing
from models.contextual_bandit.linucb import LinUCBActionSpace
from models.contextual_bandit.linucb_shared import LinUCBShared, LinUCBSharedParserOptions


class LinUCBSharedAPI(LinUCBShared):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #  Add initials

    def take_measurement(self, selected_target) -> typing.Tuple[float, bool]:
        raise NotImplementedError()

    # The below methods are necessary for controlled environments, ignore them.
    def parse_block_list(self):
        pass

    def get_blocklist_coverage(self) -> float:
        return 0

    def set_blocklist_unique_counts_based_on_action_space(self):
        pass

    def init_blockers(self):
        pass

    def update_blocklist_target_found(self, target_found: str):
        pass

if __name__ == "__main__":
    parser = LinUCBSharedParserOptions()
    params = parser.parse()
    run_multiprocessing(LinUCBSharedAPI, params,
                        addition_model_run_kwargs={
                            "action_space_klass": LinUCBActionSpace,
                        })
