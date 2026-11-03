import pytest
from unittest.mock import MagicMock, patch
from Infrastructure.utils.structures import NodeState


@pytest.fixture
def mock_batch_ucb():
    """Create a mock BatchUCB model with the attributes RegionalNode expects."""
    model = MagicMock()
    model.measurements_per_episode = 100
    model.num_episodes = 1
    model.current_epoch_num = 0
    model.outfile = "/tmp/test_node.csv"
    model.output_directory = "/tmp"
    model.save = MagicMock()
    model.action_space = MagicMock()
    model.action_space.wake_up_all_nodes = MagicMock()
    model.can_step = MagicMock(return_value=True)
    return model


@pytest.fixture
def minimal_node(mock_batch_ucb, tmp_path):
    """Create a RegionalNode with a mocked model, bypassing full initialization."""
    from Infrastructure.main.node import RegionalNode

    with patch.object(RegionalNode, '__init__', lambda self, *a, **kw: None):
        node = RegionalNode.__new__(RegionalNode)

    node.country = "TestCountry"
    node.country_name_standard = "TestCountry"
    node.model = mock_batch_ucb
    node.state = NodeState.IDLE
    node.batch_size = 5
    node.in_flight = 0
    node.episode_stats = []
    node.episode_all_stats = []
    node.episode_idx = 1
    node.stat_df = None
    node.save_stats = True
    node.params = {"outfile_csv": str(tmp_path / "test.csv")}
    node.output_folder = str(tmp_path)
    return node
