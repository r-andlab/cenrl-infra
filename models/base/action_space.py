import collections
import copy
import os
import random
import time
import typing
from itertools import islice

import networkx as nx
import numpy as np
import pandas as pd
from networkx.classes.reportviews import NodeView

from common.utils import TARGET_FEATURE__DOMAIN, TARGET_FEATURE__SERVICE_IP, SERVER_FEATURE_ORDER

NAME = "name"
LABEL = "label"
KEY = "key"
ACTION_ATTEMPTS = "action_attempts"
SLEEPING = "sleeping"
Q_VALUE = "q_value"
AVG_REWARD = "avg_reward"
MIN_Q_VALUE = -9999999
DEFAULT_Q_VALUE = 0
EXPLORED = "explored"
IS_TARGET_NODE = "is_target_node"
PARENTS = "parents"
RANK = "rank"

ROOT_KEY = "action_space_root"

# features
NODE_TYPE_KEY = "type"
NODE_TYPE__CATEGORY = "category"
NODE_TYPE__TLD = "tld"
NODE_TYPE__ENTITY = "entity"
NODE_TYPE__RANK_BIN = "bin"
NODE_TYPE__UNKNOWN = "unknown"

POSSIBLE_NODE_FEATURES = {NODE_TYPE__CATEGORY: 1,
                          NODE_TYPE__TLD: 1,
                          NODE_TYPE__ENTITY: 1,
                          NODE_TYPE__RANK_BIN: 1,
                          NODE_TYPE__UNKNOWN: 1}

POSSIBLE_TARGET_FEATURES = {TARGET_FEATURE__DOMAIN: 1, TARGET_FEATURE__SERVICE_IP: 1}


def create_default_node_attributes(
        node_key: str,
        node_name: str,
        node_type: str,
        default_q_value: float = DEFAULT_Q_VALUE,
        is_target_node: bool = False,
        sleeping: bool = False,
        explored: bool = False,
        action_attempts: int = 0,
        rank: int = -1,
        others: dict = None) -> dict:
    attributes = dict()
    # used to retrieve the node
    attributes[KEY] = node_key
    # name may not be the same as key
    attributes[NAME] = node_name
    attributes[NODE_TYPE_KEY] = node_type
    attributes[ACTION_ATTEMPTS] = action_attempts
    attributes[Q_VALUE] = default_q_value
    attributes[AVG_REWARD] = default_q_value
    attributes[SLEEPING] = sleeping
    attributes[EXPLORED] = explored
    attributes[IS_TARGET_NODE] = is_target_node
    attributes[PARENTS] = []
    attributes[RANK] = rank
    if others:
        for k, v in others.items():
            if k:
                attributes[k] = v if v is not None else ""
    return attributes


def add_root(g: nx.DiGraph) -> str:
    # add the root
    add_node(g, ROOT_KEY, create_default_node_attributes(ROOT_KEY,
                                                         ROOT_KEY,
                                                         ROOT_KEY))
    return ROOT_KEY


def add_node(g: nx.DiGraph,
             node: str,
             node_attributes: dict,
             parent: str = None) -> bool:
    # add root to parent or default to root
    if not g.has_node(node):
        g.add_nodes_from([(node, node_attributes)])
        if parent:
            g.add_edge(parent, node)
        return True
    else:
        # node already exists but edge may not
        if parent and not g.has_edge(parent, node):
            # print(f"existing parents: {list(g.predecessors(node))} --> {node}")
            # print(f"{node} already exists but added edge: {parent} --> {node}")
            g.add_edge(parent, node)
    return False


def is_active_leaf_node(graph: nx.DiGraph, node: str, node_type: str) -> bool:
    n_data = graph.nodes[node]
    return graph.out_degree(node) == 0 and \
        n_data[NODE_TYPE_KEY] == node_type and \
        not n_data[SLEEPING]


class ActionSpaceBase:

    def __init__(self, output_directory: str,
                 df: pd.DataFrame,
                 features: typing.List,
                 target_feature: str,
                 default_q_value: float = DEFAULT_Q_VALUE,
                 multiple_parents: bool = False,
                 action_value_file: str = None):
        self.output_directory = output_directory
        self.default_q_value = default_q_value
        self.multiple_parents = multiple_parents
        self.features = features
        self.target_feature = target_feature

        # default_q_value_from_file is a arm name -> default q value read from a given action value file
        self._default_q_value_from_file = dict()
        self.action_value_file = action_value_file
        self.read_action_value_file()

        self._df = df

        # TODO uncomment if want to test a portion of the action space
        #SAMPLE_ACTION_SPACE = 10000
        #if len(self._df) > SAMPLE_ACTION_SPACE:
        #    self._df = self._df.sample(SAMPLE_ACTION_SPACE, random_state=40)
        #    print(f"Warning: building a {SAMPLE_ACTION_SPACE} sample of the action space only")

        self._graph: nx.DiGraph = self.build_graph()
        self.save()

    def read_action_value_file(self):
        if self.action_value_file is not None:
            action_value_df = pd.read_csv(self.action_value_file)
            required_columns = {'episode', 'time', 'action', 'q_value'}
            assert required_columns.issubset(
                action_value_df.columns), f"action value file is missing required columns: {required_columns - set(action_value_df.columns)}"

            # sort then get last occurring q_value of each action, and average
            action_value_df = action_value_df.sort_values(by=['episode', 'time'])
            last_occurrences = action_value_df.groupby(['episode', 'action']).last().reset_index()
            action_averages = last_occurrences.groupby('action')[['q_value']].mean().reset_index()
            # convert to dict to use
            self._default_q_value_from_file = action_averages.set_index('action')['q_value'].to_dict()
            print(f"Built default q values from file {self.action_value_file}, {action_averages.head()}")

    @classmethod
    def get_classname(cls):
        return cls.__name__

    def reset(self):
        self._graph = self.build_graph()

    def get_default_q_value(self, node_key: str = None) -> typing.Tuple[float, bool]:

        _default_q_value = self.default_q_value
        match_found = False
        if node_key in self._default_q_value_from_file:
            _default_q_value = self._default_q_value_from_file[node_key]
            print(f"Using default q_value found from file {node_key}: {_default_q_value}")
            match_found = True

        return _default_q_value, match_found

    def build_graph(self) -> nx.DiGraph:
        print(f"Action space Process {os.getpid()} - Building action space using root > {' > '.join(self.features)} > {self.target_feature}")
        before_build_time = time.time()
        g = nx.DiGraph()
        root = add_root(g)

        default_q_value_match_count = 0

        # build first layer
        feature_key = self.features[0]
        feature_values = self._df[feature_key].unique().tolist()
        for feature_value in feature_values:
            _node_key = feature_key + " " + str(feature_value)
            _default_q_value, _default_q_value_match = self.get_default_q_value(node_key=_node_key)
            if _default_q_value_match:
                default_q_value_match_count += 1

            add_node(g,
                     _node_key,
                     node_attributes=create_default_node_attributes(_node_key,
                                                                    feature_value,
                                                                    feature_key,
                                                                    default_q_value=_default_q_value),
                     parent=root)

        # build subsequent layers, which will also build the target feature
        subsequent_layers = self.features[1:] + [self.target_feature]
        for feature_key in subsequent_layers:
            is_target_layer = feature_key == self.target_feature

            leaves = [v for v, d in g.out_degree() if d == 0 and v != root]
            # simple paths are from root to leaves with no repeat nodes
            paths = nx.all_simple_paths(g, root, leaves)
            # for each path, filter the df to those feature values
            for p in paths:
                _df_tmp = self._df.copy()
                # skip the root in the path and filter data based on the current path

                for n in p[1:]:
                    n_data = g.nodes.get(n)
                    node_feature_value = n_data[NAME]
                    node_feature_key = n_data[NODE_TYPE_KEY]
                    _df_tmp = _df_tmp[_df_tmp[node_feature_key] == node_feature_value]

                # now filter rows using the current feature_key, if none returns, then no new nodes will be added
                feature_values = _df_tmp[feature_key].unique().tolist()
                # parent of these new nodes will be the last in the path
                _parent = p[-1]
                for feature_value in feature_values:
                    _node_key = feature_key + " " + str(feature_value)
                    # if a node cannot have multiple parents, make it tie to _parent by having a unique name
                    if not self.multiple_parents and not is_target_layer:
                        _node_key = _parent + " > " + _node_key

                    rank = -1
                    if is_target_layer and RANK in _df_tmp.columns:
                        rank = _df_tmp[_df_tmp[feature_key] == feature_value][RANK].iloc[0]

                    other_target_node_attributes = dict()
                    if is_target_layer and self.target_feature == TARGET_FEATURE__SERVICE_IP:
                        for target_attr in SERVER_FEATURE_ORDER:
                            target_attr_val = None
                            if is_target_layer and target_attr in _df_tmp.columns:
                                target_attr_val = _df_tmp[_df_tmp[feature_key] == feature_value][target_attr].iloc[0]
                            other_target_node_attributes[target_attr] = target_attr_val
                        #print(f"{_node_key}: {other_target_node_attributes}")
                    _default_q_value, _default_q_value_match = self.get_default_q_value(node_key=_node_key)
                    if _default_q_value_match:
                        default_q_value_match_count += 1

                    add_node(g,
                             _node_key,
                             node_attributes=create_default_node_attributes(
                                 _node_key,
                                 feature_value,
                                 feature_key,
                                 default_q_value=_default_q_value,
                                 is_target_node=is_target_layer,
                                 rank=rank,
                                 others=other_target_node_attributes),
                             parent=_parent)
        if self.action_value_file is not None and default_q_value_match_count == 0:
            print("WARNING: action_value_file was given but was not used during building of action space")
        else:
            print(f"values from action_value_file was used : {default_q_value_match_count} times")

        print(f"Action space Process {os.getpid()} - Done building action space, it took: {int(time.time() - before_build_time)} seconds")
        return g

    def save(self) -> bool:
        print(f"Action space Process {os.getpid()} - Number of nontarget nodes: {self.get_number_of_nontarget_nodes()}")
        print(f"Action space Process {os.getpid()} - Number of target nodes: {self.get_number_of_target_nodes()}")

        # connect all orphans and remove parent attributes before saving
        for n, n_data in self._graph.nodes(data=True):
            if n_data[PARENTS]:
                self.reconnect_to_parents(n)

            n_data[PARENTS] = ""

        graph_graphml_file = self.output_directory + os.sep + f"action_space.graphml"
        nx.write_graphml(self._graph, graph_graphml_file)

        # reset all parents to []
        for n, n_data in self._graph.nodes(data=True):
            n_data[PARENTS] = []

        return True

    def checkpoint_save(self, path: str) -> bool:
        """Write GraphML snapshot to path without mutating the live graph.

        Unlike save(), this method deep-copies the graph before reconnecting
        orphaned sleeping nodes and clearing PARENTS lists, so the in-memory
        graph used by the running UCB algorithm is never modified. Writes
        atomically via .tmp + os.replace to prevent corrupted state files
        on crash mid-write (POSIX rename semantics).
        """
        g_copy = copy.deepcopy(self._graph)

        # Reconnect sleeping/orphaned nodes on the COPY only, then convert
        # PARENTS list to "" because GraphML cannot serialize list values.
        for n, n_data in g_copy.nodes(data=True):
            parents = n_data.get(PARENTS, [])
            if parents:
                for p in parents:
                    if not g_copy.has_edge(p, n):
                        g_copy.add_edge(p, n)
            n_data[PARENTS] = ""

        # Atomic write: .tmp then os.replace (atomic on POSIX rename syscall)
        tmp_path = path + ".tmp"
        nx.write_graphml(g_copy, tmp_path)
        os.replace(tmp_path, path)
        return True

    def contains(self, node: str) -> bool:
        return self._graph.has_node(node)

    def get_graph(self) -> nx.DiGraph:
        return self._graph

    def get(self, node: str):
        return self._graph.nodes[node]

    def get_by_property(self, property_name: str, property_value: typing.Any):
        for node, n_data in self._graph.nodes(data=True):
            if property_name in n_data and n_data[property_name] == property_value:
                return n_data

    def get_nodes(self) -> NodeView:
        return self._graph.nodes

    def get_number_of_nodes(self) -> int:
        return self._graph.number_of_nodes()

    def get_number_of_edges(self) -> int:
        return self._graph.number_of_edges()

    def get_root(self) -> typing.Optional[str]:
        if self.contains(ROOT_KEY):
            return ROOT_KEY
        print(f"Action space Process {os.getpid()} - Could not find root node")
        return None

    def gen_active_target_nodes_and_data(self) -> typing.Tuple[str, dict]:
        for node in nx.descendants(self._graph, self.get_root()):
            n_data = self.get(node)
            if n_data[IS_TARGET_NODE] and not n_data[SLEEPING]:
                yield node, n_data

    def has_target_successors(self, source_node: str) -> bool:
        for node in self._graph.successors(source_node):
            n_data = self._graph.nodes[node]
            if n_data[IS_TARGET_NODE]:
                return True

        return False

    def has_active_nontarget_node(self, source_node: str = None) -> bool:
        for node in nx.descendants(self._graph, source_node if source_node else self.get_root()):
            n_data = self._graph.nodes[node]
            if not n_data[IS_TARGET_NODE]:
                return True
        return False

    def get_number_of_nontarget_nodes(self) -> int:
        results = 0
        root = self.get_root()
        for node, n_data in self._graph.nodes(data=True):
            if node != root:
                if not n_data[IS_TARGET_NODE]:
                    results += 1
        return results

    def get_number_of_target_nodes(self) -> int:
        results = 0
        root = self.get_root()
        for node, n_data in self._graph.nodes(data=True):
            if node != root:
                if n_data[IS_TARGET_NODE]:
                    results += 1
        return results

    def has_active_successors(self, source_node: str = None) -> bool:
        s_node = source_node if source_node else self.get_root()
        return self._graph.out_degree(s_node) > 0

    def get_active_nontarget_successors(self, source_node: str) -> list:
        successors = []
        for node in self._graph.successors(source_node):
            n_data = self._graph.nodes[node]
            if not n_data[IS_TARGET_NODE]:
                successors.append(node)
        return successors

    def put_to_sleep(self, node: str) -> typing.List[str]:

        # print(f"put selected node to sleep {node}")
        newly_sleeping_nodes = []
        d = collections.deque()
        d.append(node)

        root = self.get_root()
        # if parent has all sleeping nodes, put it to sleep
        while d:
            n = d.popleft()
            if n == root:
                break
            if not self.has_active_successors(n):
                self.get(n)[SLEEPING] = True
                # print(f"put parent to sleep {n}")
                newly_sleeping_nodes.append(n)
                # add parents to d first before disconnecting
                for p in self._graph.predecessors(n):
                    d.append(p)
                self.disconnect_from_parents(n)

        return newly_sleeping_nodes

    def wake_up_all_nodes(self):
        for n, n_data in self._graph.nodes(data=True):
            n_data[SLEEPING] = False
            self.reconnect_to_parents(n)

    def reset_action_attempts(self):
        for n, n_data in self._graph.nodes(data=True):
            if ACTION_ATTEMPTS in n_data:
                n_data[ACTION_ATTEMPTS] = 0
            if EXPLORED in n_data:
                n_data[EXPLORED] = False

    def disconnect_from_parents(self, node: str):
        parents = list(self._graph.predecessors(node))
        for p in parents:
            self._graph.remove_edge(p, node)

        self.get(node)[PARENTS] += parents

    def reconnect_to_parents(self, node: str):
        n_data = self.get(node)
        for p in n_data[PARENTS]:
            if not self._graph.has_edge(p, node):
                self._graph.add_edge(p, node)
        n_data[PARENTS].clear()

    def sample_successors(self, source_node: str, n_samples: int = 1,
                          use_rank_weights: bool = False) -> list:
        """
        Randomly samples a leaf node that are descendents source_node and has type node_type, active (not sleeping)
        Args:
            source_node: node of starting point
            n_samples: number of samples
            use_rank_weights: whether the sampling should be biased towards the ranking of the targets

        Returns: list of sampled leaf nodes

        """
        # if too many n_samples, return everything that meets our condition
        num_of_successors = self._graph.out_degree(source_node)
        if num_of_successors <= n_samples:
            sampled_nodes = list(self._graph.successors(source_node))
        else:
            if not use_rank_weights:
                sampled_nodes = set()
                while len(sampled_nodes) < n_samples:
                    n = next(
                        islice(self._graph.successors(source_node), random.randint(0, num_of_successors - 1), None))
                    sampled_nodes.add(n)
            else:
                successors = list(self._graph.successors(source_node))
                weights = 1 / np.array([self.get(x)[RANK] for x in successors])
                weights /= weights.sum()
                sampled_nodes = np.random.choice(successors, size=n_samples, replace=False, p=weights)

        return list(sampled_nodes)
