import os
from torch import nn
from typing import List, Optional, Tuple
from copy import deepcopy

from KAS import NextSerializer, Sampler, Statistics, Next

from KAS.AbstractExplorer import (
    AbstractChild,
    AbstractExplorer,
    AbstractPredicate,
    AbstractResponse,
)

from .node import TreePath, TreeNode
from .tree import MCTSTree

EXPLORER_TEMP_ID = 0


class MCTSExplorer(AbstractExplorer[TreeNode]):
    def __init__(self, model: nn.Module, sampler: Sampler, mcts: MCTSTree):
        super().__init__(model, sampler)
        self._serializer = NextSerializer()
        self._mcts = mcts

    def _tree_path_to_explorer_path(self, path: TreePath) -> List[str]:
        explorer_path = []
        for next in path:
            explorer_path.append(self._serializer.serialize_type(next.type))
            explorer_path.append(str(next.key))
        return explorer_path

    def state_of(self, current_path: List[str]) -> Optional[TreeNode]:
        # Path: alternating type and key
        path = deepcopy(current_path)
        if len(current_path) % 2 == 1:
            path += ["0"]
        abstract_path = [
            Next(self._serializer.deserialize_type(next_type), int(next_key))
            for next_type, next_key in zip(path[::2], path[1::2])
        ]
        path = TreePath(abstract_path)
        node = self._mcts.visit(path, on_tree=False)
        if node is None or node.is_dead_end():
            return None
        return node

    def children(self, state: TreeNode) -> List[AbstractChild]:
        handles = state.get_children(filter_simulate_failure=False)
        if state._is_mid:
            return [
                AbstractChild(
                    str(nxt),
                    f"{state.to_node().get_arc_from_handle(Next(state._type, nxt))} {edge_state} with l-rave {state.l_rave[state.to_node().get_arc_from_handle(Next(state._type, nxt))]} {'(simulation failed)' if child_node._simulate_fail else ''}",
                )
                for nxt, child_node, edge_state in handles
            ]
        else:
            return [
                AbstractChild(
                    self._serializer.serialize_type(nxt),
                    f"{str(nxt).split('.')[-1]} {edge_state} {'(simulation failed)' if child_node._simulate_fail else ''}",
                )
                for nxt, child_node, edge_state in handles
            ]

    def info(self, state: TreeNode) -> str:
        return (
            f"Node: {state.to_node()}\n"
            f"\tis_mid: {state._is_mid}\n"
            f"\tis_terminal: {state.is_terminal()}\n"
            f"\tis_final: {state.is_final()}, reward={state.reward}, filtered={state.filtered}\n"
            if state.is_final()
            else f"\tis_final: {state.is_final()}\n"
            f"\tis_dead_end: {state.is_dead_end()}\n"
            f"\tis_dead: {state._is_dead}\n"
            f"\tsimulate_fail: {state._simulate_fail}\n"
            f"\tis_alive: {state._not_dead}\n"
            f"\tis_exhausted: {state._exhausted}\n"
            f"\tis_in_tree: {state._isin_tree}\n"
            f"\tstates: {state.state}\n"
            f"\tvirtual_loss: {state._virtual_loss}\n"
        )

    custom_predicates = (
        AbstractPredicate("expand", "Expand the current node for given layers.", ("3")),
        AbstractPredicate(
            "sample",
            "Sample final nodes. Randomly jump to any of the results. The argument is the number of trials.",
            ("1024",),
        ),
        AbstractPredicate(
            "graphviz", "Generate a graphviz file and print it for the current node."
        ),
        AbstractPredicate(
            "goto",
            "Go to the given serialized path. Example: `goto 1234_5678`.",
            ("",),
            can_work_if_state_invalid=True,
        ),
        AbstractPredicate("composing", "Print the composing arcs of the current node."),
        AbstractPredicate(
            "statistics",
            "Print statistics of the sampler.",
            can_work_if_state_invalid=True,
        ),
        AbstractPredicate("realize", "Realize the current node."),
    )

    def available_custom_predicates(self) -> Tuple[AbstractPredicate, ...]:
        return self.custom_predicates

    def expand(self, state: TreeNode, depth: str) -> AbstractResponse:
        depth = int(depth)
        assert depth > 0, f"Depth must be positive, but got {depth}."
        state.to_node().expand(depth)
        return AbstractResponse(f"Expanded {depth} layers from current node.")

    def sample(self, state: TreeNode, trials: str) -> AbstractResponse:
        trials = int(trials)
        assert trials > 0, f"Trials must be positive, but got {trials}."
        # TODO: add TreePath to state, and avoid this conversion
        results = self.sampler.random_final_nodes_with_prefix(
            state.to_node().get_possible_path(), trials, None, 2
        )
        if len(results) == 0:
            return AbstractResponse("All trials failed.")
        result = max(results, key=lambda result: len(result.path))
        return AbstractResponse(
            f"Sampled {len(results)} final nodes. Jumping to deepest {result.path}.",
            next_state=self._tree_path_to_explorer_path(TreePath(result.path)),
        )

    def graphviz(self, state: TreeNode) -> AbstractResponse:
        # Create a temporary file.
        global EXPLORER_TEMP_ID
        filename = f"mcts_explorer_{EXPLORER_TEMP_ID}.dot"
        path = os.path.join(self.working_dir, filename)
        EXPLORER_TEMP_ID += 1
        state.to_node().generate_graphviz(path, "preview")
        with open(path, "r") as f:
            result = f.read()
        return AbstractResponse(
            f"Generated graphviz file {filename}.", returned_file=filename
        )

    def goto(self, state: Optional[TreeNode], serialized_path: str) -> AbstractResponse:
        path = TreePath.deserialize(serialized_path)
        explorer_path = self._tree_path_to_explorer_path(path)
        return AbstractResponse(
            f"Going to {path}.",
            next_state=explorer_path,
        )

    def composing(self, state: TreeNode) -> AbstractResponse:
        return AbstractResponse(
            "Composing arcs:\n"
            + "".join(f"\t{arc}\n" for arc in state.to_node().get_composing_arcs())
        )

    def statistics(self, state: Optional[TreeNode]) -> AbstractResponse:
        return AbstractResponse(Statistics.Summary(self.sampler))

    def realize(self, state: TreeNode) -> AbstractResponse:
        kernel_loader = self.sampler.realize(self.model, state.to_node())
        global EXPLORER_TEMP_ID
        filename = f"mcts_explorer_{EXPLORER_TEMP_ID}.tar.gz"
        path = os.path.join(self.working_dir, filename)
        EXPLORER_TEMP_ID += 1
        kernel_loader.archive_to(path)
        return AbstractResponse(f"Realized to {filename}.", returned_file=filename)
