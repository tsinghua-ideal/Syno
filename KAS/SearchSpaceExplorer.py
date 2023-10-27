import os
import tempfile
from torch import nn
from typing import List, Optional, Tuple, Union

from KAS import NextSerializer, Node, Path, Sampler, Statistics, VisitedNode

from KAS.AbstractExplorer import AbstractChild, AbstractExplorer, AbstractPredicate, AbstractResponse


EXPLORER_TEMP_ID = 0

class SearchSpaceExplorer(AbstractExplorer[VisitedNode]):
    def __init__(self, model: nn.Module, sampler: Sampler):
        super().__init__(model, sampler)
        self._serializer = NextSerializer()

    def state_of(self, current_path: List[str]) -> Optional[VisitedNode]:
        abstract_path = [self._serializer.deserialize_next(handle) for handle in current_path]
        path = Path(abstract_path)
        node = self.sampler.visit(path)
        if node is None:
            return None
        node.expand(1)
        if node.is_dead_end():
            return None
        return node

    def children(self, state: VisitedNode) -> List[AbstractChild]:
        handles = state.get_children_handles()
        return [
            AbstractChild(
                self._serializer.serialize_next(handle),
                state.get_child_description(handle) or "Failed to get description.",
                self._serializer.serialize_type(handle.type),
            )
            for handle in handles
        ]

    def info(self, state: VisitedNode) -> str:
        return (
            f"Node: {state.to_node()}\n"
            f"\tis_terminal: {state.is_terminal()}\n"
            f"\tis_final: {state.is_final()}\n"
            f"\tis_dead_end: {state.is_dead_end()}\n"
            f"\tdiscovered_final_descendant: {state.discovered_final_descendant()}\n"
            f"\tshape_distance: {state.get_shape_distance()}\n"
            f"Children summary: {state.get_children_types()}\n"
        )

    custom_predicates = (
        AbstractPredicate("expand", "Expand the current node for given layers.", ("3",)),
        AbstractPredicate("fill_lattice", "Expand the lattice of the current node."),
        AbstractPredicate("graphviz", "Generate a graphviz file and print it for the current node."),
        AbstractPredicate("goto", "Go to the given serialized path. Example: `goto 1234_5678`.", ("",), can_work_if_state_invalid=True),
        AbstractPredicate("composing", "Print the composing arcs of the current node."),
        AbstractPredicate("statistics", "Print statistics of the sampler.", can_work_if_state_invalid=True),
        AbstractPredicate("realize", "Realize the current node."),
    )

    def available_custom_predicates(self) -> Tuple[AbstractPredicate, ...]:
        return SearchSpaceExplorer.custom_predicates

    def expand(self, state: VisitedNode, depth: str) -> AbstractResponse:
        depth = int(depth)
        assert depth > 0, f"Depth must be positive, but got {depth}."
        state.expand(depth)
        return AbstractResponse(f"Expanded {depth} layers from current node.")

    def fill_lattice(self, state: VisitedNode) -> AbstractResponse:
        self._sampler.root().expand_to(state)
        return AbstractResponse(f"Expanded lattice.")

    def graphviz(self, state: VisitedNode) -> Union[str, Tuple[str, List[str]]]:
        # Create a temporary file.
        global EXPLORER_TEMP_ID
        filename = f"search_space_explorer_{EXPLORER_TEMP_ID}.dot"
        path = os.path.join(self.working_dir, filename)
        EXPLORER_TEMP_ID += 1
        state.generate_graphviz(path, "preview")
        with open(path, "r") as f:
            result = f.read()
        return AbstractResponse(f"Generated graphviz file {filename}.", returned_file=filename)

    def goto(self, state: Optional[VisitedNode], serialized_path: str) -> Union[str, Tuple[str, List[str]]]:
        path = Path.deserialize(serialized_path)
        return AbstractResponse(f"Going to {path}.", next_state=[self._serializer.serialize_next(next) for next in path])

    def composing(self, state: VisitedNode) -> Union[str, Tuple[str, List[str]]]:
        return AbstractResponse(
            "Composing arcs:\n" +
            "".join(
                f"\t{arc}\n"
                for arc in state.get_composing_arcs()
            )
        )

    def statistics(self, state: Optional[VisitedNode]) -> Union[str, Tuple[str, List[str]]]:
        return AbstractResponse(Statistics.Summary())

    def realize(self, state: VisitedNode) -> Union[str, Tuple[str, List[str]]]:
        kernel_loader = self.sampler.realize(self.model, state)
        global EXPLORER_TEMP_ID
        filename = f"search_space_explorer_{EXPLORER_TEMP_ID}.tar.gz"
        path = os.path.join(self.working_dir, filename)
        EXPLORER_TEMP_ID += 1
        kernel_loader.archive_to(path)
        return AbstractResponse(f"Realized to {filename}.", returned_file=filename)
