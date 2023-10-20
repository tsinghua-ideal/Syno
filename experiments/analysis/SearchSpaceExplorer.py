import os
import tempfile
from typing import List, Optional, Tuple, Union

from KAS import NextSerializer, Node, Path, Sampler, Statistics, VisitedNode

from AbstractExplorer import AbstractExplorer, Child as AbstractChild, Predicate as AbstractPredicate


EXPLORER_TEMP_ID = 0

class SearchSpaceExplorer(AbstractExplorer[VisitedNode]):
    def __init__(self, sampler: Sampler):
        super().__init__()
        self._sampler = sampler
        self._serializer = NextSerializer()

    def state_of(self, current_path: List[str]) -> Optional[VisitedNode]:
        abstract_path = [self._serializer.deserialize_next(handle) for handle in current_path]
        path = Path(abstract_path)
        return self._sampler.visit(path)

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
            f"Children summary: {state.get_children_types()}\n"
        )

    custom_predicates = (
        AbstractPredicate("expand", "Expand the current node for given layers.", ("3")),
        AbstractPredicate("graphviz", "Generate a graphviz file and print it for the current node."),
        AbstractPredicate("goto", "Go to the given serialized path. Example: `goto 1234_5678`.", (""), can_work_if_state_invalid=True),
        AbstractPredicate("composing", "Print the composing arcs of the current node."),
        AbstractPredicate("statistics", "Print statistics of the sampler.", can_work_if_state_invalid=True),
        # TODO: Add realize.
    )

    def available_custom_predicates(self) -> Tuple[AbstractPredicate, ...]:
        return SearchSpaceExplorer.custom_predicates

    def expand(self, state: VisitedNode, depth: str) -> Union[str, Tuple[str, List[str]]]:
        depth = int(depth)
        assert depth > 0, f"Depth must be positive, but got {depth}."
        state.expand(depth)
        return f"Expanded {depth} layers from current node."

    def graphviz(self, state: VisitedNode) -> Union[str, Tuple[str, List[str]]]:
        # Create a temporary file.
        global EXPLORER_TEMP_ID
        file_name = os.path.join(tempfile.gettempdir(), f"kas_preview_{EXPLORER_TEMP_ID}.dot")
        EXPLORER_TEMP_ID += 1
        state.generate_graphviz(file_name, "preview")
        with open(file_name, "r") as f:
            result = f.read()
        # Delete the temporary file.
        os.remove(file_name)
        return result

    def goto(self, state: Optional[VisitedNode], serialized_path: str) -> Union[str, Tuple[str, List[str]]]:
        path = Path.deserialize(serialized_path)
        return f"Going to {path}.", [self._serializer.serialize_next(next) for next in path]

    def composing(self, state: VisitedNode) -> Union[str, Tuple[str, List[str]]]:
        return (
            "Composing arcs:\n" +
            "".join(
                f"\t{arc}\n"
                for arc in state.get_composing_arcs()
            )
        )

    def statistics(self, state: Optional[VisitedNode]) -> Union[str, Tuple[str, List[str]]]:
        return Statistics.Summary()
