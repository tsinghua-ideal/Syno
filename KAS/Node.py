from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

from . import Bindings
from .Bindings import Next, Arc


AbsolutePath = List[Next]
'''
AbsolutePath is a list of Next(type=Next.Type, key=int).
'''

PseudoNext = Union[Next, Tuple[str, int], str]
PseudoPath = Union['Path', List[PseudoNext]]

class Path:
    """A path in Python, not necessarily corresponding to a C++ path."""

    @staticmethod
    def to_next(tup: PseudoNext) -> Next:
        if isinstance(tup, Next):
            return tup
        if isinstance(tup, str):
            # is of form 'type(key)'
            t, k = tup.split('(')
            k = int(k.split(')')[0])
        else:
            t, k = tup
        if isinstance(t, str):
            t = getattr(Next, t)
        return Next(t, k)

    def serialize(self) -> str:
        serialized = [str(int(n.type)) + str(n.key) for n in self.abs_path]
        return '_'.join(serialized)

    @ staticmethod
    def deserialize(serialized: str) -> 'Path':
        deserialized_list = serialized.split('_')
        return Path([Next(Next.Type(n[0]), int(n[1:])) for n in deserialized_list])

    def __init__(self, path: PseudoPath) -> None:
        self.abs_path: AbsolutePath = [Path.to_next(n) for n in path]

    def __len__(self) -> int:
        return len(self.abs_path)

    def __iter__(self):
        return iter(self.abs_path)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Path):
            return False
        return self.abs_path == __value.abs_path

    def __hash__(self) -> int:
        return hash(tuple(self.abs_path))

    def append(self, next: PseudoNext):
        self.abs_path.append(Path.to_next(next))

    def pop(self):
        self.abs_path.pop()

    def concat(self, next: PseudoNext) -> 'Path':
        return Path(self.abs_path + [Path.to_next(next)])

    def to_identifier(self) -> str:
        return '_'.join(str(next) for next in self.abs_path).replace('(', '').replace(')', '')

    def __repr__(self) -> str:
        return f'[{", ".join(str(next) for next in self.abs_path)}]'


class Node:
    """A node in Python, corresponding to a C++ Node."""

    def __init__(self, node: Bindings.Node) -> None:
        self._node = node

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Node):
            return False
        return self._node == __value._node

    def __hash__(self) -> int:
        return hash(self._node)

    def children_count(self) -> int:
        """Get the number of all children of a node."""
        return self._node.children_count()

    def get_children_handles(self) -> List[Next]:
        """Get all children of a node."""
        return self._node.get_children_handles()

    def collect_operations(self) -> Dict[Next.Type, List[int]]:
        """Group the children of a node by their type."""
        handles = self.get_children_handles()
        result = defaultdict(list)
        for handle in handles:
            result[handle.type].append(handle.key)
        return result

    def get_children_types(self) -> Dict[str, int]:
        """Count the number of children of each type."""
        handles = self.get_children_handles()
        result = defaultdict(int)
        for handle in handles:
            result[str(handle.type)] += 1
        return result

    def get_children_arcs(self) -> List[Arc]:
        """Get all arcs of a node."""
        return self._node.get_children_arcs()

    def get_arc_from_handle(self, handle: Next) -> Optional[Arc]:
        """Get the arc from a Next."""
        return self._node.get_arc_from_handle(handle)

    def get_child(self, next: PseudoNext) -> Optional['Node']:
        """Get the child node of a node with a Next."""
        child_node = self._node.get_child(Path.to_next(next))
        if child_node is None:
            return None
        return Node(child_node)

    def get_child_from_arc(self, arc: Arc) -> 'Node':
        """Get the child node of a node with an Arc."""
        child_node = self._node.get_child_from_arc(arc)
        return Node(child_node)

    def get_possible_path(self) -> Path:
        """Get a possible path of a node."""
        return Path(self._node.get_possible_path())

    def get_composing_arcs(self) -> List[Arc]:
        """Get all composing arcs of a node."""
        return self._node.get_composing_arcs()

    def get_child_description(self, next: PseudoNext) -> Optional[str]:
        """Get the description of Next."""
        return self._node.get_child_description(Path.to_next(next))

    def is_final(self) -> bool:
        """Check if a node is final, which means it can be realized as a Halide kernel."""
        return self._node.is_final()

    def is_dead_end(self) -> bool:
        """Check if a node is a dead end, which means it has no children and is not final."""
        return self._node.is_dead_end()

    def discovered_final_descendant(self) -> bool:
        """Check if a node has a final descendant."""
        return self._node.discovered_final_descendant()

    def is_terminal(self) -> bool:
        """Check if a node is terminal, which means it is either final or a dead end."""
        return self.is_final() or self.is_dead_end()

    def _realize_as_final(self, all_mappings: List[Dict[str, int]], halide_options: Bindings.CodeGenOptions) -> Bindings.Kernel:
        return self._node.realize_as_final(all_mappings, halide_options)

    def estimate_total_flops_as_final(self) -> int:
        return self._node.estimate_total_flops_as_final()

    def generate_graphviz(self, dir: str, name: str) -> None:
        self._node.generate_graphviz(dir, name)

    def generate_graphviz_as_final(self, dir: str, name: str) -> None:
        self._node.generate_graphviz_as_final(dir, name)

    def get_nested_loops_as_final(self) -> str:
        return self._node.get_nested_loops_as_final()

    def to_node(self) -> 'Node':
        return Node(self._node)

    def __repr__(self) -> str:
        return str(self._node)

class VisitedNode(Node):
    """Node with Path."""

    def __init__(self, path: Path, node: Bindings.Node) -> None:
        super().__init__(node)
        self.path = path

    def __eq__(self, __value: object) -> bool:
        raise ValueError("VisitedNode should not be compared.")
    def __hash__(self) -> int:
        raise ValueError("VisitedNode should not be hashed.")

    def get_child(self, next: PseudoNext) -> Optional['VisitedNode']:
        """Get the child node of a node with a Next."""
        child_node = self._node.get_child(Path.to_next(next))
        if child_node is None:
            return None
        return VisitedNode(self.path.concat(next), child_node)

    def to_node(self) -> 'Node':
        return Node(self._node)

    def __repr__(self) -> str:
        return f"VisitedNode({self.path}, {self._node})"

class MockNodeMetadata:
    """Bindings.Node equivalent."""

    def __init__(self, mock_sampler, id: int, name: Union[str, None] = None, is_final: bool = False, accuracy: float = 0.0, total_flops: int = 114514, **additional_attributes) -> None:
        self._mock_sampler = mock_sampler
        assert id is not None
        self._id = id
        self._path = None
        self._name = name if name is not None else f"MockNode({id})"
        self._is_final = is_final
        self._accuracy = accuracy
        self._total_flops = total_flops
        self._additional_attributes = additional_attributes

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, MockNodeMetadata):
            return False
        return self._id == __value._id
    
    def __hash__(self) -> int:
        return hash(self._id)

    def mock_get_id(self) -> int:
        return self._id

    def mock_get_path(self) -> Union[None, Path]:
        return self._path

    def mock_set_path(self, path: Path) -> None:
        self._path = path

    def _mock_children(self) -> Dict[Next, 'MockNodeMetadata']:
        return self._mock_sampler.mock_get_children(self._id)

    def children_count(self) -> int:
        return len(self._mock_children())

    def get_children_handles(self) -> List[Next]:
        return list(self._mock_children().keys())

    def get_children_arcs(self) -> List[Next]:
        return self.get_children_handles()

    def get_arc_from_handle(self, handle: Next) -> Optional[Next]:
        return handle

    def get_child(self, next: Next) -> Optional['MockNodeMetadata']:
        return self._mock_children().get(next, None)

    def get_child_from_arc(self, arc: Next) -> 'MockNodeMetadata':
        return self.get_child(arc)

    def get_possible_path(self) -> Path:
        return self._path

    def get_composing_arcs(self) -> List[Next]:
        return self.get_possible_path().abs_path

    def _get_child_description_helper(self, next: Next) -> Optional[str]:
        if next in self._mock_children():
            return f"{next} of {self._name}"
        return None

    def is_final(self) -> bool:
        return self._is_final

    def is_dead_end(self) -> bool:
        return not self.is_final() and self.children_count() == 0

    def discovered_final_descendant(self) -> bool:
        return self.is_final()

    def estimate_total_flops_as_final(self) -> int:
        return self._total_flops

    def generate_graphviz(self, dir: str, name: str) -> None:
        pass

    def generate_graphviz_as_final(self, dir: str, name: str) -> None:
        assert self._is_final

    def get_nested_loops_as_final(self) -> str:
        assert self._is_final
        return f"for (int i = 0; i < 10; i++) {{\tthis is {self._name}\n}}\n"

    def mock_get_accuracy(self) -> float:
        return self._accuracy

    def mock_get(self, key: str) -> Any:
        return self._additional_attributes[key]

    def __repr__(self) -> str:
        return self._name

class MockNode(Node):
    """A mock node for testing."""

    def __init__(self, node: MockNodeMetadata) -> None:
        self._node = node

    def mock_get_id(self) -> int:
        return self._node.mock_get_id()

    def mock_get_path(self) -> Union[None, List[Next]]:
        return self._node.mock_get_path()

    def mock_set_path(self, path: List[Next]) -> None:
        self._node.mock_set_path(path)

    def get_child(self, next: PseudoNext) -> Optional['MockNode']:
        child_node = self._node.get_child(Path.to_next(next))
        if child_node is None:
            return None
        return MockNode(child_node)

    def get_child_description(self, next: PseudoNext) -> Optional[str]:
        return self._node._get_child_description_helper(Path.to_next(next))

    def _realize_as_final(self, all_mappings: List[Dict[str, int]], halide_options: Bindings.CodeGenOptions) -> Bindings.Kernel:
        raise ValueError("MockNode cannot be realized as final.")

    def to_node(self) -> 'MockNode':
        return MockNode(self._node)

    def mock_get_accuracy(self) -> float:
        return self._node.mock_get_accuracy()

    def mock_get(self, key: str) -> Any:
        return self._node.mock_get(key)

    def __repr__(self) -> str:
        return str(self._node)

class MockVisitedNode(MockNode):
    """MockNode with Path."""

    def __init__(self, path: Path, node: MockNodeMetadata) -> None:
        super().__init__(node)
        self.path = path

    def __eq__(self, __value: object) -> bool:
        raise ValueError("MockVisitedNode should not be compared.")
    def __hash__(self) -> int:
        raise ValueError("MockVisitedNode should not be hashed.")

    def get_child(self, next: PseudoNext) -> Optional['MockVisitedNode']:
        child_node = self._node.get_child(Path.to_next(next))
        if child_node is None:
            return None
        return MockVisitedNode(self.path.concat(next), child_node)

    def __repr__(self) -> str:
        return f"MockVisitedNode({self.path}, {self._node})"
