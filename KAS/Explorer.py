import os
import re
from typing import List, Optional

from .Bindings import Next
from .Node import Path, Node
from .Sampler import Sampler
from .Statistics import Statistics
from .Utils import NextSerializer


class Explorer:
    def __init__(self, sampler: Sampler):
        self._sampler = sampler
        self._serializer = NextSerializer()
        self._collected = set()

    def _add_node(self, node):
        self._collected.add(node.to_node())

    def _info(self, node: Node) -> None:
        print(f"Node: {node.to_node()}")
        print(f"\tis_terminal: {node.is_terminal()}")
        print(f"\tis_final: {node.is_final()}")
        print(f"\tis_dead_end: {node.is_dead_end()}")
        print(f"\tdiscovered_final_descendant: {node.discovered_final_descendant()}")
        print(f"Children summary: {node.get_children_types()}")

    def _children(self, node: Node, ty: Optional[Next.Type]) -> None:
        print("Children:")
        to_print = filter(lambda x: ty is None or x.type == ty, node.get_children_handles())
        for child in to_print:
            child_node = node.get_child(child)
            if child_node is None:
                continue
            self._add_node(child_node)
            print(f"\t{child}:\t{node.get_child_description(child)}")

    def _collected(self) -> None:
        print(f"Collected in total {len(self._collected)} nodes.")
        print(f"Among which,")
        final = list(filter(lambda x: x.is_final(), self._collected))
        print(f"\t{len(final)} are final.")

    def _deadend(self) -> None:
        dead_ends = list(filter(lambda x: x.is_dead_end(), self._collected))
        print(f"\t{len(dead_ends)} are dead ends.")

    def _expand(self, node: Node, depth: int) -> None:
        node.expand(depth)
        print(f"Expanded {depth} layers from current node.")

    def _graphviz(self, node: Node, working_dir: os.PathLike) -> None:
        # generate graphviz file and print it
        file_name = os.path.join(working_dir, 'preview.dot')
        node.generate_graphviz(file_name, "preview")
        print(f"Generated as {file_name}.")

    def _visit(self, path: Path, suffix: List[Next]) -> None:
        for next in suffix:
            path.append(next)

    def _composing(self, node: Node) -> None:
        print("Composing arcs:")
        for arc in node.get_composing_arcs():
            print(f"\t{arc}")

    def interactive(self, working_dir: os.PathLike = '.') -> Optional[Node]:
        """
        Interactive exploration.
        The current path is displayed.
        Several commands:
        - `info`: print the description of the node.
        - `children [<ty>]`: list all children of the current node. You can also specify a type filter. Example: `children Share`
        - `collected`: print statistics of collected nodes.
        - `deadend`: print the number of dead ends.
        - `expand <depth>`: expand the current node for given layers.
        - `graphviz`: generate a graphviz file and print it for the current node.
        - `visit <ty>(<key>)`: go to the child `Next(ty, key)` with the given type and key. Example: `visit Share(0)`
        - `back`: go back to the parent.
        - `statistics`: print statistics of the sampler.
        - `exit`: exit the interactive mode.
        """
        print("Type `help` for help.")
        os.makedirs(working_dir, exist_ok=True)
        path = Path([])
        while True:
            print()
            current_node = self._sampler.visit(path)
            print(f"{path}")
            if current_node is not None:
                self._add_node(current_node)
            else:
                print("Warning: this node does not exist.")
            command = input(">>> ")
            if command == "info":
                if current_node is None:
                    print("Error: this node does not exist.")
                    continue
                self._info(current_node)
            elif command.startswith("children"):
                if current_node is None:
                    print("Error: this node does not exist.")
                    continue
                ty = None
                segments = command.split()
                if len(segments) > 1:
                    try:
                        ty = self._serializer.deserialize_type(segments[1])
                    except:
                        print("Invalid type.")
                        continue
                self._children(current_node, ty)
            elif command == "collected":
                self._collected()
            elif command == "deadend":
                self._deadend()
            elif command.startswith("expand"):
                # expand the current node
                if current_node is None:
                    print("Error: this node does not exist.")
                    continue
                depth = 1
                segments = command.split()
                if len(segments) > 1:
                    try:
                        depth = int(segments[1])
                    except:
                        print("Invalid depth.")
                        continue
                self._expand(current_node, depth)
            elif command == "graphviz":
                if current_node is None:
                    print("Error: this node does not exist.")
                    continue
                self._graphviz(current_node, working_dir)
            elif command.startswith("visit"):
                if current_node is None:
                    print("Error: this node does not exist.")
                    continue
                suffix = []
                try:
                    # support multiple comma separated
                    sequence = list(filter(None, command.split(' ', 1)))[1]
                    nexts_strings = list(filter(None, re.split(r'[, ]', sequence)))
                    for next_string in nexts_strings:
                        next_type, key = next_string.split('(')
                        key = int(key.split(')')[0])
                        next = Next(self._serializer.deserialize_type(next_type), key)
                        suffix.append(next)
                except:
                    print("Invalid command.")
                self._visit(path, suffix)
            elif command.startswith("goto"):
                if current_node is None:
                    print("Error: this node does not exist.")
                    continue
                serialized = command.split(' ', 1)[1]
                path = Path.deserialize(serialized)
            elif command == "composing":
                if current_node is None:
                    print("Error: this node does not exist.")
                    continue
                self._composing(current_node)
            elif command == "back":
                # go back to parent node
                path.pop()
            elif command == "statistics":
                # print statistics
                Statistics.Print()
            elif command == "exit":
                # exit interactive mode
                return current_node
            elif command == "help":
                # print help
                print(self.interactive.__doc__)
            else:
                print("Invalid command. Enter `help` for help.")
