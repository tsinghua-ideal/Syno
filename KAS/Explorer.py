import os

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

    def _expand(self, fro: Node, depth: int) -> None:
        if depth <= 0:
            return
        for next in fro.get_children_handles():
            to = fro.get_child(next)
            if to is None:
                continue
            self._add_node(to)
            self._expand(to, depth - 1)

    def interactive(self, working_dir: str = '.') -> None:
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
                print(f"Node: {current_node.to_node()}")
                print(f"\tis_terminal: {current_node.is_terminal()}")
                print(f"\tis_final: {current_node.is_final()}")
                print(f"\tis_dead_end: {current_node.is_dead_end()}")
                print(f"\tdiscovered_final_descendant: {current_node.discovered_final_descendant()}")
                print(f"Children summary: {current_node.get_children_types()}")
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
                # list all children
                print("Children:")
                to_print = filter(lambda x: ty is None or x.type == ty, current_node.get_children_handles())
                for child in to_print:
                    child_node = current_node.get_child(child)
                    if child_node is None:
                        continue
                    self._add_node(child_node)
                    print(f"\t{child}:\t{current_node.get_child_description(child)}")
            elif command == "collected":
                print(f"Collected in total {len(self._collected)} nodes.")
                print(f"Among which,")
                final = list(filter(lambda x: x.is_final(), self._collected))
                print(f"\t{len(final)} are final.")
            elif command == "deadend":
                dead_ends = list(filter(lambda x: x.is_dead_end(), self._collected))
                print(f"\t{len(dead_ends)} are dead ends.")
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
                print(f"Expanded {depth} layers from current node.")
            elif command == "graphviz":
                if current_node is None:
                    print("Error: this node does not exist.")
                    continue
                # generate graphviz file and print it
                current_node.generate_graphviz(working_dir, "preview")
                print(f"Generated as {os.path.join(working_dir, 'preview.dot')}.")
            elif command.startswith("visit"):
                if current_node is None:
                    print("Error: this node does not exist.")
                    continue
                # go to child node
                try:
                    next_type, key = command.split()[1].split('(')
                    key = int(key.split(')')[0])
                    next = Next(self._serializer.deserialize_type(next_type), key)
                    path.append(next)
                except:
                    print("Invalid command.")
                if next not in current_node.get_children_handles():
                    print("This child does not exist.")
            elif command == "back":
                # go back to parent node
                path.pop()
            elif command == "statistics":
                # print statistics
                Statistics.Print()
            elif command == "exit":
                # exit interactive mode
                break
            elif command == "help":
                # print help
                print(self.interactive.__doc__)
            else:
                print("Invalid command. Enter `help` for help.")
