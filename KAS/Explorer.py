import os

from .Bindings import Next
from .Node import Path
from .Sampler import Sampler
from .Utils import NextSerializer


class Explorer:
    def __init__(self, sampler: Sampler):
        self._sampler = sampler
        self._serializer = NextSerializer()

    def interactive(self, working_dir: str = '.') -> None:
        """
        Interactive exploration.
        The current path is displayed.
        Several commands:
        - `info`: print the description of the node.
        - `children [<ty>]`: list all children of the current node. You can also specify a type filter. Example: `children Share`
        - `graphviz`: generate a graphviz file and print it for the current node.
        - `visit <ty>(<key>)`: go to the child `Next(ty, key)` with the given type and key. Example: `visit Share(0)`
        - `back`: go back to the parent.
        - `exit`: exit the interactive mode.
        """
        print("Type `help` for help.")
        path = Path([])
        while True:
            print()
            current_node = self._sampler.visit(path)
            print(f"{path}")
            children_handles = current_node.get_children_handles()
            command = input(">>> ")
            if command == "info":
                print(f"Node: {current_node.to_node()}")
                print(f"is_terminal: {current_node.is_terminal()}, is_final: {current_node.is_final()}, is_dead_end: {current_node.is_dead_end()}")
                print(f"Children summary: {current_node.get_children_types()}")
            elif command.startswith("children"):
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
                to_print = filter(lambda x: ty is None or x.type == ty, children_handles)
                for child in to_print:
                    print(f"\t{child}:\t{current_node.get_child_description(child)}")
            elif command == "graphviz":
                # generate graphviz file and print it
                current_node.generate_graphviz(working_dir, "preview")
                print(f"Generated as {os.path.join(working_dir, 'preview.dot')}.")
            elif command.startswith("visit"):
                # go to child node
                try:
                    next_type, key = command.split()[1].split('(')
                    key = int(key.split(')')[0])
                    next = Next(self._serializer.deserialize_type(next_type), key)
                    path.append(next)
                except:
                    print("Invalid command.")
                if next not in children_handles:
                    print("This child does not exist.")
            elif command == "back":
                # go back to parent node
                path.pop()
            elif command == "exit":
                # exit interactive mode
                break
            elif command == "help":
                # print help
                print(self.interactive.__doc__)
            else:
                print("Invalid command. Enter `help` for help.")
