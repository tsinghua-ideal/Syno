import os

from .Bindings import Next
from .Node import Node
from .Utils import NextSerializer
from .Tree import MCTS
from .TreeNode import TreeNode, TreePath

class TreeExplorer:
    """
    An explorer to explore a mcts. 
    """
    def __init__(self, mcts: MCTS):
        self._mcts = mcts
        self._serializer = NextSerializer()
        self.on_tree = False

    def _expand(self, fro: Node, depth: int) -> None:
        fro.expand(depth)

    def interactive(self, working_dir: str = '.') -> None:
        """
        Interactive exploration.
        The current path is displayed.
        Several commands:
        - `(i)nfo`: print the description of the node.
        - `(c)hildren`: list all children of the current node. 
        - `(g)rave [<ty>]`: print the grave dictionary. Example: `grave Share`
        - `(l)rave [<ty>]`: print the lrave dictionary. Example: `lrave Share`
        - `graphviz`: generate a graphviz file and print it for the current node.
        - `(v)isit <ty>(<key>)`: go to the child `Next(ty, key)` with the given type and key. Example: `visit Share(0)`
        - `(b)ack`: go back to the parent.
        - `explore_tree`: see only visible nodes.
        - `explore_all`: see all nodes.
        - `exit`: exit the interactive mode.
        """
        print("Type `help` for help.")
        path = TreePath([])
        node_hierarchy = [self._mcts.tree_root]
        while True:
            current_node = node_hierarchy[-1]
            command = input(f"({path}) >>> ")
            if command in ["i", "info"]:
                print(f"Node: {current_node.to_node()}")
                print(f"\t is_terminal: {current_node.is_terminal(self._mcts._treenode_store)}")
                print(f"\t is_final: {current_node.is_final()}")
                print(f"\t is_dead_end: {current_node._is_dead}")
                print(f"\t states:")
                print(f"\t\t N={current_node.state.N}")
                print(f"\t\t mean={current_node.state.mean}")
                print(f"\t\t std={current_node.state.std}")
            elif command.startswith("c"):
                # list all children
                print("Children:")
                to_print = current_node.get_children(
                        self._mcts._treenode_store, 
                        auto_initialize=False, 
                        on_tree=self.on_tree
                    )
                for nxt, child_node, edge_state in to_print:
                    if current_node._is_mid:
                        child = Next(current_node._type, nxt)
                        print(f"\t{child}:\t{current_node._node.get_child_description(child)}, edge(N={edge_state.N}, mean={edge_state.mean}, std={edge_state.std})")
                    else:
                        assert isinstance(nxt, Next.Type)
                        print(f"\t{nxt}:\t{child_node.children_count(self._mcts._treenode_store, on_tree=self.on_tree)} children, edge(N={edge_state.N}, mean={edge_state.mean}, std={edge_state.std})")
            elif command in ["g", "grave"]:
                # TODO: type filtering
                ty = None
                segments = command.split()
                if len(segments) > 1:
                    try:
                        ty = self._serializer.deserialize_type(segments[1])
                    except:
                        print(f"Invalid type {ty}, please try again. ")
                        continue
                print("Grave:")
                for key, value in self._mcts.g_rave.items():
                    if ty is None or ty == key.ty:
                        print(f"\t{key}: (N={value.N}, mean={value.mean})")
            elif command in ["l", "lrave"]:
                ty = None
                segments = command.split()
                if len(segments) > 1:
                    try:
                        ty = self._serializer.deserialize_type(segments[1])
                    except:
                        print(f"Invalid type {ty}, please try again. ")
                        continue
                print("Lrave:")
                for key, value in current_node.l_rave.items():
                    if ty is None or ty == key.ty:
                        print(f"\t{key}: (N={value.N}, mean={value.mean})")
            elif command == "graphviz":
                if current_node._is_mid:
                    print("Warning: graphviz is intended to run on non-mid nodes.")
                    continue
                # generate graphviz file and print it
                current_node._node.generate_graphviz(working_dir, "preview")
                print(f"Generated as {os.path.join(working_dir, 'preview.dot')}.")
            elif command.startswith("v"):
                # go to child node
                try:
                    tree_next = command.split()[1]
                    if current_node._is_mid:
                        tree_next = int(tree_next)
                    else:
                        tree_next = self._serializer.deserialize_type(tree_next)
                    child_node, _ = current_node.get_child(tree_next, self._mcts._treenode_store, auto_initialize=False, on_tree=self.on_tree)
                    if child_node is None:
                        print(f"Child {tree_next} does not exist.")
                    else:
                        node_hierarchy.append(child_node)
                        path = path.concat(tree_next)
                except:
                    print("Invalid command.")
            elif command in ["b", "back"]:
                # go back to parent node
                node_hierarchy.pop()
            elif command == 'explore_all':
                self.on_tree = False
            elif command == 'explore_tree':
                self.on_tree = True
            elif command == "exit":
                # exit interactive mode
                break
            elif command in ["h", "help"]:
                # print help
                print(self.interactive.__doc__)
            else:
                print(f"Invalid command {command}. Enter `help` for help.")
