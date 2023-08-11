import itertools
import os
import math
from tqdm import tqdm
from time import time
from typing import List, Callable
from KAS.Bindings import Next
from KAS.Node import Node
from KAS.Utils import NextSerializer

from .node import TreePath
from .tree import MCTSTree


class MCTSExplorer:
    """
    An explorer to explore a mcts.
    """

    def __init__(self, mcts: MCTSTree):
        self._mcts = mcts
        self._serializer = NextSerializer()
        self.on_tree = False
        self.show_zero = False
        self.node_hierarchy = [(TreePath([]), self._mcts.tree_root)]

    @property
    def ucb_score(self) -> Callable:
        _, current_node = self.node_hierarchy[-1]
        log_N_vertex = math.log(current_node.N)

        def ucb1_tuned(key) -> float:
            _, child, edge = key
            if edge.N == 0:
                return 1e9
            return edge.mean + self._mcts._exploration_weight * math.sqrt(
                (log_N_vertex / edge.N)
                * min(0.25, edge.std * edge.std + math.sqrt(2 * log_N_vertex / edge.N))
            )

        return ucb1_tuned

    def handle_one_command(self, command: str) -> None:
        path, current_node = self.node_hierarchy[-1]
        if command in ["i", "info"]:
            print(f"Node: {current_node.to_node()}")
            print(
                f"\t is_terminal: {current_node.is_terminal(self._mcts._treenode_store)}"
            )
            print(f"\t is_final: {current_node.is_final()}")
            if current_node.is_final():
                print(f"\t\t reward: {current_node.reward}")
                print(f"\t\t filtered: {current_node.filtered}")
            print(f"\t is_dead_end: {current_node._is_dead}")
            print(f"\t is_exhausted: {current_node._exhausted}")
            print(f"\t states:")
            print(f"\t\t N={current_node.state.N}")
            print(f"\t\t mean={current_node.state.mean}")
            print(f"\t\t std={current_node.state.std}")

        elif command.startswith("c"):
            # list all children
            print("Children:")
            # assert self.on_tree == False
            children = current_node.get_children(
                self._mcts._treenode_store, auto_initialize=True, on_tree=self.on_tree
            )
            for nxt, child_node, edge_state in children:
                if self.show_zero and edge_state.N == 0:
                    continue
                if current_node._is_mid:
                    child = Next(current_node._type, nxt)
                    print(
                        f"\t{child}:\t{current_node._node.get_child_description(child)}, edge(N={edge_state.N}, mean={edge_state.mean}, std={edge_state.std})"
                    )
                else:
                    assert isinstance(nxt, Next.Type)
                    print(
                        f"\t{nxt}:\t{child_node.children_count(self._mcts._treenode_store, on_tree=self.on_tree)} children, edge(N={edge_state.N}, mean={edge_state.mean}, std={edge_state.std})"
                    )
        elif command.startswith("sel"):
            # list all children
            print("UCD:")
            children = current_node.get_children(
                self._mcts._treenode_store, auto_initialize=False, on_tree=self.on_tree
            )
            log_N_vertex = math.log(current_node.N)

            def ucb1_tuned(key) -> float:
                """
                Upper confidence bound.
                UCB1-tuned.
                """
                _, child, edge = key
                if edge.N == 0:
                    return 1e9
                return edge.mean + self._mcts._exploration_weight * math.sqrt(
                    (log_N_vertex / edge.N)
                    * min(
                        0.25, edge.std * edge.std + math.sqrt(2 * log_N_vertex / edge.N)
                    )
                )

            ucb_score = [ucb1_tuned(c) for c in children]
            for (nxt, child_node, edge_state), score in zip(children, ucb_score):
                if current_node._is_mid:
                    child = Next(current_node._type, nxt)
                    print(
                        f"\t{child}:\t{current_node._node.get_child_description(child)}, score={score}"
                    )
                else:
                    assert isinstance(nxt, Next.Type)
                    print(
                        f"\t{nxt}:\t{child_node.children_count(self._mcts._treenode_store, on_tree=self.on_tree)} children, score={score}"
                    )
        elif command in ["g", "grave"]:
            ty = None
            segments = command.split()
            if len(segments) > 1:
                try:
                    ty = self._serializer.deserialize_type(segments[1])
                except:
                    print(f"Invalid type {ty}, please try again. ")
                    return
            sorted_grave = sorted(
                filter(
                    lambda x: (ty is None or x[0]._type == ty),
                    list(self._mcts.g_rave.items()),
                ),
                key=lambda x: x[1].N,
                reverse=True,
            )
            sorted_grave_hasvalue = [grave for grave in sorted_grave if grave[1].N > 0]

            print(
                f"Grave ({len(sorted_grave)} in total, in which {len(sorted_grave_hasvalue)} has positive value):"
            )
            for i, (key, value) in enumerate(sorted_grave_hasvalue):
                print(f"\t{i+1}. {key}: (N={value.N}, mean={value.mean})")
        elif command in ["l", "lrave"]:
            ty = None
            segments = command.split()
            if len(segments) > 1:
                try:
                    ty = self._serializer.deserialize_type(segments[1])
                except:
                    print(f"Invalid type {ty}, please try again. ")
                    return
            sorted_lrave = sorted(
                filter(
                    lambda x: (ty is None or x[0]._type == ty),
                    list(current_node.l_rave.items()),
                ),
                key=lambda x: x[1].N,
                reverse=True,
            )
            sorted_lrave_hasvalue = [lrave for lrave in sorted_lrave if lrave[1].N > 0]
            print(
                f"Lrave ({len(sorted_lrave)} in total, in which {len(sorted_lrave_hasvalue)} has positive value):"
            )
            for i, (key, value) in enumerate(sorted_lrave):
                print(f"\t {i+1}. {key}: (N={value.N}, mean={value.mean})")
        elif command.startswith("expand"):
            try:
                depth = int(command.split()[1])
                current_node._node.expand(depth)
                print(f"Expanded {depth} layers from current node.")
            except:
                print("Invalid command.")
        elif command.startswith("simul"):
            try:
                if current_node.is_dead_end(self._mcts._treenode_store):
                    print("Dead end, no need to simulate. ")
                else:
                    for retry_count in tqdm(itertools.count()):
                        result = self._mcts._simulate(path, current_node)
                        if result:
                            break
                    print(f"Take {retry_count} times to find a final node.")
            except Exception as e:
                print(f"Invalid command. {e}")
        elif command.startswith("rollout"):
            path, trials = self._mcts.do_rollout()
            print(f"Rollout find {path}")
        elif command == "graphviz":
            if current_node._is_mid:
                print("Warning: graphviz is intended to run on non-mid nodes.")
                return
            # generate graphviz file and print it
            file_name = os.path.join(self.working_dir, "preview.dot")
            current_node._node.generate_graphviz(file_name, "preview")
            print(f"Generated as {file_name}.")
        elif command.startswith("v"):
            # go to child node
            try:
                tree_next = command.split()[1]
                if current_node._is_mid:
                    tree_next = int(tree_next)
                else:
                    tree_next = self._serializer.deserialize_type(tree_next)
                child_node, _ = current_node.get_child(
                    tree_next,
                    self._mcts._treenode_store,
                    auto_initialize=True,
                    on_tree=self.on_tree,
                )
                if child_node is None:
                    print(f"Child {tree_next} does not exist.")
                else:
                    path = path.concat(tree_next)
                    self.node_hierarchy.append((path, child_node))
            except:
                print("Invalid command.")
        elif command == "best":
            children = current_node.get_children(
                self._mcts._treenode_store, auto_initialize=True, on_tree=self.on_tree
            )
            best_children = max(children, key=lambda x: x[2].mean)
            nxt, child_node, edge_state = best_children
            if current_node._is_mid:
                child = Next(current_node._type, nxt)
                print(
                    f"The best child is {child}:\t{current_node._node.get_child_description(child)}, edge(N={edge_state.N}, mean={edge_state.mean}, std={edge_state.std})"
                )
            else:
                assert isinstance(nxt, Next.Type)
                print(
                    f"The best child is {nxt}:\t{child_node.children_count(self._mcts._treenode_store, on_tree=self.on_tree)} children, edge(N={edge_state.N}, mean={edge_state.mean}, std={edge_state.std})"
                )
        elif command == "best_select":
            children = current_node.get_children(
                self._mcts._treenode_store, auto_initialize=True, on_tree=self.on_tree
            )

            best_children = max(children, key=self.ucb_score)
            nxt, child_node, edge_state = best_children
            if current_node._is_mid:
                child = Next(current_node._type, nxt)
                print(
                    f"The best child in selection is {child}:\t{current_node._node.get_child_description(child)}, edge(N={edge_state.N}, mean={edge_state.mean}, std={edge_state.std}), score={self.ucb_score(best_children)}"
                )
            else:
                assert isinstance(nxt, Next.Type)
                print(
                    f"The best child is {nxt}:\t{child_node.children_count(self._mcts._treenode_store, on_tree=self.on_tree)} children, edge(N={edge_state.N}, mean={edge_state.mean}, std={edge_state.std}), score={self.ucb_score(best_children)}"
                )
        elif command in ["b", "back"]:
            # go back to parent node
            if len(self.node_hierarchy) == 1:
                print("Already at the root, can't go back further")
                return
            self.node_hierarchy = self.node_hierarchy[:-1]
        elif command == "explore_all":
            self.on_tree = False
        elif command == "explore_tree":
            self.on_tree = True
        elif command == "show_zero":
            self.show_zero = True
        elif command == "hide_zero":
            self.show_zero = False
        elif command == "exit":
            # exit interactive mode
            exit(0)
        elif command in ["h", "help"]:
            # print help
            print(self.interactive.__doc__)
        else:
            print(f"Invalid command {command}. Enter `help` for help.")

    def interactive(self, working_dir: str = ".", prefix: List[str] = []) -> None:
        """
        Interactive exploration.
        The current path is displayed.
        Several commands:
        - `(i)nfo`: print the description of the node.
        - `(sel)ect`: print the ucd score of each children.
        - `(c)hildren`: list all children of the current node.
        - `(g)rave [<ty>]`: print the grave dictionary. Example: `grave Share`
        - `(l)rave [<ty>]`: print the lrave dictionary. Example: `lrave Share`
        - `graphviz`: generate a graphviz file and print it for the current node.
        - `(v)isit <ty>(<key>)`: go to the child `Next(ty, key)` with the given type and key. Example: `visit Share(0)`
        - `(b)ack`: go back to the parent.
        - `expand [<depth>]`: expand the current node. Example: `expand 2`
        - `explore_tree`: see only visible nodes.
        - `explore_all`: see all nodes.
        - `exit`: exit the interactive mode.
        """
        print("Type `help` for help.")
        os.makedirs(working_dir, exist_ok=True)
        self.working_dir = working_dir
        start = time()
        for command in prefix:
            self.handle_one_command(command)
        print(f"Prefix: Elapsed {time() - start} seconds. ")
        while True:
            command = input(f"({self.node_hierarchy[-1][0]}) >>> ")
            start = time()
            self.handle_one_command(command)
            print(f"Elapsed {time() - start} seconds. ")
