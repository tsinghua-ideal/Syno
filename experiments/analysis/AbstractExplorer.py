from abc import ABC, abstractmethod
from dataclasses import dataclass
import itertools
import re
from typing import Dict, Generic, List, Optional, Tuple, TypeVar, Union


@dataclass
class Predicate:
    """
    A predicate represents a function of the explorer.
    """
    name: str
    help_message: str = ""
    # Other args of the corresponding function. None means no default value, and str means default value.
    additional_args: Tuple[Optional[str], ...] = tuple()
    can_work_if_state_invalid: bool = False

@dataclass
class Child:
    """
    A child of a state.
    """
    value: str
    caption: str
    label: Optional[str] = None

T = TypeVar("T")

class AbstractExplorer(ABC, Generic[T]):
    """
    Abstract explorer class. Note that this class shall be stateless.
    Basically a state machine, it is just that this class only defines the transition table. The state is T. For a simple explorer, the state can be the current node in the tree. An explorer abstracts a DAG.
    You must implement the abstract methods.
    You can also add your own methods, by filling them in available_custom_predicates(). They should return Union[str, Tuple[str, List[str]]], that is, the message and optionally the next state.
    """

    @abstractmethod
    def __init__(self):
        """
        Add your own data structures. Sampler for example.
        DO NOT KEEP ANY STATE!
        ANY ATRRIBUTES SHALL BE IMMUTABLE!
        """
        pass

    @abstractmethod
    def state_of(self, current_path: List[str]) -> Optional[T]:
        """
        Return the state corresponding to the given path.
        """
        pass

    @abstractmethod
    def children(self, state: T) -> List[Child]:
        """
        Return the children of the current state.
        The children can be optionally grouped by labels.
        """
        pass

    @abstractmethod
    def info(self, state: T) -> str:
        """
        Give information about the current state.
        """
        pass

    @abstractmethod
    def available_custom_predicates(self) -> Tuple[Predicate, ...]:
        """
        Return the custom predicates that can be used in the explorer.
        """
        pass

    def execute(self, state: Optional[T], predicate: str, args: List[str]) -> Union[str, Tuple[str, List[str]]]:
        """
        Apply the predicate to the current state.
        Can return the next state.
        """
        found = [p for p in self.available_custom_predicates() if p.name == predicate]
        assert len(found) == 1, f"Invalid command {predicate}."
        found = found[0]
        assert state is not None or found.can_work_if_state_invalid, "This node does not exist."
        assert len(args) == len(found.additional_args), f"Predicate {predicate} expects {len(found.additional_args)} arguments, but {len(args)} are given."
        return getattr(self, predicate)(state, *args)

    def help_message(self) -> str:
        """
        Get help message.
        """
        prelude = [
            ("info", "Print the description of the node."),
            ("children", "List all children of the current node. You can also specify a label filter. Example: `children Share`."),
            ("visit", "Go to the specified child."),
            ("root", "Go to the root."),
        ]
        predicates = [(p.name, p.help_message) for p in self.available_custom_predicates()]
        epilogue = [
            ("back", "Go back to the parent."),
            ("exit", "Exit the interactive mode."),
            ("help", "Print this help message.")
        ]
        return "".join(
            f"- `{name}`: {help_message}\n"
            for name, help_message in itertools.chain(prelude, predicates, epilogue)
        )

    def interactive(self) -> None:
        """
        Interactive exploration.
        """
        print("Type `help` for help.")
        current_path: List[str] = []
        while True:
            state = self.state_of(current_path)
            print(f"[{', '.join(current_path)}]")
            if state is None:
                print("Warning: this node does not exist.")

            # Handle command.
            command = input(">>> ")
            decomposed_command = list(filter(None, command.split(' ', 1)))
            if len(decomposed_command) == 0:
                continue
            elif len(decomposed_command) == 1:
                predicate = decomposed_command[0]
                args = []
            else:
                predicate, args = decomposed_command
                args = list(filter(None, re.split(r'[, ]', args)))

            # dispatch
            try:
                if predicate == "info":
                    if state is None:
                        print("Error: this node does not exist.")
                        continue
                    assert not args
                    print(self.info(state))
                elif predicate == "children":
                    if state is None:
                        print("Error: this node does not exist.")
                        continue
                    children = self.children(state)
                    if args:
                        label = args[0]
                        children = [child for child in children if child.label == label]
                    for child in children:
                        print(f"\t{child.value}:\t{child.caption}")
                elif predicate == "visit":
                    if state is None:
                        print("Error: this node does not exist.")
                        continue
                    # support multiple comma separated
                    current_path += args
                elif predicate == "root":
                    assert not args
                    current_path = []
                elif predicate == "back":
                    assert not args
                    current_path.pop()
                elif predicate == "exit":
                    assert not args
                    return
                elif predicate == "help":
                    assert not args
                    # print help
                    print(self.help_message())
                else:
                    result = self.execute(state, predicate, args)
                    if isinstance(result, str):
                        print(result)
                    else:
                        message, new_path = result
                        print(message)
                        current_path = new_path
            except KeyboardInterrupt:
                print("KeyboardInterrupt")
                return
            except EOFError:
                print("EOF")
                return
            except Exception as e:
                print(f"Exception raised: {e}")
