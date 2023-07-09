import logging
import json
import os

from KAS import Sampler, MCTS, TreeExplorer

def explore(sampler: Sampler, path:str="mcts.json"):
    assert os.path.exists(path), f"{path} is not a file"
    
    mcts_serialize = json.load(open(path, "r"))
    mcts = MCTS.deserialize(mcts_serialize, sampler)
    
    explorer = TreeExplorer(mcts)
    try:
        explorer.interactive()
    except KeyboardInterrupt:
        pass
