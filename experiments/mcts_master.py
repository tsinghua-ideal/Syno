import math
import json
import logging
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from KAS import MCTS, TreePath

from utils import log, parser, models


class MCTSSession:
    # TODO: may move to program arguments
    # TODO: load/store from file
    # TODO: add random sample
    virtual_loss_constant = 5.
    leaf_parallelization_number = 5
    exploration_weight = math.sqrt(2)
    max_iterations = 100
    b = 0.4
    c_l = 40.

    def __init__(self, args) -> None:
        _, sampler = models.get_model(args, return_sampler=True)
        self.mcts = MCTS(sampler, self.virtual_loss_constant, self.leaf_parallelization_number,  self.exploration_weight, self.b, self.c_l)
        self.best = None
        self.pending, self.waiting = set(), set()
        self.path_meta_data = dict()

    def update_reward(self, path, reward):
        logging.info(f'Updating path: {path}, reward: {reward}')
        assert path in self.waiting
        assert path not in self.pending
        self.waiting.remove(path)
        root, leaf_paths, node = self.path_meta_data[path]
        path = TreePath.deserialize(path)

        # Update node    
        node.reward = reward
        if self.best is None or self.best.reward < reward:
            self.best = node

        # Back propagate
        if reward < 0:
            for leaf_path in leaf_paths:
                self.mcts.remove(receipt=(root, leaf_path), trial=node)    
        else:
            for leaf_path in leaf_paths:
                self.mcts.back_propagate(receipt=(root, leaf_path), reward=reward, path_to_trial=path)

    def launch_new_iteration(self):
        logging.info('Launching new iteration ...')
        start_time = time.time()

        rollout = self.mcts.do_rollout(self.mcts._sampler.root())
        if rollout is None:
            return None

        new_paths = dict()
        assert len(rollout) > 0
        (root, leaf_path), trials = rollout
        for path, node in trials:
            assert node.is_final()
            if node.reward < 0:
                # Unevaluated
                new_paths[path.serialize()] = (root, leaf_path, node)
            else:
                # Already evaluated, back propagate
                assert not node.filtered
                self.mcts.back_propagate(receipt=(root, leaf_path), reward=node.reward, path_to_trial=path)

        logging.info(f'Iteration finished in {time.time() - start_time} seconds')
        return new_paths
    
    def get_new_path(self):
        n_iterations = 0
        while len(self.pending) == 0:
            n_iterations += 1
            new_paths = self.launch_new_iteration()
            if n_iterations > self.max_iterations:
                return 'retry'
            elif new_paths is None:
                return 'end'
            
            for path, (root, leaf_path, node) in new_paths.items():
                if path not in self.waiting:
                    self.pending.add(path)
                if path not in self.path_meta_data:
                    self.path_meta_data[path] = (root, [], node)
                self.path_meta_data[path][1].append(leaf_path)

        assert len(self.pending) > 0
        assert len(self.waiting.intersection(self.pending)) == 0
        path = self.pending.pop()
        self.waiting.add(path)
        return path


class Handler(BaseHTTPRequestHandler):
    def __init__(self, mcts_session, *args):
        self.mcts_session = mcts_session
        super().__init__(*args)

    def do_GET(self):
        remote_ip = self.client_address[0]
        logging.info(f'Incoming GET request {self.path} from {remote_ip} ...')
        func_name = urllib.parse.urlparse(self.path).path.split('/')[1]
        getattr(self, func_name)()

    def do_POST(self):
        remote_ip = self.client_address[0]
        logging.info(f'Incoming POST request {self.path} from {remote_ip} ...')
        func_name = urllib.parse.urlparse(self.path).path.split('/')[1]
        getattr(self, func_name)()

    def update_result(self, pack, reward):
        path, meta = pack['path'], pack['meta']
        self.mcts_session.update(meta, reward, TreePath.deserialize(path))

    def sample(self):
        response = dict()
        new_path = self.mcts_session.get_new_path()
        if new_path:
            response['path'] = new_path
            logging.info(f'Path returned to /sample request: {new_path}')
        else:
            logging.info('No path returned to /sample request')

        # Send response
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def reward(self):
        params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        print(params)
        path, reward = params['path'][0], float(params['value'][0])
        print(path, reward)
        self.mcts_session.update_reward(path, reward)
        
        # Send response
        self.send_response(200)
        self.send_header('Content-type', 'text')
        self.end_headers()


if __name__ == '__main__':
    # Logging
    log.setup()
    
    # Arguments
    args = parser.arg_parse()
    
    # TODO: resume search
    # TODO: MCTS session
    logging.info('Starting MCTS session ...')
    session = MCTSSession(args)

    logging.info(f'Starting server at {args.kas_server_addr}:{args.kas_server_port} ...')
    server_address = (args.kas_server_addr, args.kas_server_port)
    server = HTTPServer(server_address, lambda *args: Handler(session, *args))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info('Shutting down server ...')
