import ctypes
import json
import math
import logging
import os
import shutil
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from KAS import MCTS, TreePath

from utils import log, parser, models


class MCTSSession:
    # TODO: may move to program arguments
    # TODO: add random sample
    virtual_loss_constant = 5.
    leaf_parallelization_number = 5
    exploration_weight = math.sqrt(2)
    max_iterations = 100
    b = 0.4
    c_l = 40.

    def __init__(self, args):
        _, self.sampler = models.get_model(args, return_sampler=True)
        self.mcts = MCTS(self.sampler, self.virtual_loss_constant, self.leaf_parallelization_number,  self.exploration_weight, self.b, self.c_l)
        self.pending, self.waiting = set(), set()
        self.path_meta_data = dict()
        self.reward_power = args.kas_reward_power
        self.last_save_time = time.time()
        self.save_interval = args.kas_server_save_interval
        self.save_dir = args.kas_server_save_dir
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

    def save(self):
        if self.save_dir is None:
            return

        logging.info(f'Saving MCTS session into {self.save_dir}')
        with open(f'{self.save_dir}/mcts.json', 'w') as f:
            json.dump(self.mcts.serialize(), f, indent=2)

    def try_save(self):
        if time.time() - self.last_save_time > self.save_interval:
            self.save()
            self.last_save_time = time.time()

    def update(self, path, accuracy):
        root, leaf_paths, node = self.path_meta_data[path]
        if self.save_dir:
            score_str = ('0' * max(0, 5 - len(f'{int(accuracy * 10000)}'))) + f'{int(accuracy * 10000)}' if accuracy >= 0 else 'ERROR'
            kernel_save_dir = os.path.join(self.save_dir, f'{score_str}_{ctypes.c_size_t(hash(path)).value}')
            os.makedirs(kernel_save_dir)
            node._node.generate_graphviz_as_final(os.path.join(kernel_save_dir, 'graph.dot'), 'kernel')
            with open(os.path.join(kernel_save_dir, 'loop.txt'), 'w') as f:
                loop_str = node._node.get_nested_loops_as_final()
                f.write(str(loop_str))

        reward = accuracy ** self.reward_power if accuracy > 0 else -1
        logging.info(f'Updating path: {path}, reward: {reward}')
        assert path in self.waiting
        assert path not in self.pending
        self.waiting.remove(path)
        path = TreePath.deserialize(path)
        node.reward = reward

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

    def sample(self):
        response = dict()
        new_path = self.mcts_session.get_new_path()
        if new_path:
            response['path'] = new_path
            logging.info(f'Path returned to /sample request: {new_path}')
        else:
            logging.info('No path returned to /sample request')
        self.mcts_session.try_save()

        # Send response
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def reward(self):
        params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        path, accuracy = params['path'][0], float(params['value'][0])
        logging.info(f'Path received to /reward request: {path}, accuracy: {accuracy}')
        self.mcts_session.update(path, accuracy)
        self.mcts_session.try_save()
        
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
    # TODO: save into directories
    logging.info('Starting MCTS session ...')
    session = MCTSSession(args)

    logging.info(f'Starting server at {args.kas_server_addr}:{args.kas_server_port} ...')
    server_address = (args.kas_server_addr, args.kas_server_port)
    server = HTTPServer(server_address, lambda *args: Handler(session, *args))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        session.save()
        logging.info('Shutting down server ...')
