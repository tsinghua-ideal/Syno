import math
import json
import logging
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from KAS import MCTS, TreePath

from utils import parser, models


class MCTSSession:
    # TODO: may move to program arguments
    virtual_loss_constant = 5.
    leaf_parallelization_number = 5
    exploration_weight = math.sqrt(2)
    b = 0.4
    c_l = 40.

    def __init__(self, args) -> None:
        _, sampler = models.get_model_and_sampler(args)
        self.mcts = MCTS(sampler, self.virtual_loss_constant, self.leaf_parallelization_number,  self.exploration_weight, self.b, self.c_l)
        self.best_node = None
        logging.info('MCTS initialized')

    def has_eval_result(self, node):
        # TODO: may have a better way to check
        return node.reward >= 0
    
    def is_dead(self, node):
        assert node.is_final()
        return node.filtered
    
    def get_eval_result(self, node):
        assert not self.check_dead(node)
        assert self.has_eval_result(node)
        return node.reward

    def set_eval_result(self, node, val):
        assert 0 <= val <= 1
        assert not self.has_eval_result(node)
        node.reward = val

    def update_result(self, meta, reward, path):
        trial, receipt = meta
        assert trial.is_final()
        if reward < 0:
            # TODO: may unify the order & API
            # TODO: what is receipt?
            self.mcts.remove(receipt, trial)
        else:
            self.set_eval_result(trial, reward)
            if self.best_node is None or self.best_node.reward < reward:
                self.best_node = trial
            self.mcts.back_propagate(receipt, reward, path)
            # TODO: may print in a `logging` style
            # Statistics.Print()

    def launch_new_iteration(self):
        logging.info('Launching new iteration ...')
        start_time = time()

        rollout = self.mcts.do_rollout(self.mcts._sampler.root())
        if rollout is None:
            return None

        new_paths = dict()
        receipt, trials = rollout        
        for path, node in trials:
            if not self.has_eval_result(node):
                new_paths[path.serialize()] = (node, receipt)
            else:
                assert not self.is_dead(node)
                reward = self.get_eval_result(node)
                self.mcts.back_propagate(receipt, reward, path)

        logging.info(f'Iteration finished in {time() - start_time} seconds')
        return new_paths

    # TODO: load/store from file


class Handler(BaseHTTPRequestHandler):
    # TODO: may move to program arguments
    # TODO: may merge failure and timeout cases
    failure_reward = -1
    timeout_reward = -1
    timeout_limit = 7200

    def __init__(self, mcts_session, *args):
        super().__init__(*args)
        self.mcts_session = mcts_session
        self.pending_evaluation = dict()
        self.waiting_result = dict()

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
        # Check timeout kernels
        timeout_paths = []
        for path, pack in self.waiting_result.items():
            if time.time() - pack['time'] > self.timeout_limit:
                logging.info(f'Kernel timeout: {TreePath.deserialize(path)}')
                self.update_result(pack, self.timeout_reward)
                timeout_paths.append(path)
        for path in timeout_paths:
            self.waiting_result.pop(path)

        # Sample a kernel implement
        while len(self.pending_evaluation) == 0:
            new_paths = self.mcts_session.launch_new_iteration()
            if new_paths is not None:
                self.pending_evaluation.update(new_paths)
        response = dict()
        if len(self.pending_evaluation) > 0:
            path, meta = self.pending_evaluation.popitem()
            logging.info(f'Response kernel: {TreePath.deserialize(path)}')
            self.waiting_result[path] = {'meta': meta, 'time': time.time()}

        # Send response
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def reward(self):
        params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        path, reward = params['path'][0], float(params['reward'][0])
        logging.info(f'Kernel: {TreePath.deserialize(path)}, reward: {reward}')

        # Update MCTS
        assert path in self.waiting_result
        assert path not in self.pending_evaluation
        self.update_result(self.waiting_result.pop(path), reward)
        
        # Send response
        self.send_response(200)
        self.send_header('Content-type', 'text')
        self.end_headers()


if __name__ == '__main__':
    args = parser.arg_parse()

    logging.info('Preparing model ...')
    model, sampler = models.get_model_and_sampler(args)
    
    # TODO: resume search
    # TODO: MCTS session
    logging.info('Starting search ...')
    session = MCTSSession(args)

    logging.info('Starting server ...')
    server_address = (args.kas_server_addr, args.kas_server_port)
    server = HTTPServer(server_address, lambda *args: Handler(session, *args))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info('Shutting down server ...')
