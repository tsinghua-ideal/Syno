import json
import time

from http.server import BaseHTTPRequestHandler

from KAS import TreePath

waiting_result_cache = {}  # path -> trial, receipt
pending_evaluate_cache = {}  # path -> trial, receipt

class Handler(BaseHTTPRequestHandler):
    def log_message(self, _, *__):
        pass

    def do_GET(self):
        # Assign an available kernel to the client.
        remote_ip = self.connection.getpeername()[0]
        print(f'Incoming GET request from {remote_ip}: {self.path}')
        if self.path != '/kernel' and self.path != '/arguments':
            print(' > Bad request')
            return

        # Detect timeout kernels.
        timeout_kernels = []
        for path, pack in waiting_result_cache.items():
            # 24 hours: 86400 secs
            if time.time() - pack['time'] > 7200:
                timeout_kernels.append(path)
        for path in timeout_kernels:
            assert path not in pending_evaluate_cache
            assert path in waiting_result_cache
            self.mcts.update_result(waiting_result_cache.pop(path)['meta'], -1)
            print(f' > Timeout kernel: {path}')

        if self.path == '/arguments':
            response_json = self.mcts.get_args()
        else:
            response_json = {'path': ''}
            if self.mcts.remain_iterations > 0:
                while len(pending_evaluate_cache.keys()) == 0:
                    pending_evaluate_cache.update(self.mcts.launch_new_iteration())
                selected_path, meta_data = pending_evaluate_cache.popitem()
                print(f' > Response kernel: {TreePath.deserialize(selected_path)}')
                response_json = {'path': selected_path}
                waiting_result_cache[selected_path] = dict(
                    meta=meta_data, time=time.time()
                )
            else:
                response_json['path'] = 'ENDTOKEN'
                print(f' > Train Ended. No available response')

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response_json).encode())

    def do_POST(self):
        # Response success or failure.
        remote_ip = self.connection.getpeername()[0]
        print(f'Incoming POST request from {remote_ip}: {self.path}')
        if not (self.path.startswith('/success?name=') or self.path.startswith('/failure?name=') or
                self.path.startswith('/test')):
            print(' > Bad request')
            return
        if not self.path.startswith('/test'):
            path = self.path[14:]
            assert len(path) > 0
        else:
            path = None
        if self.mcts.remain_iterations <= 0:
            print("> Search ended, not receiving new inputs")
            return
        if self.path.startswith('/success'):
            path, state, reward = path.split('$')
            reward = float(reward)
            assert path not in pending_evaluate_cache
            assert path in waiting_result_cache, f"{path} not in {waiting_result_cache}"
            self.mcts.update_result(waiting_result_cache.pop(path)['meta'], reward ** 2)
            self.mcts.remain_iterations -= 1
            print(f"Remaining iterations: {self.mcts.remain_iterations}")
            if self.mcts.remain_iterations == 0:
                self.mcts.dump_result()
                print(f' > MCTS iteration complete. ')
                return
            print(f' > Successfully trained: {TreePath.deserialize(path)}, accuracy {reward}')
        elif self.path.startswith('/failure'):
            path, state = path.split('$')
            if path in waiting_result_cache:
                self.mcts.update_result(waiting_result_cache.pop(path)['meta'], -1)
            print(f' > Failed to train {path} because of {state}')
        else:
            print(f' > Testing message ...')
        self.send_response(200)
        self.send_header('Content-type', 'text')
        self.end_headers()
