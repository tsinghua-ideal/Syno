import json
import time

from http.server import HTTPServer, BaseHTTPRequestHandler


class Handler(BaseHTTPRequestHandler):
    def log_message(self, _, *__):
        pass

    def do_GET(self):
        # Assign an available kernel to the client.
        remote_ip = self.connection.getpeername()[0]
        print(f'Incoming GET request from {remote_ip}: {self.path}')
        if self.path != '/kernel' or self.path != '/arguments':
            print(' > Bad request')
            return

        # Detect timeout kernels.
        timeout_kernels = []
        for path, pack in self.mcts.waiting_result_cache.items():
            # 24 hours: 86400 secs
            if time.time() - pack['time'] > 86400 * 10:
                timeout_kernels.append(path)
        for path in timeout_kernels:
            assert path not in self.mcts.pending_evaluate_cache
            assert path in self.mcts.waiting_result_cache
            self.mcts.pending_evaluate_cache[path] = self.mcts.waiting_result_cache.pop(
                path)['meta']
            print(f' > Timeout kernel: {path}')

        if self.path == '/arguments':
            response_json = self.mcts.get_args()
        else:
            while len(self.mcts.pending_evaluate_cache.keys()) == 0:
                new_iter_flag = self.mcts.launch_new_iteration()
                if not new_iter_flag:
                    self.mcts.dump_result()
                    print(f' > MCTS iteration complete. ')
            response_json = {'path': ''}
            if self.mcts.end_flag:
                print(f' > No available response')
            else:
                selected_path, meta_data = self.mcts.pending_evaluate_cache.popitem()
                print(f' > Response kernel: {selected_path}')
                response_json = {'path': selected_path}
                self.mcts.waiting_result_cache[selected_path] = dict(
                    meta=meta_data, time=time.time()
                )

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
        if self.path.startswith('/success'):
            assert path not in self.mcts.pending_evaluate_cache
            assert path in self.mcts.waiting_result_cache
            path, reward = path.split('#')
            self.mcts.update_result(path, reward)
            print(f' > Successfully trained: {path}, accuracy {reward}')
        elif self.path.startswith('/failure'):
            assert path not in self.mcts.pending_evaluate_cache
            assert path in self.mcts.waiting_result_cache
            self.mcts.update_result(path)
            print(f' > Failed to train: {path}')
        else:
            print(f' > Testing message ...')
        self.send_response(200)
        self.send_header('Content-type', 'text')
        self.end_headers()
