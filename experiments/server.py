import json
import logging
import os
import urllib.parse
import sys
from traceback import format_exc
from http.server import BaseHTTPRequestHandler, HTTPServer

from KAS import AbstractExplorer
from base import log, parser, models, mem, dataset
from search import get_session, Session


class Handler(BaseHTTPRequestHandler):
    def __init__(self, session: Session, *args):
        self.session = session
        super().__init__(*args)

    def do_GET(self):
        remote_ip = self.client_address[0]
        logging.info(f"Incoming GET request {self.path} from {remote_ip} ...")
        request = urllib.parse.urlparse(self.path).path.split("/")
        func_name = request[1]
        assert not func_name.startswith("_")
        getattr(self, func_name)(*request[2:])

    def do_POST(self):
        remote_ip = self.client_address[0]
        logging.info(f"Incoming POST request {self.path} from {remote_ip} ...")
        func_name = urllib.parse.urlparse(self.path).path.split("/")[1]
        assert not func_name.startswith("_")
        getattr(self, func_name)()

    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def sample(self):
        response = {"path": self.session.sample()}
        logging.info(f'Path sampled: {response["path"]}')
        self.session.save(force=False)
        self.session.print_stats(force=False)

        # Send response
        try:
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        except BrokenPipeError:
            logging.debug(format_exc())
            logging.debug(
                f"Encountered BrokenPipeError while processing {response['path']}"
            )
            self.session.update(
                response["path"], -1.0, int(1e9), int(1e9), "EMPTY", None
            )

    def _send_file(self, file_path: str) -> None:
        self.send_response(200)
        self.send_header("Content-type", "application/octet-stream")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        with open(file_path, "rb") as f:
            while True:
                file_data = f.read(32768)  # use an appropriate chunk size
                if file_data is None or len(file_data) == 0:
                    break
                self.wfile.write(file_data)

    def fetch(self, path: str):
        """
        Fetch a record.
        """
        try:
            file_directory = self.session.path_to_file(path)
        except Exception as e:
            logging.debug(f"Encountered {e}, fetching failed. ")
            logging.debug(format_exc())
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"Exception": str(e)}).encode())
            return
        try:
            self._send_file(file_directory)
        except BrokenPipeError:
            logging.debug(f"Encountered BrokenPipeError while processing {path}")
            logging.debug(format_exc())

    def reward(self):
        params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        path, accuracy, flops, nparams, kernel_flag, loss = (
            params["path"][0],
            float(params["accuracy"][0]),
            int(params["flops"][0]),
            int(params["params"][0]),
            params["kernel_flag"][0],
            float(params["loss"][0]),
        )
        logging.info(
            f"Path received to /reward request: {path}, accuracy: {accuracy}, flops: {flops}, params: {nparams}, kernel_flag: {kernel_flag}, loss: {loss}"
        )
        self.session.update(path, accuracy, flops, nparams, kernel_flag, loss)
        self.session.save(force=False)
        self.session.print_stats(force=False)

        # Send response
        self.send_response(200)
        self.send_header("Content-type", "text")
        self.end_headers()

    def explore(self):
        """
        Invoke explorer. A query parameter `explorer` is required.
        Note that this is stateless.
        """
        parse_results = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parse_results.query)
        explorer = params["explorer"][0]
        if explorer == "search_space":
            explorer = self.session.search_space_explorer
        elif explorer == "algorithm":
            explorer = self.session.algo.explorer
        else:
            raise ValueError(f"Unknown explorer: {explorer}")
        assert isinstance(explorer, AbstractExplorer)

        payload_size = int(self.headers["Content-Length"])
        payload = self.rfile.read(payload_size)
        payload = json.loads(payload.decode())

        response = explorer.serve(
            payload, f"{parse_results.scheme}://{parse_results.netloc}"
        )

        try:
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        except BrokenPipeError:
            logging.debug(f"Encountered BrokenPipeError while processing {payload}")
            logging.debug(format_exc())

    def explorer_files(self, file_name: str):
        """
        Serve files generated by explorer.
        """
        file_name = os.path.basename(file_name)
        file_path = os.path.join(
            self.session.search_space_explorer.working_dir, file_name
        )
        assert os.path.exists(file_path), f"File {file_path} does not exist."
        self._send_file(file_path)


def main():
    sample_input = None
    if "gcn" in args.model:
        train_dataloader, _ = dataset.get_dataloader(args)
        sample_input = train_dataloader

    # Sampler
    logging.info("Preparing sampler ...")
    model, sampler = models.get_model(
        args, return_sampler=True, sample_input=sample_input
    )

    # Get search session
    logging.info("Starting search session ...")
    session = get_session(sampler, model, args)

    # Start server
    logging.info(
        f"Starting server at {args.kas_server_addr}:{args.kas_server_port} ..."
    )
    server_address = (args.kas_server_addr, args.kas_server_port)
    server = HTTPServer(server_address, lambda *args: Handler(session, *args))
    server.serve_forever()


if __name__ == "__main__":
    # Logging
    log.setup()

    # Arguments
    args = parser.arg_parse()

    # Set memory limit
    mem.memory_limit(args.server_mem_limit)

    main()
