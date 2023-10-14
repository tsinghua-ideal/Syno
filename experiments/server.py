import json
import logging
import urllib.parse
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

from base import log, parser, models, mem
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
        getattr(self, func_name)(*request[2:])

    def do_POST(self):
        remote_ip = self.client_address[0]
        logging.info(f"Incoming POST request {self.path} from {remote_ip} ...")
        func_name = urllib.parse.urlparse(self.path).path.split("/")[1]
        getattr(self, func_name)()

    def sample(self):
        response = {"path": self.session.sample()}
        logging.info(f'Path sampled: {response["path"]}')
        self.session.save(force=False)
        self.session.print_stats(force=False)

        # Send response
        try:
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        except BrokenPipeError:
            logging.debug(
                f"Encountered BrokenPipeError while processing {response['path']}"
            )
            self.session.update(response["path"], -1.0, int(1e9), int(1e9), "EMPTY")
    
    def fetch(self, path: str):
        """
        Fetch a record. 
        """
        file_directory = self.session.path_to_file(path)
        try:
            self.send_response(200)
            self.send_header("Content-type", "application/octet-stream")
            self.end_headers()
            with open(file_directory, 'rb') as f:
                while True:
                    file_data = f.read(32768) # use an appropriate chunk size
                    if file_data is None or len(file_data) == 0:
                        break
                    self.wfile.write(file_data) 
        except BrokenPipeError:
            logging.debug(
                f"Encountered BrokenPipeError while processing {path}"
            )
        

    def reward(self):
        params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        path, accuracy, flops, nparams, kernel_dir = (
            params["path"][0],
            float(params["accuracy"][0]),
            int(params["flops"][0]),
            int(params["params"][0]),
            params["kernel_dir"][0]
        )
        logging.info(
            f"Path received to /reward request: {path}, accuracy: {accuracy}, flops: {flops}, params: {nparams}, kernel_dir: {kernel_dir}"
        )
        self.session.update(path, accuracy, flops, nparams, kernel_dir)
        self.session.save(force=False)
        self.session.print_stats(force=False)

        # Send response
        self.send_response(200)
        self.send_header("Content-type", "text")
        self.end_headers()


def main():

    # Sampler
    model, sampler = models.get_model(args, return_sampler=True)

    # Get search session
    logging.info("Starting search session ...")
    session = get_session(sampler, model, args)

    # Start server
    logging.info(
        f"Starting server at {args.kas_server_addr}:{args.kas_server_port} ..."
    )
    server_address = (args.kas_server_addr, args.kas_server_port)
    server = HTTPServer(server_address, lambda *args: Handler(session, *args))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        session.save()
        session.print_stats()
        logging.info("Shutting down server ...")


if __name__ == "__main__":
    # Logging
    log.setup()

    # Arguments
    args = parser.arg_parse()

    # Set memory limit
    mem.memory_limit(args.server_mem_limit)

    try:
        main()
    except MemoryError:
        sys.stderr.write("\n\nERROR: Memory Exception\n")
        sys.exit(1)
