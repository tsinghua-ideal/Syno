import json
import logging
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

from base import log, parser, models
from search import get_session


class Handler(BaseHTTPRequestHandler):
    def __init__(self, session, *args):
        self.session = session
        super().__init__(*args)

    def do_GET(self):
        remote_ip = self.client_address[0]
        logging.info(f"Incoming GET request {self.path} from {remote_ip} ...")
        func_name = urllib.parse.urlparse(self.path).path.split("/")[1]
        getattr(self, func_name)()

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
            self.session.update(response["path"], -1.0, int(1e9), int(1e9))

    def reward(self):
        params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        path, accuracy, flops, nparams = (
            params["path"][0],
            float(params["accuracy"][0]),
            int(params["flops"][0]),
            int(params["params"][0]),
        )
        logging.info(
            f"Path received to /reward request: {path}, accuracy: {accuracy}, flops: {flops}, params: {nparams}"
        )
        self.session.update(path, accuracy, flops, nparams)
        self.session.save(force=False)
        self.session.print_stats(force=False)

        # Send response
        self.send_response(200)
        self.send_header("Content-type", "text")
        self.end_headers()


if __name__ == "__main__":
    # Logging
    log.setup()

    # Arguments
    args = parser.arg_parse()

    # Sampler
    _, sampler = models.get_model(args, return_sampler=True)

    # Get search session
    logging.info("Starting search session ...")
    session = get_session(sampler, args)

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
