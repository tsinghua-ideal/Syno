import json
import logging
import os
import ctypes
import time
import shutil
import urllib.parse
from traceback import format_exc
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import List, Tuple

from base import log, parser, models, mem, dataset
from base.models.model import KASModel

from KAS import Path, Sampler


def collect_kernels(args) -> List[Tuple[os.PathLike, str]]:
    all_kernels = []
    for dir in args.dirs:
        kernels = []
        for kernel_fmt in os.listdir(dir):
            kernel_dir = os.path.join(dir, kernel_fmt)
            if not os.path.isdir(kernel_dir):
                continue
            if "ERROR" in kernel_dir:
                continue
            if "cache" in kernel_dir:
                continue
            if "rej" in kernel_dir:
                continue
            if "uncanon" in kernel_dir:
                continue
            files = list(os.listdir(kernel_dir))
            assert "graph.dot" in files and "loop.txt" in files and "meta.json" in files

            meta_path = os.path.join(kernel_dir, "meta.json")
            with open(meta_path, "r") as f:
                meta = json.load(f)

            if meta["accuracy"] >= args.kas_min_accuracy:
                kernels.append((kernel_dir, meta["path"]))
        all_kernels.extend(kernels)

    return all_kernels


class index_generator:
    def __init__(self) -> None:
        self.index = 0


class ReevaluateHandler(BaseHTTPRequestHandler):
    def __init__(
        self,
        model: KASModel,
        sampler: Sampler,
        kernels: List[Tuple[os.PathLike, str]],
        cache_dir: os.PathLike,
        save_dir: os.PathLike,
        id: index_generator,
        *args,
    ):
        self.sampler = sampler
        self.model = model
        self.kernels = kernels
        self.cache_dir = cache_dir
        self.idgen = id
        self.save_dir = save_dir
        self.path_to_evaluate = [path for _, path in kernels]
        self.path2dir = {path: kernel_dir for kernel_dir, path in kernels}
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
        if self.idgen.index < len(self.path_to_evaluate):
            path = self.path_to_evaluate[self.idgen.index]
            self.idgen.index += 1
            logging.info(f"Sampled kernel {self.idgen.index}. ")
        else:
            path = "end"
        if path != "end" and self.sampler.visit(Path.deserialize(path)) is None:
            logging.warning(f"{path} does not exist! ")
            path = "retry"
        response = {"path": path}
        logging.info(f'Path sampled: {response["path"]}')

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
            node = self.sampler.visit(Path.deserialize(path))
            kernel = self.sampler.realize(self.model, node)
            directory = kernel.get_directory()
            working_dir = os.path.join(self.cache_dir, os.path.basename(directory))
            file_directory = os.path.join(working_dir, "kernel.tar.gz")
            kernel.archive_to(file_directory, overwrite=False)
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

        node = self.sampler.visit(Path.deserialize(path)).to_node()

        os.makedirs(self.save_dir, exist_ok=True)

        acc_str = (
            ("0" * max(0, 5 - len(f"{int(accuracy * 10000)}")))
            + f"{int(accuracy * 10000)}"
            if accuracy >= 0
            else "ERROR"
        )
        hash_str = f"{ctypes.c_size_t(hash(path)).value}"
        kernel_save_dir = os.path.join(self.save_dir, "_".join([acc_str, hash_str]))
        if not (os.path.exists(kernel_save_dir) and acc_str == "ERROR"):
            os.makedirs(kernel_save_dir, exist_ok=True)

            # GraphViz
            node.generate_graphviz_as_final(
                os.path.join(kernel_save_dir, "graph.dot"), "kernel"
            )

            # Loop
            with open(os.path.join(kernel_save_dir, "loop.txt"), "w") as f:
                f.write(str(node.get_nested_loops_as_final()))

            # Meta information
            with open(os.path.join(kernel_save_dir, "meta.json"), "w") as f:
                json.dump(
                    {
                        "path": path,
                        "accuracy": accuracy,
                        "flops": flops,
                        "params": nparams,
                        "kernel_flag": kernel_flag,
                        "loss": 0,
                        "original_dir": self.path2dir[path],
                    },
                    f,
                    indent=2,
                )

            # copying kernel dir
            if kernel_flag != "MOCKPATH":
                shutil.copytree(
                    self.sampler.realize(self.model, node).get_directory(),
                    os.path.join(kernel_save_dir, "kernel_scheduler_dir"),
                )

        # Send response
        self.send_response(200)
        self.send_header("Content-type", "text")
        self.end_headers()


def main(args):
    sample_input = None
    if "gcn" in args.model:
        train_dataloader, _ = dataset.get_dataloader(args)
        sample_input = train_dataloader

    # Fetching kernels
    kernels = collect_kernels(args)
    logging.info(f"collected {len(kernels)} kernels. ")

    if os.path.exists(args.kas_server_save_dir):
        for kernel_dir in os.listdir(args.kas_server_save_dir):
            meta_path = os.path.join(args.kas_server_save_dir, kernel_dir, "meta.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path, "r") as f:
                meta = json.load(f)
            for _, path in kernels:
                if meta["path"] == path:
                    kernels.pop(kernels.index((_, path)))
                    break
    logging.info(f"Need to evaluate {len(kernels)} kernels")

    # Sampler
    logging.info("Preparing sampler ...")
    model, sampler = models.get_model(
        args, return_sampler=True, sample_input=sample_input
    )

    cache_dir = args.kas_send_cache_dir
    server_save_dir = args.kas_server_save_dir
    id = index_generator()

    count = 0
    logging.info("Testing if all kernels are in the search space ...")
    for _, path in kernels:
        if sampler.visit(Path.deserialize(path)) is None:
            logging.info(f"{path} not in space. ")
            for parent_path in Path.deserialize(path).hierarchy:
                if sampler.visit(parent_path) is None:
                    logging.info(f"{parent_path} is the first one that is not in space. ")
                    break
            count += 1
    if count == 0:
        logging.info("All kernels are in the search space. ")
    else:
        logging.warning(f"{count} kernels not in space. ")

    # Start server
    logging.info(
        f"Starting server at {args.kas_server_addr}:{args.kas_server_port} ..."
    )
    server_address = (args.kas_server_addr, args.kas_server_port)
    server = HTTPServer(
        server_address,
        lambda *args: ReevaluateHandler(
            model, sampler, kernels, cache_dir, server_save_dir, id, *args
        ),
    )
    server.serve_forever()


if __name__ == "__main__":
    # Logging
    log.setup()

    # Arguments
    args = parser.arg_parse()

    # Set memory limit
    mem.memory_limit(args.server_mem_limit)

    main(args)
