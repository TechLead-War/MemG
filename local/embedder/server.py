#!/usr/bin/env python3
"""Local embedding gRPC server powered by sentence-transformers.

Usage:
    python server.py                              # default: all-MiniLM-L6-v2 on port 50051
    python server.py --model all-mpnet-base-v2    # use a different model
    python server.py --port 50052 --device cuda   # custom port and GPU
"""

import argparse
import signal
import sys
from concurrent import futures

import grpc
from sentence_transformers import SentenceTransformer


def generate_stubs():
    """Generate proto stubs if they don't exist."""
    try:
        import embedder_pb2  # noqa: F401
    except ImportError:
        import subprocess
        import os

        proto_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "proto", "embedder.proto"
        )
        proto_path = os.path.abspath(proto_path)
        out_dir = os.path.dirname(__file__) or "."

        subprocess.run(
            [
                sys.executable,
                "-m",
                "grpc_tools.protoc",
                f"-I{os.path.dirname(proto_path)}",
                f"--python_out={out_dir}",
                f"--grpc_python_out={out_dir}",
                proto_path,
            ],
            check=True,
        )
        print("Generated proto stubs.")


def parse_args():
    parser = argparse.ArgumentParser(description="Local embedding gRPC server")
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name or path (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="gRPC port to listen on (default: 50051)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device: cpu, cuda, cuda:0, mps, etc. (default: auto-detect)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Max gRPC worker threads (default: 4)",
    )
    return parser.parse_args()


def main():
    generate_stubs()

    import embedder_pb2_grpc
    from service import EmbedderServicer

    args = parse_args()

    print(f"Loading model: {args.model}")
    if args.device:
        model = SentenceTransformer(args.model, device=args.device)
    else:
        model = SentenceTransformer(args.model)

    dim = model.get_sentence_embedding_dimension()
    print(f"Model loaded: {args.model} (dimension={dim}, device={model.device})")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.workers))
    servicer = EmbedderServicer(model, args.model)
    embedder_pb2_grpc.add_EmbedderServiceServicer_to_server(servicer, server)

    address = f"[::]:{args.port}"
    server.add_insecure_port(address)
    server.start()
    print(f"Serving on port {args.port}")

    def shutdown(signum, frame):
        print("\nShutting down...")
        server.stop(grace=5)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    server.wait_for_termination()


if __name__ == "__main__":
    main()
