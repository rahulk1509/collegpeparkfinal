#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Entry point for the CollegePark Parking Environment server.

Usage:
    python run.py
    python run.py --port 7860
    python run.py --host 0.0.0.0 --port 7860

Or with uvicorn directly:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(
        description="CollegePark Parking Environment Server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port number to listen on (default: 7860)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    
    args = parser.parse_args()
    
    print(f"Starting CollegePark Parking Environment Server...")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Workers: {args.workers}")
    print(f"  Reload: {args.reload}")
    print()
    
    uvicorn.run(
        "server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )


if __name__ == "__main__":
    main()
