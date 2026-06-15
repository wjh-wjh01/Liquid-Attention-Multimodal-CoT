#!/usr/bin/env python3
from __future__ import annotations

import argparse

from ct_mcot.data import SyntheticConfig, make_synthetic_reachability


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic multimodal reachability data.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-examples", type=int, default=1000)
    parser.add_argument("--num-tokens", type=int, default=24)
    parser.add_argument("--input-dim", type=int, default=128)
    parser.add_argument("--branching-factor", type=int, default=3)
    parser.add_argument("--path-length", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    cfg = SyntheticConfig(
        num_examples=args.num_examples,
        num_tokens=args.num_tokens,
        input_dim=args.input_dim,
        branching_factor=args.branching_factor,
        path_length=args.path_length,
        seed=args.seed,
    )
    make_synthetic_reachability(cfg, args.output)


if __name__ == "__main__":
    main()
