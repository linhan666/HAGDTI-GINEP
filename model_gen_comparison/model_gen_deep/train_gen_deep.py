"""Training entry point for the sequence-based Deep generation model."""
import argparse
from train_gen_deep_nowide import train, build_parser as _build_parser


def build_parser():
    parser = _build_parser()
    parser.set_defaults(output_dir="./output_fusion_final_deep")
    parser.add_argument("--esm_cls_path", help=argparse.SUPPRESS)
    parser.add_argument("--morgan_dim", type=int, default=1024, help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    train(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
