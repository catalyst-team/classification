# usage:
# python scripts/prepare_config.py \
#     --in-template=./configs/templates/focal.yml \
#     --out-config=./configs/_class.yml \
#     --expdir=./src \
#     --dataset-path=./data \
#     --num-workers=4 \
#     --batch-size=64 \
#     --max-image-size=224 \
#     --balance-strategy=1024

import argparse
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
import safitty


def build_args(parser):
    """Constructs the command-line arguments for ``prepare_config``."""
    parser.add_argument("--in-template", type=Path, required=True)
    parser.add_argument("--out-config", type=Path, required=True)
    parser.add_argument("--expdir", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)

    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--max-image-size", default=224, type=int)
    parser.add_argument("--balance-strategy", default="null", type=str)
    parser.add_argument("--criterion", default="FocalLossMultiClass", type=str)

    return parser


def parse_args():
    """Parses the command line arguments for the main method."""
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def render_config(
    in_template: Path,
    out_config: Path,
    expdir: Path,
    dataset_path: Path,
    num_workers: int,
    batch_size: int,
    max_image_size: int,
    balance_strategy: str,
    criterion: str,
):
    """Render catalyst config with specified parameters."""
    template_path = str(in_template.absolute().parent)
    env = Environment(
        loader=FileSystemLoader([template_path]),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    template = env.get_template(in_template.name)

    tag2class = safitty.load(dataset_path / "tag2class.json")
    num_classes = len(tag2class)
    class_names = [
        key for key, _ in sorted(tag2class.items(), key=lambda x: x[1])
    ]

    out_config.parent.mkdir(parents=True, exist_ok=True)

    out_config.write_text(
        template.render(
            expdir=str(expdir),
            dataset_path=str(dataset_path),
            num_classes=num_classes,
            class_names=class_names,
            num_workers=num_workers,
            batch_size=batch_size,
            max_image_size=max_image_size,
            balance_strategy=balance_strategy,
            criterion=criterion,
        )
    )


def main(args, _=None):
    """Run the ``prepare_config`` script."""
    args = args.__dict__
    render_config(**args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
