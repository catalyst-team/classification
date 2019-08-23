#!/usr/bin/env python
# usage:
# python prepare_config.py \
#     --in-template=./configs/templates/class.yml \
#     --out-config=./configs/_class.yml \
#     --expdir=./src \
#     --dataset-path=./data \
#     --max-image-size=224 \
#     --num-workers=4 \
#     --batch-size=64

import json
import argparse
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


def build_args(parser):
    parser.add_argument("--in-template", type=Path, required=True)
    parser.add_argument("--out-config", type=Path, required=True)
    parser.add_argument("--expdir", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)

    parser.add_argument("--max-image-size", default=224, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--batch-size", default=64, type=int)

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def render_config(
    in_template: Path,
    out_config: Path,
    expdir: Path,
    dataset_path: Path,
    max_image_size: int,
    num_workers: int,
    batch_size: int,
):
    _template_path = in_template.absolute().parent

    _env = Environment(
        loader=FileSystemLoader(
            [str(_template_path)]
        ),
        trim_blocks=True, lstrip_blocks=True
    )

    template = _env.get_template(in_template.name)
    with (dataset_path / "tag2class.json").open() as f:
        num_classes = len(json.load(f))

    out_config.parent.mkdir(parents=True, exist_ok=True)

    out_config.write_text(template.render(
        expdir=str(expdir),
        dataset_path=str(dataset_path),
        num_classes=num_classes,
        max_image_size=max_image_size,
        num_workers=num_workers,
        batch_size=batch_size,
    ))


def main(args, _=None):
    args = args.__dict__
    render_config(**args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
