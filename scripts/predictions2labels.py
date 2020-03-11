import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import safitty


def build_args(parser):
    parser.add_argument("--in-npy", type=Path, required=True)
    parser.add_argument("--in-csv-infer", type=Path, required=True)
    parser.add_argument("--in-csv-train", type=Path, required=True)
    parser.add_argument("--in-tag2cls", type=Path, required=True)
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threshold", default=0.95, type=float)
    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)


def path2name(x):
    return Path(x).name


def main(args, _=None):
    logits = np.load(args.in_npy, mmap_mode="r")
    probs = softmax(logits)
    confidence = np.max(probs, axis=1)
    preds = np.argmax(logits, axis=1)

    df_infer = pd.read_csv(args.in_csv_infer)
    df_train = pd.read_csv(args.in_csv_train)

    df_infer["filename"] = df_infer["filepath"].map(lambda x: Path(x).name)
    df_train["filename"] = df_train["filepath"].map(lambda x: Path(x).name)

    tag2lbl = safitty.load(args.in_tag2cls)
    cls2tag = {int(v): k for k, v in tag2lbl.items()}

    df_infer["tag"] = [cls2tag[x] for x in preds]
    df_infer["confidence"] = confidence

    train_filepath = df_train["filename"].tolist()
    df_infer = df_infer[~df_infer["filename"].isin(train_filepath)]

    if df_infer.shape[0] == 0:
        raise NotImplementedError(
            "Pseudo Lgabeling done. Nothing more to label."
        )

    counter_ = 0
    for i, row in df_infer.iterrows():
        if row["confidence"] < args.threshold:
            continue

        filepath_src = args.in_dir / row["filepath"]
        filename = filepath_src.name
        filepath_dst = args.out_dir / row["tag"] / filename
        filepath_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(filepath_src, filepath_dst)

        counter_ += 1
    print(f"Predicted: {counter_} ({100 * counter_ / len(df_infer):2.2f}%)")


if __name__ == "__main__":
    args = parse_args()
    main(args)
