import argparse
import numpy as np
import json

from datasets import load_dataset
from run_classifier import METRICS, compute_metrics


def main(prediction_file, output_file, dataset, config, sep=',', split='test', regression=False):
    preds = list()
    with open(prediction_file, "r") as f:
        for line in f:
            line = line.strip().split(sep)
            line = [float(x) for x in line]
            preds.append(line)
    preds = np.array(preds)

    df = load_dataset(dataset, config)
    refs = np.array([item['label'] for item in df[split]])

    metrics = compute_metrics(
        preds,
        refs,
        df[split].features['label'].names,
        bool(regression),
        METRICS
    )

    metrics = dict(map(
        lambda x: (f'{dataset}-{config}_predict_{x[0]}', x[1].item() if isinstance(x[1], np.generic) else x[1]),
        metrics.items()
    ))
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", help="File containing the predictions", type=str)
    parser.add_argument("--output_file", help="Output file to save the results", type=str)
    parser.add_argument("--dataset", help="Dataset name", type=str)
    parser.add_argument("--config", help="Datasete config name", type=str)
    parser.add_argument("--sep", help="Logit separator", type=str, default=',')
    parser.add_argument("--split", help="Dataset split", type=str, default='test')
    parser.add_argument("--regression", help="Is it a regression task?", type=int, default=0)

    args = parser.parse_args()
    main(**vars(args))
