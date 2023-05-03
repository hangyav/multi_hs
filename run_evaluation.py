import argparse
import numpy as np
import json

from datasets import load_dataset
from run_classifier import METRICS, compute_metrics


def print_default_format(preds, df, output_file):
    labels = df.features['label'].names
    with open(output_file, "w") as f:
        print('index\tprediction', file=f)
        print(
            '\n'.join([
                f'{i}\t{labels[p]}' for i, p in enumerate(preds)
            ]),
            file=f
        )


def print_haspeede3_format(preds, df, output_file):
    labels = df.features['label'].names
    labels = {i: i for i, _ in enumerate(labels)}
    with open(output_file, "w") as f:
        print('anonymized_tweet_id,label', file=f)
        print(
            '\n'.join([
                f'{df[i]["id"]},{labels[p]}' for i, p in enumerate(preds)
            ]),
            file=f
        )


LABELS_FORMATS = {
    'default': print_default_format,
    'haspeede3': print_haspeede3_format,
        }


def main(prediction_file, output_file, dataset, config, sep=',', split='test',
         regression=False, print_labels=False, format='default'):
    preds = list()
    with open(prediction_file, "r") as f:
        for line in f:
            line = line.strip().split(sep)
            line = [float(x) for x in line]
            preds.append(line)
    preds = np.array(preds)

    df = load_dataset(dataset, config)
    refs = np.array([item['label'] for item in df[split]])

    if not print_labels:
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
    else:
        preds = np.squeeze(preds) if regression else np.argmax(preds, axis=1)

        LABELS_FORMATS[format](preds, df[split], output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", help="File containing the predictions", type=str)
    parser.add_argument("--output_file", help="Output file to save the results", type=str)
    parser.add_argument("--dataset", help="Dataset name", type=str)
    parser.add_argument("--config", help="Datasete config name", type=str)
    parser.add_argument("--sep", help="Logit separator", type=str, default=',')
    parser.add_argument("--split", help="Dataset split", type=str, default='test')
    parser.add_argument("--regression", help="Is it a regression task?", type=int, default=0)
    parser.add_argument("--print_labels", help="Print the predicted labels instead of the metrics", type=int, default=0)
    parser.add_argument("--format", help="Format for the labels output", type=str, default='default', choices=LABELS_FORMATS.keys())

    args = parser.parse_args()
    main(**vars(args))
