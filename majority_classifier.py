import sys
import os
import importlib
import json
from collections import Counter
import numpy as np
from datasets import load_dataset, load_metric

if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise ValueError('Parameters: <dataset_name> <dataset_config> <output> [train|validation|test]')

    dataset_name = sys.argv[1]
    dataset_config = sys.argv[2]
    output = sys.argv[3]
    split = 'test'
    if len(sys.argv) > 4:
        split = sys.args[4]

    dataset = load_dataset(
        importlib.import_module(
            f'src.data.{dataset_name}.{dataset_name}'
        ).__file__,
        dataset_config,
    )

    label = max(
        Counter(dataset['train']['label']).items(),
        key=lambda x: x[1],
    )[0]

    predictions = [label] * len(dataset[split])

    metric = load_metric(
        importlib.import_module(
            "src.metrics.f1_report.f1_report"
        ).__file__
    )

    res = metric.compute(
        predictions=predictions,
        references=dataset[split]['label'],
        label_names=dataset[split].features['label'].names,
    )
    res = {
        k: (v if type(v) != np.int64 else int(v))
        for k, v in res.items()
    }

    dir = os.path.dirname(output)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(output, 'w') as fout:
        json.dump(res, fout, indent=4)
