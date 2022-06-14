import sys
import importlib
from datasets import load_dataset

if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise ValueError('Arguments: <dataset> <config> <path_to_predictions>')

    with open(sys.argv[3]) as fin:
        fin.readline()
        predictions = dict(
            map(
                lambda x: (int(x[0]), x[1]),
                [line.strip().split() for line in fin]
            )
        )

    dataset = load_dataset(
        importlib.import_module(
            f'src.data.{sys.argv[1]}.{sys.argv[1]}'
        ).__file__,
        sys.argv[2],
        split='test',
    )

    print('IDX\tID\tGOLD\tPREDICTION\tTEXT')
    for idx, item in enumerate(dataset):
        label = predictions[idx]
        gold = dataset.features['label'].names[item['label']]

        color_s = ''
        color_e = ''
        if label != gold:
            color_s = '\033[31m'
            color_e = '\033[0m'

        print(color_s+'\t'.join([str(idx), str(item['id']), gold, label, item['text']])+color_e)
