import csv
import datasets
from sklearn.model_selection import train_test_split


_CITATION = """
"""

_DESCRIPTION = """
"""

_DATA_URL = {
    'offensive': {
        'train': 'https://raw.githubusercontent.com/idontflow/OLID/master/olid-training-v1.0.tsv',
        'test': {
            'texts': 'https://raw.githubusercontent.com/idontflow/OLID/master/testset-levela.tsv',
            'labels': 'https://raw.githubusercontent.com/idontflow/OLID/master/labels-levela.csv',
        },
    },
    'targeted': {
        'train': 'https://raw.githubusercontent.com/idontflow/OLID/master/olid-training-v1.0.tsv',
        'test': {
            'texts': 'https://raw.githubusercontent.com/idontflow/OLID/master/testset-levelb.tsv',
            'labels': 'https://raw.githubusercontent.com/idontflow/OLID/master/labels-levelb.csv',
        },
    },
    'target': {
        'train': 'https://raw.githubusercontent.com/idontflow/OLID/master/olid-training-v1.0.tsv',
        'test': {
            'texts': 'https://raw.githubusercontent.com/idontflow/OLID/master/testset-levelc.tsv',
            'labels': 'https://raw.githubusercontent.com/idontflow/OLID/master/labels-levelc.csv',
        },
    },
}


class OLID(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="offensive", version=VERSION),
        datasets.BuilderConfig(name="targeted", version=VERSION),
        datasets.BuilderConfig(name="target", version=VERSION),
    ]

    def _info(self):
        if self.config.name == 'offensive':
            features = datasets.Features(
                {
                    'id': datasets.Value('int64'),
                    'text': datasets.Value('string'),
                    'label': datasets.features.ClassLabel(
                        names=[
                            'NOT',
                            'OFF',
                        ]
                    ),
                }
            )
        elif self.config.name == 'targeted':
            features = datasets.Features(
                {
                    'id': datasets.Value('int64'),
                    'text': datasets.Value('string'),
                    'label': datasets.features.ClassLabel(
                        names=[
                            'UNT',
                            'TIN',
                        ]
                    ),
                }
            )
        elif self.config.name == 'target':
            features = datasets.Features(
                {
                    'id': datasets.Value('int64'),
                    'text': datasets.Value('string'),
                    'label': datasets.features.ClassLabel(
                        names=[
                            'IND',
                            'GRP',
                            'OTH',
                        ]
                    ),
                }
            )
        else:
            raise NotImplementedError(f'Not supported config: {self.config.name}')

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage='https://github.com/idontflow/OLID',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(_DATA_URL[self.config.name])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={'filepath': dl_dir['train'], 'split': 'train'}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={'filepath': dl_dir['train'], 'split': 'validation'}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={'filepath': dl_dir['test'], 'split': 'test'}),
        ]

    def _generate_examples(self, filepath, split):
        if split != 'test':
            return self._generate_train_valid_examples(filepath, split)
        else:
            return self._generate_test_examples(filepath)

    def _generate_train_valid_examples(self, filepath, split):
        data = list()
        with open(filepath, encoding='utf-8') as fin:
            csv_reader = csv.reader(
                fin,
                quotechar='"',
                delimiter='\t',
                quoting=csv.QUOTE_NONE,
                skipinitialspace=True,
            )

            # skip header
            next(csv_reader)

            for row in csv_reader:
                id, text, offensive, targeted, target = row

                if self.config.name == 'offensive':
                    label = offensive
                elif self.config.name == 'targeted':
                    if targeted == 'NULL':
                        continue
                    label = targeted
                else:
                    if target == 'NULL':
                        continue
                    label = target

                data.append({
                    'id': id,
                    'text': text,
                    'label': label,
                })

        train, valid = train_test_split(
            data,
            test_size=0.2,
            random_state=0,
            stratify=[item['label'] for item in data]
        )

        if split == 'train':
            data = train
        else:
            data = valid

        for idx, item in enumerate(data):
            yield idx, item

    def _generate_test_examples(self, filepath):
        with open(filepath['labels']) as fin:
            labels = dict()
            for line in fin:
                line = line.strip().split(',')
                labels[line[0]] = line[1]

        with open(filepath['texts'], encoding='utf-8') as fin:
            csv_reader = csv.reader(
                fin,
                quotechar='"',
                delimiter='\t',
                quoting=csv.QUOTE_NONE,
                skipinitialspace=True,
            )

            # skip header
            next(csv_reader)

            idx = 0
            for row in csv_reader:
                id, text = row

                if id not in labels:
                    continue

                yield idx, {
                    'id': id,
                    'text': text,
                    'label': labels[id],
                }
                idx += 1
