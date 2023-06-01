import os
import csv
import datasets
from sklearn.model_selection import train_test_split
from datasets.utils import logging


logger = logging.get_logger(__name__)


_CITATION = """
"""

_DESCRIPTION = """
"""


class HaSpeeDe3SharedTask(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="politics_text", version=VERSION),
        datasets.BuilderConfig(name="religious_text", version=VERSION),
    ]

    def _info(self):
        features = datasets.Features(
            {
                'id': datasets.Value('int64'),
                'text': datasets.Value('string'),
                'label': datasets.features.ClassLabel(
                    names=[
                        'noHate',
                        'Hate',
                    ]
                ),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage='https://github.com/mirkolai/EVALITA2023-HaSpeeDe3',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        url = os.getenv("HASPEEDE3_URL")
        assert url, 'Set HASPEEDE3_URL environment variable to point to downloaded tweets.'
        data_type = self.config.name.split('_')[1]
        if data_type == 'text':
            file = 'textual.csv'
            test_suffix = 'gold/test_textual_gold.csv'
        else:
            file = 'contextual.csv'
            test_suffix = 'test/test_contextual.csv'

        data_domain = self.config.name.split('_')[0]
        res = [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    'filepath': f'{url}/{test_suffix}',
                    'split': 'test'
                }
            )
        ]

        if data_domain == 'politics':
            res = [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        'filepath': f'{url}/development/training_{file}',
                        'split': 'train'
                    }
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        'filepath': f'{url}/development/training_{file}',
                        'split': 'validation'
                    }
                ),
            ] + res

        return res

    def _generate_examples(self, filepath, split):
        data_type = self.config.name.split('_')[1]
        if data_type != 'text':
            raise NotImplementedError()
        dataset_type = self.config.name.split('_')[0]
        data = list()
        with open(filepath) as fin:
            csv_reader = csv.reader(
                fin,
                quotechar='"',
                delimiter=",",
                quoting=csv.QUOTE_ALL,
                skipinitialspace=True
            )
            next(csv_reader)

            for row in csv_reader:
                id, text, label, dataset = row

                if dataset_type not in dataset:
                    continue

                if label == '0':
                    label = 'noHate'
                else:
                    label = 'Hate'

                data.append({
                    'id': id,
                    'text': text,
                    'label': label,
                })

        if split != 'test':
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
