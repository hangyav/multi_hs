import csv
import os

import datasets
from sklearn.model_selection import train_test_split


_CITATION = """\
"""

_DESCRIPTION = """\
"""

_DATA_URL = "https://raw.githubusercontent.com/avaapm/hatespeech/master/dataset_v2/hate_speech_dataset_v2.csv"

_HOME_PAGE = "https://github.com/avaapm/hatespeech"


class LargeScaleXDomain(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name=f'{lang}-{type}-{setup}', version=datasets.Version("1.0.0"))
        for type in ['binary', 'fine_grained']
        for lang in ['en', 'tr']
        for setup in ['full', 'politics', 'religion']
    ]

    def _info(self):
        lang, type, setup = self.config.name.split('-')
        if type == "binary":
            features = datasets.Features(
                {
                    'id': datasets.Value('string'),
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=['normal', 'hate']),
                }
            )
        elif type == "fine_grained":
            features = datasets.Features(
                {
                    'id': datasets.Value('string'),
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=['normal', 'offensive', 'hate']),
                }
            )
        else:
            raise NotImplementedError(f'Not supported config: {self.config.name}')

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOME_PAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(_DATA_URL)
        data = os.getenv('LARGE_SCALE_XDOMAIN_TWEETS')
        assert data is not None, 'Set LARGE_SCALE_XDOMAIN_TWEETS environment variable to point to downloaded tweets.'

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": dl_dir,
                    'datapath': data,
                    'split': 'train',
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": dl_dir,
                    'datapath': data,
                    'split': 'validation',
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": dl_dir,
                    'datapath': data,
                    'split': 'test',
                }
            ),
        ]

    def _generate_examples(self, filepath, datapath, split):
        lang, type, setup = self.config.name.split('-')
        tweets = dict()
        with open(datapath) as fin:
            csv_reader = csv.reader(
                fin,
                quotechar='"',
                delimiter=",",
                quoting=csv.QUOTE_ALL,
                skipinitialspace=True
            )
            for row in csv_reader:
                if len(row) < 2:
                    continue
                id, tweet = row
                tweets[id] = tweet

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
                id, langid, topic, hate = row

                if (lang == 'tr' and langid == '1') or (lang == 'en' and langid == '0'):
                    continue

                if setup == 'politics' and topic != '3':
                    continue

                if setup == 'religion' and topic != '0':
                    continue

                if type == 'binary':
                    label = 'normal' if hate == '0' else 'hate'
                elif type == 'fine-grained':
                    label = 'normal' if hate == '0' else 'offensive' if hate == '1' else 'hate'

                if id in tweets:
                    data.append({
                        'id': id,
                        'text': tweets[id],
                        'label': label,
                    })

        train, test = train_test_split(
            data,
            test_size=0.2,
            random_state=0,
            stratify=[item['label'] for item in data]
        )
        train, valid = train_test_split(
            train,
            test_size=0.2,
            random_state=0,
            stratify=[item['label'] for item in train]
        )

        if split == 'train':
            data = train
        elif split == 'validation':
            data = valid
        else:
            data = test

        for idx, item in enumerate(data):
            yield idx, item

