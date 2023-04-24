import csv
import os

import datasets
from sklearn.model_selection import train_test_split


_CITATION = """\
"""

_DESCRIPTION = """\
"""

_DATA_URL = {
    'en': "https://raw.githubusercontent.com/dhfbk/religious-hate-speech/main/data/dataset_en-portion.tsv",
    'it': "https://raw.githubusercontent.com/dhfbk/religious-hate-speech/main/data/dataset_it-portion.tsv",
}

_HOME_PAGE = "https://github.com/dhfbk/religious-hate-speech"


class LargeScaleXDomain(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name=f'{lang}-{type}', version=datasets.Version("1.0.0"))
        for type in ['abusive']
        for lang in ['en', 'it']
    ]

    def _info(self):
        lang, type = self.config.name.split('-')
        if type == "abusive":
            features = datasets.Features(
                {
                    'id': datasets.Value('string'),
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=['NOT-ABUSIVE', 'ABUSIVE']),
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
        data_dir = os.getenv('RELIGIOUS_HATE_URL')
        assert data_dir is not None, 'Set RELIGIOUS_HATE_URL environment variable to point to downloaded tweets.'

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": dl_dir,
                    'datapath': data_dir,
                    'split': 'train',
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": dl_dir,
                    'datapath': data_dir,
                    'split': 'validation',
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": dl_dir,
                    'datapath': data_dir,
                    'split': 'test',
                }
            ),
        ]

    def _generate_examples(self, filepath, datapath, split):
        lang, type = self.config.name.split('-')
        tweets = dict()
        datapath = os.path.join(datapath, f'dataset_{lang}-portion_tweets.csv')
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
        with open(filepath[lang]) as fin:
            csv_reader = csv.reader(
                fin,
                quotechar='"',
                delimiter="\t",
                quoting=csv.QUOTE_ALL,
                skipinitialspace=True
            )

            next(csv_reader)

            for row in csv_reader:
                id = row[0]
                label = row[1]
                label = label.replace(' ', '-')
                if label == 'ABUS':
                    label = 'ABUSIVE'

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

