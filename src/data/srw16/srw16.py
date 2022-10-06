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

_DATA_URL = 'https://raw.githubusercontent.com/zeeraktalat/hatespeech/master/NAACL_SRW_2016.csv'


class SRW16(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="binary", version=VERSION),
        datasets.BuilderConfig(name="fine_grained", version=VERSION),
        datasets.BuilderConfig(name="fine_grained_ab1", version=VERSION),  # ablation study
        datasets.BuilderConfig(name="sexism", version=VERSION),
        datasets.BuilderConfig(name="racism", version=VERSION),
    ]

    def _info(self):
        if self.config.name == 'binary':
            features = datasets.Features(
                {
                    'id': datasets.Value('int64'),
                    'text': datasets.Value('string'),
                    'label': datasets.features.ClassLabel(
                        names=[
                            'offensive',
                            'non-offensive',
                        ]
                    ),
                }
            )
        elif self.config.name == 'fine_grained':
            features = datasets.Features(
                {
                    'id': datasets.Value('int64'),
                    'text': datasets.Value('string'),
                    'label': datasets.features.ClassLabel(
                        names=[
                            'sexism',
                            'racism',
                            'none',
                        ]
                    ),
                }
            )
        elif self.config.name == 'fine_grained_ab1':
            features = datasets.Features(
                {
                    'id': datasets.Value('int64'),
                    'text': datasets.Value('string'),
                    'label': datasets.features.ClassLabel(
                        names=[
                            'sexism',
                            # 'racism',
                            'none',
                        ]
                    ),
                }
            )
        elif self.config.name == 'sexism':
            features = datasets.Features(
                {
                    'id': datasets.Value('int64'),
                    'text': datasets.Value('string'),
                    'label': datasets.features.ClassLabel(
                        names=[
                            'sexism',
                            'non-sexism',
                        ]
                    ),
                }
            )
        elif self.config.name == 'racism':
            features = datasets.Features(
                {
                    'id': datasets.Value('int64'),
                    'text': datasets.Value('string'),
                    'label': datasets.features.ClassLabel(
                        names=[
                            'racism',
                            'non-racism',
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
            homepage='https://github.com/zeeraktalat/hatespeech',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(_DATA_URL)
        data = os.getenv('SRW16_TWEETS')
        assert data is not None, 'Set SRW16_TWEETS environment variable to point to downloaded tweets.'

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    'filepath': dl_dir,
                    'datapath': data,
                    'split': 'train'
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    'filepath': dl_dir,
                    'datapath': data,
                    'split': 'validation'
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    'filepath': dl_dir,
                    'datapath': data,
                    'split': 'test'
                }
            )
        ]

    def _generate_examples(self, filepath, datapath, split):
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
        seen = dict()
        with open(filepath) as fin:
            csv_reader = csv.reader(
                fin,
                quotechar='"',
                delimiter=",",
                quoting=csv.QUOTE_ALL,
                skipinitialspace=True
            )

            for row in csv_reader:
                id, label = row

                if self.config.name == 'binary':
                    label = 'non-offensive' if label == 'none' else 'offensive'
                elif self.config.name == 'sexism':
                    label = 'sexism' if label == 'sexism' else 'non-sexism'
                elif self.config.name == 'racism':
                    label = 'racism' if label == 'racism' else 'non-racism'
                elif self.config.name == 'fine_grained_ab1':
                    if label == 'racism':
                        continue

                if id in tweets:
                    if id not in seen:
                        # there are some conflicting annotations
                        seen[id] = len(data)
                        data.append({
                            'id': id,
                            'text': tweets[id],
                            'label': label,
                        })
                    else:
                        if seen[id] is not None:
                            data.pop(seen[id])
                            seen[id] = None
                else:
                    logger.info(f'Missing tweets for id: {id}')

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
