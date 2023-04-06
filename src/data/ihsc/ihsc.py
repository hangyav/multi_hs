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

_DATA_URL = 'https://raw.githubusercontent.com/msang/hate-speech-corpus/master/IHSC_ids.tsv'

_HOME_PAGE = 'https://github.com/msang/hate-speech-corpus'


class IHSC(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="hate", version=VERSION),
        datasets.BuilderConfig(name="stereotype", version=VERSION),
    ]

    def _info(self):
        if self.config.name == 'hate':
            features = datasets.Features(
                {
                    'id': datasets.Value('int64'),
                    'text': datasets.Value('string'),
                    'label': datasets.features.ClassLabel(
                        names=[
                            'no',
                            'yes',
                        ]
                    ),
                }
            )
        elif self.config.name == 'stereotype':
            features = datasets.Features(
                {
                    'id': datasets.Value('int64'),
                    'text': datasets.Value('string'),
                    'label': datasets.features.ClassLabel(
                        names=[
                            'no',
                            'yes',
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
            homepage=_HOME_PAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(_DATA_URL)
        data = os.getenv('IHSC_TWEETS')
        assert data is not None, 'Set IHSC_TWEETS environment variable to point to downloaded tweets.'

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
        with open(filepath) as fin:
            csv_reader = csv.reader(
                fin,
                quotechar='"',
                delimiter="\t",
                quoting=csv.QUOTE_ALL,
                skipinitialspace=True
            )

            next(csv_reader)

            for row in csv_reader:
                id, hs, agressiveness, offensiveness, irony, stereotype = row

                if self.config.name == 'hate':
                    label = hs
                elif self.config.name == 'stereotype':
                    label = stereotype

                if id in tweets:
                    data.append({
                        'id': id,
                        'text': tweets[id],
                        'label': label,
                    })
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
