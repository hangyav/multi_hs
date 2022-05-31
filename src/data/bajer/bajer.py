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

_DATA_URL = None


class Bajer(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name='abusive', version=VERSION),
        # only level 3 in annotation hierachy: https://aclanthology.org/2021.acl-long.247v2.pdf
        datasets.BuilderConfig(name='fine_grained', version=VERSION),
        # sexism vs all other (incdin non-abusive)
        datasets.BuilderConfig(name='sexism_binary', version=VERSION),
        datasets.BuilderConfig(name='racism_binary', version=VERSION),
    ]

    def _info(self):
        if self.config.name == 'abusive':
            features = datasets.Features(
                {
                    'id': datasets.Value('int64'),
                    'text': datasets.Value('string'),
                    'label': datasets.features.ClassLabel(
                        names=[
                            'ABUS',
                            'NOT',
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
                            'SEX',
                            'RAC',
                            'OTH',
                        ]
                    ),
                }
            )
        elif self.config.name == 'sexism_binary':
            features = datasets.Features(
                {
                    'id': datasets.Value('int64'),
                    'text': datasets.Value('string'),
                    'label': datasets.features.ClassLabel(
                        names=[
                            'SEX',
                            'NOT',
                        ]
                    ),
                }
            )
        elif self.config.name == 'racism_binary':
            features = datasets.Features(
                {
                    'id': datasets.Value('int64'),
                    'text': datasets.Value('string'),
                    'label': datasets.features.ClassLabel(
                        names=[
                            'RAC',
                            'NOT',
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
            homepage='https://github.com/phze22/Online-Misogyny-in-Danish-Bajer',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        url = os.getenv("BAJER_URL")
        assert url
        dl_dir = dl_manager.download_and_extract(url)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    'filepath': dl_dir,
                    'split': 'train'
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    'filepath': dl_dir,
                    'split': 'validation'
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    'filepath': dl_dir,
                    'split': 'test'
                }
            )
        ]

    def _generate_examples(self, filepath, split):
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
                id, _, _, text, _, _, abusive, _, type, _, _, _, _, _ = row

                if self.config.name == 'abusive':
                    label = abusive
                elif self.config.name == 'fine_grained':
                    if type == '':
                        continue
                    label = type
                elif self.config.name == 'sexism_binary':
                    label = 'SEX' if type == 'SEX' else 'NOT'
                elif self.config.name == 'racism_binary':
                    label = 'RAC' if type == 'RAC' else 'NOT'
                else:
                    raise NotADirectoryError('Should not get here!')

                data.append({
                    'id': id,
                    'text': text,
                    'label': label,
                })

        train, test = train_test_split(
            data,
            test_size=0.2,
            random_state=0,
            stratify=[item['label'] for item in data]
        )

        if split == 'test':
            data = test
        else:
            data = train

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
