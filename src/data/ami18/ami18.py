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


class AMI18(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name=f'{lang}-{type}', version=datasets.Version("1.0.0"))
        for type in ['misogyny', 'misogyny_category', 'target'] + ['misogyny_category_ab1']  # ab -> ablation
        for lang in ['en', 'it']
    ]

    def _info(self):
        if self.config.name in ['en-misogyny', 'it-misogyny']:
            features = datasets.Features(
                {
                    'id': datasets.Value('int64'),
                    'text': datasets.Value('string'),
                    'label': datasets.features.ClassLabel(
                        names=[
                            'misogyny',
                            'non-misogyny',
                        ]
                    ),
                }
            )
        elif self.config.name in ['en-misogyny_category', 'it-misogyny_category']:
            features = datasets.Features(
                {
                    'id': datasets.Value('int64'),
                    'text': datasets.Value('string'),
                    'label': datasets.features.ClassLabel(
                        names=[
                            'stereotype',
                            'dominance',
                            'derailing',
                            'sexual_harassment',
                            'discredit',
                        ]
                    ),
                }
            )
        elif self.config.name in ['en-misogyny_category_ab1', 'it-misogyny_category_ab1']:
            features = datasets.Features(
                {
                    'id': datasets.Value('int64'),
                    'text': datasets.Value('string'),
                    'label': datasets.features.ClassLabel(
                        names=[
                            'stereotype',
                            'dominance',
                            # 'derailing',
                            'sexual_harassment',
                            'discredit',
                        ]
                    ),
                }
            )
        elif self.config.name in ['en-target', 'it-target']:
            features = datasets.Features(
                {
                    'id': datasets.Value('int64'),
                    'text': datasets.Value('string'),
                    'label': datasets.features.ClassLabel(
                        names=[
                            'active',
                            'passive',
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
            homepage='https://amievalita2018.wordpress.com',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        url = os.getenv("AMI18_URL")
        assert url
        dl_dir = dl_manager.download_and_extract(url)
        lang = self.config.name.split('-')[0]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    'filepath': os.path.join(
                        dl_dir,
                        f'{lang}_training.tsv'
                    ),
                    'split': 'train'
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    'filepath': os.path.join(
                        dl_dir,
                        f'{lang}_training.tsv'
                    ),
                    'split': 'validation'
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    'filepath': os.path.join(
                        dl_dir,
                        f'{lang}_testing.tsv'
                    ),
                    'split': 'test'
                }
            )
        ]

    def _generate_examples(self, filepath, split):
        data = list()
        data_type = self.config.name.split('-')[1]
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
                id, text, misogyny, category, target = row

                if data_type == 'misogyny':
                    label = 'misogyny' if misogyny == '1' else 'non-misogyny'
                elif data_type == 'misogyny_category':
                    label = category
                elif data_type == 'misogyny_category_ab1':
                    label = category
                    if label == 'derailing':
                        continue
                elif data_type == 'target':
                    label = target
                    if label == 'pass':
                        # there seems to be at least one annotation error
                        label = 'passive'
                else:
                    raise ValueError('Should not get here!')

                if label == '0':
                    continue

                data.append({
                    'id': id,
                    'text': text,
                    'label': label,
                })

        if split in ['train', 'validation']:
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
