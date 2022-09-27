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

_DATA_URL = 'https://raw.githubusercontent.com/JAugusto97/ToLD-Br/main/ToLD-BR.csv'


class ToLD_Br(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="fine_grained", version=VERSION),
    ]

    def _info(self):
        if self.config.name == 'fine_grained':
            features = datasets.Features(
                {
                    'id': datasets.Value('int64'),
                    'text': datasets.Value('string'),
                    'label': datasets.features.ClassLabel(
                        names=[
                            'homophobia',
                            'obscene',
                            'insult',
                            'racism',
                            'misogyny',
                            'xenophobia',
                            'none',
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
            homepage='https://github.com/JAugusto97/ToLD-Br',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(_DATA_URL)

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
                delimiter=",",
                quoting=csv.QUOTE_ALL,
                skipinitialspace=True
            )

            # Skip header
            next(csv_reader)

            for idx, row in enumerate(csv_reader):
                text, homophobia, obscene, insult, racism, misogyny, xenophobia = row
                homophobia = int(float(homophobia))
                obscene = int(float(obscene))
                insult = int(float(insult))
                racism = int(float(racism))
                misogyny = int(float(misogyny))
                xenophobia = int(float(xenophobia))

                if sum([homophobia, obscene, insult, racism, misogyny, xenophobia]) > 3:
                    logger.info('Number of annotations is more than 3. Skippking...')
                    continue

                if homophobia > 1:
                    label = 'homophobia'
                elif obscene > 1:
                    label = 'obscene'
                elif insult > 1:
                    label = 'insult'
                elif racism > 1:
                    label = 'racism'
                elif misogyny > 1:
                    label = 'misogyny'
                elif xenophobia > 1:
                    label = 'xenophobia'
                else:
                    label = 'none'

                item = {
                    'id': idx,
                    'text': text,
                    'label': label,
                }
                data.append(item)

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
