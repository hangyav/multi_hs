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

_DATA_URL = 'https://zenodo.org/record/5291339/files/RP-Mod-Crowd.csv?download=1'


class RP21(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        # binary
        datasets.BuilderConfig(name="mod", version=VERSION),
        datasets.BuilderConfig(name="crowd_binary", version=VERSION),
        # fine_grained
        datasets.BuilderConfig(name="crowd_fine_grained", version=VERSION),
    ]

    def _info(self):
        if self.config.name == 'mod':
            features = datasets.Features(
                {
                    'id': datasets.Value('int64'),
                    'text': datasets.Value('string'),
                    'label': datasets.features.ClassLabel(
                        names=[
                            'abusive',
                            'non-abusive',
                        ]
                    ),
                }
            )
        elif self.config.name == 'crowd_binary':
            features = datasets.Features(
                {
                    'id': datasets.Value('int64'),
                    'text': datasets.Value('string'),
                    'label': datasets.features.ClassLabel(
                        names=[
                            'abusive',
                            'non-abusive',
                        ]
                    ),
                }
            )
        elif self.config.name == 'crowd_fine_grained':
            features = datasets.Features(
                {
                    'id': datasets.Value('int64'),
                    'text': datasets.Value('string'),
                    'label': datasets.features.ClassLabel(
                        names=[
                            'sexism',
                            'racism',
                            'threat',
                            'insult',
                            'profanity',
                            'meta',
                            'advertisment',
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
            homepage='https://zenodo.org/record/5291339#.YpDM03UzaEJ',
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

            next(csv_reader)

            for row in csv_reader:
                _, _, id, text, mod, crowd, _, sexism, racism, threat, insult, profanity, meta, advert = row

                if self.config.name == 'mod':
                    label = 'abusive' if mod == '1' else 'non-abusive'
                elif self.config.name == 'crowd_binary':
                    label = 'abusive' if mod == '1' else 'non-abusive'
                elif self.config.name == 'crowd_fine_grained':
                    if crowd == '0':
                        label = 'none'
                    else:
                        for name, val in [('sexism', sexism), ('racism',
                            racism), ('threat', threat), ('insult', insult),
                            ('profanity', profanity), ('meta', meta),
                            ('advertisment', advert)]:
                            # there were 5 annotators
                            if float(val) >= 3.0:
                                label = name
                                break
                else:
                    raise NotImplementedError('We should not get here!')

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
