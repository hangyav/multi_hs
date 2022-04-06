import csv
import datasets


_CITATION = """
"""

_DESCRIPTION = """
"""

_DATA_URL = {
    'train': 'https://raw.githubusercontent.com/uds-lsv/GermEval-2018-Data/master/germeval2018.training.txt',
    'test': 'https://raw.githubusercontent.com/uds-lsv/GermEval-2018-Data/master/germeval2018.test.txt',
}


class GermEval18(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="binary", version=VERSION),
        datasets.BuilderConfig(name="fine_grained", version=VERSION),
    ]

    def _info(self):
        if self.config.name == 'binary':
            features = datasets.Features(
                {
                    'text': datasets.Value('string'),
                    'label': datasets.features.ClassLabel(
                        names=[
                            'OTHER',
                            'OFFENSE',
                        ]
                    ),
                }
            )
        elif self.config.name == 'fine_grained':
            features = datasets.Features(
                {
                    'text': datasets.Value('string'),
                    'label': datasets.features.ClassLabel(
                        names=[
                            'OTHER',
                            'PROFANITY',
                            'INSULT',
                            'ABUSE',
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
            homepage='https://github.com/uds-lsv/GermEval-2018-Data',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(_DATA_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={'filepath': dl_dir['train']}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={'filepath': dl_dir['test']}
            ),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding='utf-8') as fin:
            csv_reader = csv.reader(
                fin,
                quotechar='"',
                delimiter='\t',
                quoting=csv.QUOTE_NONE,
                skipinitialspace=True,
            )

            idx = 0
            for row in csv_reader:
                text, label_binary, label_fine_grained = row

                yield idx, {
                    'text': text,
                    'label': label_binary if self.config.name == 'binary' else label_fine_grained,
                }
                idx += 1
