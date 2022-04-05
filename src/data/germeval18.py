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

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    'text': datasets.Value('string'),
                    'label_binary': datasets.features.ClassLabel(
                        names=[
                            'OTHER',
                            'OFFENSE',
                        ]
                    ),
                    'label_fine_grained': datasets.features.ClassLabel(
                        names=[
                            'OTHER',
                            'PROFANITY',
                            'INSULT',
                            'ABUSE',
                        ]
                    ),
                }
            ),
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

            for idx, row in enumerate(csv_reader):
                text, label_binary, label_fine_grained = row

                yield idx, {
                    'text': text,
                    'label_binary': label_binary,
                    'label_fine_grained': label_fine_grained,
                }
