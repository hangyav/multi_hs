import os
import csv
import datasets


_CITATION = """
"""

_DESCRIPTION = """
"""

_DATA_URL = {
        'en': 'https://hasocfire.github.io/hasoc/2019/files/english_dataset.zip',
        'de': 'https://hasocfire.github.io/hasoc/2019/files/german_dataset.zip',
        # Hindi is code-switched!!!
        'hi': 'https://hasocfire.github.io/hasoc/2019/files/hindi_dataset.zip',
}

_DATA_PATHS = {
    'en': {
        'train': 'english_dataset/english_dataset.tsv',
        'test': 'english_dataset/hasoc2019_en_test-2919.tsv',
    },
    'de': {
        'train': 'german_dataset/german_dataset.tsv',
        'test': 'german_dataset/hasoc_de_test_gold.tsv',
    },
    'hi': {
        'train': 'hindi_dataset/hindi_dataset.tsv',
        'test': 'hindi_dataset/hasoc2019_hi_test_gold_2919.tsv',
    },
}


class HASOC19(datasets.GeneratorBasedBuilder):

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    'id': datasets.Value('string'),
                    'text': datasets.Value('string'),
                    'label_binary': datasets.features.ClassLabel(
                        names=[
                            'NOT',
                            'HOF',
                        ]
                    ),
                    'label_fine_grained': datasets.features.ClassLabel(
                        names=[
                            'HATE',
                            'OFFN',
                            'PRFN',
                            'NONE',
                        ]
                    ),
                    'targeted': datasets.features.ClassLabel(
                        names=[
                            'UNT',
                            'TIN',
                            'NONE',
                        ]
                    ),
                }
            ),
            supervised_keys=None,
            homepage='https://hasocfire.github.io/hasoc/2019/call_for_participation.html',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(_DATA_URL)
        language = self.config.name

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    'filepath': os.path.join(
                        dl_dir[language],
                        _DATA_PATHS[language]['train'],
                    ),
                    'is_targeted': language != 'de',
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    'filepath': os.path.join(
                        dl_dir[language],
                        _DATA_PATHS[language]['test'],
                    ),
                    'is_targeted': language != 'de',
                },
            ),
        ]

    def _generate_examples(self, filepath, is_targeted):
        with open(filepath, encoding='utf-8') as fin:
            csv_reader = csv.reader(
                fin,
                quotechar='"',
                delimiter='\t',
                quoting=csv.QUOTE_NONE,
                skipinitialspace=True,
            )

            # Skip header
            next(csv_reader)

            for idx, row in enumerate(csv_reader):
                if is_targeted:
                    id, text, label_binary, label_fine_grained, targeted = row
                else:
                    id, text, label_binary, label_fine_grained = row
                    targeted = 'NONE'

                yield idx, {
                    'id': id,
                    'text': text,
                    'label_binary': label_binary,
                    'label_fine_grained': label_fine_grained,
                    'targeted': targeted,
                }
