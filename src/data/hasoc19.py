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
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name=f'{lang}-{type}', version=datasets.Version("1.0.0"))
        for lang in ['en', 'de', 'hi']
        for type in ['binary', 'fine_grained', 'targeted']
        if f'{lang}-{type}' not in ['de-targeted']
    ]

    def _info(self):
        lang, type = self.config.name.split('-')
        if type == 'binary':
            features = datasets.Features(
                {
                    'id': datasets.Value('string'),
                    'text': datasets.Value('string'),
                    'label': datasets.features.ClassLabel(
                        names=[
                            'NOT',
                            'HOF',
                        ]
                    ),
                }
            )
        elif type == 'fine_grained':
            features = datasets.Features(
                {
                    'id': datasets.Value('string'),
                    'text': datasets.Value('string'),
                    'label': datasets.features.ClassLabel(
                        names=[
                            'HATE',
                            'OFFN',
                            'PRFN',
                        ]
                    ),
                }
            )
        elif type == 'targeted':
            features = datasets.Features(
                {
                    'id': datasets.Value('string'),
                    'text': datasets.Value('string'),
                    'label': datasets.features.ClassLabel(
                        names=[
                            'UNT',
                            'TIN',
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
            homepage='https://hasocfire.github.io/hasoc/2019/call_for_participation.html',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        language, _ = self.config.name.split('-')
        dl_dir = dl_manager.download_and_extract(_DATA_URL[language])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    'filepath': os.path.join(
                        dl_dir,
                        _DATA_PATHS[language]['train'],
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    'filepath': os.path.join(
                        dl_dir,
                        _DATA_PATHS[language]['test'],
                    ),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        language, type = self.config.name.split('-')
        is_targeted = language != 'de'

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

            idx = 0
            for row in csv_reader:
                if is_targeted:
                    id, text, label_binary, label_fine_grained, targeted = row
                else:
                    id, text, label_binary, label_fine_grained = row
                    targeted = 'NONE'

                if type == 'binary':
                    label = label_binary
                elif type == 'fine_grained':
                    label = label_fine_grained
                else:
                    label = targeted

                if label == 'NONE':
                    continue

                yield idx, {
                    'id': id,
                    'text': text,
                    'label': label,
                }
                idx += 1
