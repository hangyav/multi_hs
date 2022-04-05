import csv
import datasets


_CITATION = """
"""

_DESCRIPTION = """
"""

_DATA_URL = {
    'train': 'https://raw.githubusercontent.com/idontflow/OLID/master/olid-training-v1.0.tsv',
    'test': {
        'offensive': {
            'texts': 'https://raw.githubusercontent.com/idontflow/OLID/master/testset-levela.tsv',
            'labels': 'https://raw.githubusercontent.com/idontflow/OLID/master/labels-levela.csv',
        },
        'targeted': {
            'texts': 'https://raw.githubusercontent.com/idontflow/OLID/master/testset-levelb.tsv',
            'labels': 'https://raw.githubusercontent.com/idontflow/OLID/master/labels-levelb.csv',
        },
        'target': {
            'texts': 'https://raw.githubusercontent.com/idontflow/OLID/master/testset-levelc.tsv',
            'labels': 'https://raw.githubusercontent.com/idontflow/OLID/master/labels-levelc.csv',
        },
    },
}


class OLID(datasets.GeneratorBasedBuilder):

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    'id': datasets.Value('int64'),
                    'text': datasets.Value('string'),
                    'offensive': datasets.features.ClassLabel(
                        names=[
                            'NOT',
                            'OFF',
                        ]
                    ),
                    'targeted': datasets.features.ClassLabel(
                        names=[
                            'NULL',
                            'UNT',
                            'TIN',
                        ]
                    ),
                    'target': datasets.features.ClassLabel(
                        names=[
                            'NULL',
                            'IND',
                            'GRP',
                            'OTH',
                        ]
                    ),
                }
            ),
            supervised_keys=None,
            homepage='https://github.com/idontflow/OLID',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(_DATA_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={'filepath': dl_dir['train'], 'train': True}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={'filepath': dl_dir['test'], 'train': False}),
        ]

    def _generate_examples(self, filepath, train):
        if train:
            return self._generate_train_examples(filepath)
        else:
            return self._generate_test_examples(filepath)

    def _generate_train_examples(self, filepath):
        with open(filepath, encoding='utf-8') as fin:
            csv_reader = csv.reader(
                fin,
                quotechar='"',
                delimiter='\t',
                quoting=csv.QUOTE_NONE,
                skipinitialspace=True,
            )

            # skip header
            next(csv_reader)

            for idx, row in enumerate(csv_reader):
                id, text, offensive, targeted, target = row

                yield idx, {
                    'id': id,
                    'text': text,
                    'offensive': offensive,
                    'targeted': targeted,
                    'target': target,
                }

    def _generate_test_examples(self, filepath):
        labels = dict()
        for key, value in filepath.items():
            with open(value['labels']) as fin:
                tmp_dict = dict()

                for line in fin:
                    line = line.strip().split(',')
                    tmp_dict[line[0]] = line[1]

                labels[key] = tmp_dict

        texts = dict()
        for key, value in filepath.items():
            with open(value['texts'], encoding='utf-8') as fin:
                csv_reader = csv.reader(
                    fin,
                    quotechar='"',
                    delimiter='\t',
                    quoting=csv.QUOTE_NONE,
                    skipinitialspace=True,
                )

                # skip header
                next(csv_reader)

                for idx, row in enumerate(csv_reader):
                    id, text = row

                    if id in texts:
                        assert texts[id] == text
                    else:
                        texts[id] = text

        for idx, id in enumerate(sorted(texts.keys())):
            text = texts[id]
            offensive = labels['offensive'][id]
            targeted = labels['targeted'][id] if id in labels['targeted'] else 'NULL'
            target = labels['target'][id] if id in labels['target'] else 'NULL'

            yield idx, {
                'id': id,
                'text': text,
                'offensive': offensive,
                'targeted': targeted,
                'target': target,
            }
