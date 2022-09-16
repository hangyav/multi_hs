# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Hate speech dataset"""


import csv
import os

import datasets
from sklearn.model_selection import train_test_split


_CITATION = """\
"""

_DESCRIPTION = """\
"""

_DATA_URL = {
    'en': {
        'train': 'https://raw.githubusercontent.com/cicl2018/HateEvalTeam/master/Data%20Files/Data%20Files/%232%20Development-English-A/train_en.tsv',
        'dev': 'https://raw.githubusercontent.com/cicl2018/HateEvalTeam/master/Data%20Files/Data%20Files/%232%20Development-English-A/dev_en.tsv',
    },
    'es': {
        'train': 'https://raw.githubusercontent.com/cicl2018/HateEvalTeam/master/Data%20Files/Data%20Files/%232%20Development-Spanish-A/train_es.tsv',
        'dev': 'https://raw.githubusercontent.com/cicl2018/HateEvalTeam/master/Data%20Files/Data%20Files/%232%20Development-Spanish-A/dev_es.tsv',
    },
}


class Hateval19(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name=f'{lang}-{type}', version=datasets.Version("1.0.0"))
        for lang in ['en', 'es']
        for type in ['hate', 'aggressive', 'target']
    ]

    def _info(self):
        lang, type = self.config.name.split('-')
        if type == "hate":
            features = datasets.Features(
                {
                    'id': datasets.Value('string'),
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=['hate', 'nohate']),
                }
            )
        elif type == "aggressive":
            features = datasets.Features(
                {
                    'id': datasets.Value('string'),
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=['aggressive', 'nonaggressive']),
                }
            )
        elif type == "target":
            features = datasets.Features(
                {
                    'id': datasets.Value('string'),
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=['individual', 'generic']),
                }
            )
        else:
            raise NotImplementedError(f'Not supported config: {self.config.name}')

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage="https://github.com/cicl2018/HateEvalTeam",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        language, _ = self.config.name.split('-')
        dl_dir = dl_manager.download_and_extract(_DATA_URL[language])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": dl_dir['train'],
                    'split': 'train',
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": dl_dir['train'],
                    'split': 'validation',
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": dl_dir['dev'],
                    'split': 'test',
                }
            ),
        ]

    def _generate_examples(self, filepath, split):
        data = list()
        language, type = self.config.name.split('-')
        with open(filepath, encoding="utf-8") as csv_file:

            csv_reader = csv.reader(
                csv_file,
                quotechar='"',
                delimiter="\t",
                quoting=csv.QUOTE_ALL,
                skipinitialspace=True
            )

            next(csv_reader)

            idx = 0
            for row in csv_reader:
                id, text, hs, tr, ag = row

                if type != 'hate' and hs != '1':
                    continue

                if type == 'hate':
                    label = 'hate' if hs == '1' else 'nohate'
                elif type == 'aggressive':
                    label = 'aggressive' if ag == '1' else 'nonaggressive'
                else:
                    label = 'individual' if tr == '1' else 'generic'

                item = {
                    'id': id,
                    'text': text,
                    'label': label,
                }
                if split == 'test':
                    yield idx, item
                else:
                    data.append(item)
                idx += 1

        if split != 'test':
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
