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

_DATA_URL = None

_HOME_PAGE = "https://github.com/msang/haspeede"


class HaSpeeDe2(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="hate", version=VERSION
        ),
        datasets.BuilderConfig(
            name="stereotype", version=VERSION  # ablation study
        ),
    ]

    def _info(self):
        type = self.config.name
        if type == 'hate':
            features = datasets.Features(
                {
                    'id': datasets.Value('string'),
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=[
                        'noHate',
                        'Hate',
                    ]),
                }
            )
        elif type == 'stereotype':
            features = datasets.Features(
                {
                    'id': datasets.Value('string'),
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=[
                        'noStereotype',
                        'Stereotype',
                    ]),
                }
            )
        else:
            raise NotImplementedError(f'Not supported config: {self.config.name}')

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOME_PAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        dl_dir = os.getenv("HASPEEDE2_URL")
        assert dl_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": dl_dir,
                    'split': 'train',
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": dl_dir,
                    'split': 'validation',
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": dl_dir,
                    'split': 'test',
                }
            ),
            # XXX consider adding separate news and twitter test splits
        ]

    def _get_dataAB(self, filepath):
        data = list()
        type = self.config.name
        with open(filepath, encoding="utf-8") as csv_file:

            csv_reader = csv.reader(
                csv_file,
                quotechar='"',
                delimiter="\t",
                quoting=csv.QUOTE_ALL,
                skipinitialspace=True
            )

            for row in csv_reader:
                id, text, hs, stereotype = row
                if type == 'hate':
                    label = 'noHate' if hs == '0' else 'Hate'
                elif type == 'stereotype':
                    label = 'noStereotype' if stereotype == '0' else 'Stereotype'

                item = {
                    'id': id,
                    'text': text,
                    'label': label,
                }
                data.append(item)

        return data

    def _generate_examples(self, filepath, split):
        if split == 'test':
            idx = 0
            for domain in ['news', 'tweets']:
                path = os.path.join(filepath, f'test_taskAB-{domain}.tsv')
                for item in self._get_dataAB(path):
                    yield idx, item
                    idx += 1

            return

        filepath = os.path.join(filepath, 'train_taskAB.tsv')
        data = self._get_dataAB(filepath)

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
