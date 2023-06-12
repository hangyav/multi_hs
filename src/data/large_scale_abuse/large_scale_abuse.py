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


class LargeScaleAbuse(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="fine_grained", version=VERSION
        ),
        datasets.BuilderConfig(
            name="fine_grained_ab1", version=VERSION  # ablation study
        ),
        datasets.BuilderConfig(
            name="fine_grained_ab2", version=VERSION  # ablation study
        ),
        datasets.BuilderConfig(
            name="fine_grained_abhate", version=VERSION  # ablation study
        ),
        datasets.BuilderConfig(
            name="fine_grained_abnormal", version=VERSION  # ablation study
        ),
        datasets.BuilderConfig(
            name="fine_grained_ababusivenormal", version=VERSION  # ablation study
        ),
        datasets.BuilderConfig(
            name="fine_grained_abhatenormal", version=VERSION  # ablation study
        ),
    ]

    def _info(self):
        type = self.config.name
        if type == 'fine_grained':
            features = datasets.Features(
                {
                    'id': datasets.Value('string'),
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=['abusive', 'hateful', 'spam', 'normal']),
                }
            )
        elif type == 'fine_grained_ab1':
            features = datasets.Features(
                {
                    'id': datasets.Value('string'),
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=[
                        # 'abusive',
                        'hateful',
                        # 'spam',
                        'normal'
                    ]),
                }
            )
        elif type == 'fine_grained_ab2':
            features = datasets.Features(
                {
                    'id': datasets.Value('string'),
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=[
                        'abusive',
                        'hateful',
                        # 'spam',
                        'normal'
                    ]),
                }
            )
        elif type == 'fine_grained_abhate':
            features = datasets.Features(
                {
                    'id': datasets.Value('string'),
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=[
                        # 'abusive',
                        'hateful',
                        # 'spam',
                        # 'normal'
                    ]),
                }
            )
        elif type == 'fine_grained_abnormal':
            features = datasets.Features(
                {
                    'id': datasets.Value('string'),
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=[
                        # 'abusive',
                        # 'hateful',
                        # 'spam',
                        'normal'
                    ]),
                }
            )
        elif type == 'fine_grained_ababusivenormal':
            features = datasets.Features(
                {
                    'id': datasets.Value('string'),
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=[
                        'abusive',
                        # 'hateful',
                        # 'spam',
                        'normal'
                    ]),
                }
            )
        elif type == 'fine_grained_abhatenormal':
            features = datasets.Features(
                {
                    'id': datasets.Value('string'),
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=[
                        # 'abusive',
                        'hateful',
                        # 'spam',
                        'normal'
                    ]),
                }
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage="https://dataverse.mpi-sws.org/dataset.xhtml?persistentId=doi:10.5072/FK2/ZDTEMN",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        url = os.getenv("LARGE_SCALE_ABUSE_URL")
        assert url
        dl_dir = dl_manager.download_and_extract(url)

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
        ]

    def _generate_examples(self, filepath, split):
        data = list()
        type = self.config.name
        with open(filepath, encoding="utf-8") as csv_file:

            csv_reader = csv.reader(
                csv_file,
                quotechar='"',
                delimiter=",",
                quoting=csv.QUOTE_ALL,
                skipinitialspace=True
            )

            for row in csv_reader:
                if len(row) < 3:
                    continue

                id, text, label = row
                if label == '':
                    continue

                if type == 'fine_grained_ab1':
                    if label in {'abusive', 'spam'}:
                        continue
                elif type == 'fine_grained_ab2':
                    if label in {'spam'}:
                        continue
                elif type == 'fine_grained_abhate':
                    if label in {'spam', 'abusive', 'normal'}:
                        continue
                elif type == 'fine_grained_abnormal':
                    if label not in {'normal'}:
                        continue
                elif type == 'fine_grained_ababusivenormal':
                    if label in {'spam', 'hateful'}:
                        continue
                elif type == 'fine_grained_abhatenormal':
                    if label not in {'normal', 'hateful'}:
                        continue

                item = {
                    'id': id,
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

        if split == 'test':
            data = test
        else:
            data = train

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
