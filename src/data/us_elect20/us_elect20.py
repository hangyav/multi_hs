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

_DATA_URL = "https://www.romanklinger.de/data-sets/GrimmingerKlingerWASSA2021.zip"

_HOME_PAGE = "https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/stance-hof"


class UsElection20(datasets.GeneratorBasedBuilder):
    """Hate speech dataset"""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="hof", version=VERSION
        ),
    ]

    def _info(self):
        if self.config.name == "hof":
            features = datasets.Features(
                {
                    "id": datasets.Value("int64"),
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=[
                        "Non-Hateful",
                        "Hateful",
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
        dl_dir = dl_manager.download_and_extract(_DATA_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(dl_dir, "train.tsv"),
                    'split': 'train',
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(dl_dir, "train.tsv"),
                    'split': 'validation',
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(dl_dir, "test.tsv"),
                    'split': 'test',
                }
            ),
        ]

    def _generate_examples(self, filepath, split):
        data = list()
        with open(filepath, encoding="utf-8") as csv_file:

            csv_reader = csv.reader(
                csv_file,
                quotechar='"',
                delimiter="\t",
                quoting=csv.QUOTE_ALL,
                skipinitialspace=True
            )

            next(csv_reader)

            for idx, row in enumerate(csv_reader):
                try:
                    text, trump, biden, west, hof = row
                except ValueError:
                    # There's an empty row at the end of the files
                    continue

                text = text.replace('[NEWLINE]', '\n')

                if self.config.name == 'hof':
                    label = hof

                data.append({
                    "id": idx,
                    "text": text,
                    "label": label,
                })

        if split in {'train', 'validation'}:
            train, valid = train_test_split(
                data,
                test_size=0.2,
                random_state=0,
                stratify=[item['label'] for item in data]
            )

            if split == 'train':
                data = train
            elif split == 'validation':
                data = valid

        for idx, item in enumerate(data):
            yield idx, item
