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
import re

import datasets
from sklearn.model_selection import train_test_split


_CITATION = """\
"""

_DESCRIPTION = """\
"""

_DATA_URL = "https://github.com/HKUST-KnowComp/MLMA_hate_speech/blob/master/hate_speech_mlma.zip?raw=true"

_DATA_FILES = {
    'en': 'hate_speech_mlma/en_dataset_with_stop_words.csv',
}

replace_by_blank_symbols = re.compile('\u00bb|\u00a0|\u00d7|\u00a3|\u00eb|\u00fb|\u00fb|\u00f4|\u00c7|\u00ab|\u00a0\ude4c|\udf99|\udfc1|\ude1b|\ude22|\u200b|\u2b07|\uddd0|\ude02|\ud83d|\u2026|\u201c|\udfe2|\u2018|\ude2a|\ud83c|\u2018|\u201d|\u201c|\udc69|\udc97|\ud83e|\udd18|\udffb|\ude2d|\udc80|\ud83e|\udd2a|\ud83e|\udd26|\u200d|\u2642|\ufe0f|\u25b7|\u25c1|\ud83e|\udd26|\udffd|\u200d|\u2642|\ufe0f|\udd21|\ude12|\ud83e|\udd14|\ude03|\ude03|\ude03|\ude1c|\udd81|\ude03|\ude10|\u2728|\udf7f|\ude48|\udc4d|\udffb|\udc47|\ude11|\udd26|\udffe|\u200d|\u2642|\ufe0f|\udd37|\ude44|\udffb|\u200d|\u2640|\udd23|\u2764|\ufe0f|\udc93|\udffc|\u2800|\u275b|\u275c|\udd37|\udffd|\u200d|\u2640|\ufe0f|\u2764|\ude48|\u2728|\ude05|\udc40|\udf8a|\u203c|\u266a|\u203c|\u2744|\u2665|\u23f0|\udea2|\u26a1|\u2022|\u25e1|\uff3f|\u2665|\u270b|\u270a|\udca6|\u203c|\u270c|\u270b|\u270a|\ude14|\u263a|\udf08|\u2753|\udd28|\u20ac|\u266b|\ude35|\ude1a|\u2622|\u263a|\ude09|\udd20|\udd15|\ude08|\udd2c|\ude21|\ude2b|\ude18|\udd25|\udc83|\ude24|\udc3e|\udd95|\udc96|\ude0f|\udc46|\udc4a|\udc7b|\udca8|\udec5|\udca8|\udd94|\ude08|\udca3|\ude2b|\ude24|\ude23|\ude16|\udd8d|\ude06|\ude09|\udd2b|\ude00|\udd95|\ude0d|\udc9e|\udca9|\udf33|\udc0b|\ude21|\udde3|\ude37|\udd2c|\ude21|\ude09|\ude39|\ude42|\ude41|\udc96|\udd24|\udf4f|\ude2b|\ude4a|\udf69|\udd2e|\ude09|\ude01|\udcf7|\ude2f|\ude21|\ude28|\ude43|\udc4a|\uddfa|\uddf2|\udc4a|\ude95|\ude0d|\udf39|\udded|\uddf7|\udded|\udd2c|\udd4a|\udc48|\udc42|\udc41|\udc43|\udc4c|\udd11|\ude0f|\ude29|\ude15|\ude18|\ude01|\udd2d|\ude43|\udd1d|\ude2e|\ude29|\ude00|\ude1f|\udd71|\uddf8|\ude20|\udc4a|\udeab|\udd19|\ude29|\udd42|\udc4a|\udc96|\ude08|\ude0d|\udc43|\udff3|\udc13|\ude0f|\udc4f|\udff9|\udd1d|\udc4a|\udc95|\udcaf|\udd12|\udd95|\udd38|\ude01|\ude2c|\udc49|\ude01|\udf89|\udc36|\ude0f|\udfff|\udd29|\udc4f|\ude0a|\ude1e|\udd2d|\uff46|\uff41|\uff54|\uff45|\uffe3|\u300a|\u300b|\u2708|\u2044|\u25d5|\u273f|\udc8b|\udc8d|\udc51|\udd8b|\udd54|\udc81|\udd80|\uded1|\udd27|\udc4b|\udc8b|\udc51|\udd90|\ude0e')
replace_by_apostrophe_symbol = re.compile('\u2019')
replace_by_dash_symbol = re.compile('\u2014')
replace_by_u_symbols = re.compile('\u00fb|\u00f9')
replace_by_a_symbols = re.compile('\u00e2|\u00e0')
replace_by_c_symbols = re.compile('\u00e7')
replace_by_i_symbols = re.compile('\u00ee|\u00ef')
replace_by_o_symbols = re.compile('\u00f4')
replace_by_oe_symbols = re.compile('\u0153')
replace_by_e_symbols = re.compile('\u00e9|\u00ea|\u0117|\u00e8')
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|,;]')


class MLMA(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="en-sentiment_rep", version=VERSION
        ),
        datasets.BuilderConfig(
            name="en-sentiment_rep_ab1", version=VERSION  # ablation study
        ),
        datasets.BuilderConfig(
            name="en-sentiment_rep_ab2", version=VERSION  # ablation study
        ),
        datasets.BuilderConfig(
            name="en-sentiment_rep_abhateoffensive", version=VERSION  # ablation study
        ),
        datasets.BuilderConfig(
            name="en-sentiment_rep_ababusivenormal", version=VERSION  # ablation study
        ),
        datasets.BuilderConfig(
            name="en-sentiment_rep_aboffensivenormal", version=VERSION  # ablation study
        ),
        datasets.BuilderConfig(
            name="en-sentiment_rep_abhatenormal", version=VERSION  # ablation study
        ),
        datasets.BuilderConfig(
            name="en-directness", version=VERSION
        ),
        # datasets.BuilderConfig(
        #     name="en-annotator_sentiment", version=VERSION
        # ),
        datasets.BuilderConfig(
            name="en-target", version=VERSION
        ),
        # datasets.BuilderConfig(
        #     name="en-group", version=VERSION
        # ),
    ]

    def _info(self):
        lang, type = self.config.name.split('-')
        if type == "sentiment_rep":
            features = datasets.Features(
                {
                    'id': datasets.Value('string'),
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=['abusive', 'hateful', 'offensive', 'disrespectful', 'fearful', 'normal']),
                }
            )
        elif type == "sentiment_rep_ab1":
            features = datasets.Features(
                {
                    'id': datasets.Value('string'),
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=[
                        # 'abusive',
                        'hateful',
                        'offensive',
                        # 'disrespectful',
                        # 'fearful',
                        'normal'
                    ]),
                }
            )
        elif type == "sentiment_rep_ab2":
            features = datasets.Features(
                {
                    'id': datasets.Value('string'),
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=[
                        'abusive',
                        'hateful',
                        'offensive',
                        # 'disrespectful',
                        # 'fearful',
                        'normal'
                    ]),
                }
            )
        elif type == "sentiment_rep_abhateoffensive":
            features = datasets.Features(
                {
                    'id': datasets.Value('string'),
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=[
                        # 'abusive',
                        'hateful',
                        'offensive',
                        # 'disrespectful',
                        # 'fearful',
                        # 'normal'
                    ]),
                }
            )
        elif type == "sentiment_rep_ababusivenormal":
            features = datasets.Features(
                {
                    'id': datasets.Value('string'),
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=[
                        'abusive',
                        # 'hateful',
                        # 'offensive',
                        # 'disrespectful',
                        # 'fearful',
                        'normal'
                    ]),
                }
            )
        elif type == "sentiment_rep_aboffensivenormal":
            features = datasets.Features(
                {
                    'id': datasets.Value('string'),
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=[
                        # 'abusive',
                        # 'hateful',
                        'offensive',
                        # 'disrespectful',
                        # 'fearful',
                        'normal'
                    ]),
                }
            )
        elif type == "sentiment_rep_abhatenormal":
            features = datasets.Features(
                {
                    'id': datasets.Value('string'),
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=[
                        # 'abusive',
                        'hateful',
                        # 'offensive',
                        # 'disrespectful',
                        # 'fearful',
                        'normal'
                    ]),
                }
            )
        elif type == "directness":
            features = datasets.Features(
                {
                    'id': datasets.Value('string'),
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=['direct', 'indirect']),
                }
            )
        elif type == "target":
            features = datasets.Features(
                {
                    'id': datasets.Value('string'),
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=['origin', 'gender', 'sexual_orientation', 'religion', 'disability', 'other']),
                }
            )
        elif type == "group":
            features = datasets.Features(
                {
                    'id': datasets.Value('string'),
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=['individual', 'women', 'special_needs', 'african_descent', 'other']),
                }
            )
        else:
            raise NotImplementedError(f'Not supported config: {self.config.name}')

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage="https://github.com/HKUST-KnowComp/MLMA_hate_speech",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(_DATA_URL)

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

    def _process_text(self, text):
        # from https://github.com/HKUST-KnowComp/MLMA_hate_speech/blob/master/annotated_data_processing.py
        text = replace_by_e_symbols.sub('e', text)
        text = replace_by_a_symbols.sub('a', text)
        text = replace_by_o_symbols.sub('o', text)
        text = replace_by_oe_symbols.sub('oe', text)
        text = replace_by_u_symbols.sub('e', text)
        text = replace_by_i_symbols.sub('e', text)
        text = replace_by_u_symbols.sub('e', text)
        text = replace_by_apostrophe_symbol.sub("'", text)
        text = replace_by_dash_symbol.sub("_", text)
        text = replace_by_blank_symbols.sub('', text)

        text = text.replace("\\", "")
        # text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
        return text

    def _generate_examples(self, filepath, split):
        data = list()
        language, type = self.config.name.split('-')
        with open(os.path.join(filepath, _DATA_FILES[language]), encoding="utf-8") as csv_file:

            csv_reader = csv.reader(
                csv_file,
                quotechar='"',
                delimiter=",",
                quoting=csv.QUOTE_ALL,
                skipinitialspace=True
            )

            next(csv_reader)

            for row in csv_reader:
                id, text, sentiment, directness, annotator, target, group = row
                text = self._process_text(text)

                if type == 'sentiment_rep':
                    for idx, label in enumerate(sentiment.split('_')):
                        item = {
                            'id': f'{id}.{idx}',
                            'text': text,
                            'label': label,
                        }
                        data.append(item)

                    continue
                elif type == 'sentiment_rep_ab1':
                    for idx, label in enumerate(sentiment.split('_')):
                        if label in {'abusive', 'disrespectful', 'fearful'}:
                            continue
                        item = {
                            'id': f'{id}.{idx}',
                            'text': text,
                            'label': label,
                        }
                        data.append(item)

                    continue
                elif type == 'sentiment_rep_ab2':
                    for idx, label in enumerate(sentiment.split('_')):
                        if label in {'disrespectful', 'fearful'}:
                            continue
                        item = {
                            'id': f'{id}.{idx}',
                            'text': text,
                            'label': label,
                        }
                        data.append(item)

                    continue
                elif type == 'sentiment_rep_abhateoffensive':
                    for idx, label in enumerate(sentiment.split('_')):
                        if label in {'disrespectful', 'fearful', 'abusive', 'normal'}:
                            continue
                        item = {
                            'id': f'{id}.{idx}',
                            'text': text,
                            'label': label,
                        }
                        data.append(item)

                    continue
                elif type == 'sentiment_rep_ababusivenormal':
                    for idx, label in enumerate(sentiment.split('_')):
                        if label not in {'abusive', 'normal'}:
                            continue
                        item = {
                            'id': f'{id}.{idx}',
                            'text': text,
                            'label': label,
                        }
                        data.append(item)

                    continue
                elif type == 'sentiment_rep_aboffensivenormal':
                    for idx, label in enumerate(sentiment.split('_')):
                        if label not in {'offensive', 'normal'}:
                            continue
                        item = {
                            'id': f'{id}.{idx}',
                            'text': text,
                            'label': label,
                        }
                        data.append(item)

                    continue
                elif type == 'sentiment_rep_abhatenormal':
                    for idx, label in enumerate(sentiment.split('_')):
                        if label not in {'hateful', 'normal'}:
                            continue
                        item = {
                            'id': f'{id}.{idx}',
                            'text': text,
                            'label': label,
                        }
                        data.append(item)

                    continue
                elif type == 'directness':
                    label = directness
                elif type == 'target':
                    label = target
                else:
                    label = group

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

