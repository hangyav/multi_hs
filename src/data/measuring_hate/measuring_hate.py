
import datasets
from datasets import load_dataset
from sklearn.model_selection import train_test_split


_CITATION = """\
"""

_DESCRIPTION = """\
"""

_DATA_URL = ""

_HOME_PAGE = "https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech"


class LargeScaleXDomain(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name=f'{type}-{setup}', version=datasets.Version("1.0.0"))
        for type in ['binary', 'fine_grained']  # binary: ignores counter speech
        for setup in ['full', 'politics']
    ]

    def _info(self):
        type, setup = self.config.name.split('-')
        if type == "binary":
            features = datasets.Features(
                {
                    'id': datasets.Value('string'),
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=['normal', 'hate']),
                }
            )
        elif type == "fine_grained":
            features = datasets.Features(
                {
                    'id': datasets.Value('string'),
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=['counter', 'normal', 'hate']),
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
        data = load_dataset('ucberkeley-dlab/measuring-hate-speech')['train']

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    'dataset': data,
                    'split': 'train',
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    'dataset': data,
                    'split': 'validation',
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    'dataset': data,
                    'split': 'test',
                }
            ),
        ]

    def _generate_examples(self, dataset, split):
        type, setup = self.config.name.split('-')

        data = list()
        for item in dataset:

            if setup == 'politics' and not item['target_religion']:
                continue

            label = 'counter'
            if -1.0 <= item['hate_speech_score'] <= 0.5:
                label = 'normal'
            else:
                label = 'hate'

            if type == 'binary' and label == 'counter':
                continue

            data.append({
                'id': item['comment_id'],
                'text': item['text'],
                'label': label,
            })

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

