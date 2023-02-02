import pytest

from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer
from src.data.sampling import balanced_random_oversample, global_balanced_random_oversample
from src.data.utils import per_label_select
from src.data.prompting import get_pvp


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained('bert-base-multilingual-cased')


@pytest.fixture
def hate_speech18_binary():
    import src.data.hate_speech18
    return load_dataset('src/data/hate_speech18', 'binary')


@pytest.fixture
def hasoc19_en_fine_grained():
    import src.data.hasoc19
    return load_dataset('src/data/hasoc19', 'en-fine_grained')


@pytest.fixture
def told_br_find_grained():
    import src.data.told_br
    return load_dataset('src/data/told_br', 'fine_grained')


@pytest.mark.parametrize('dataset', [
    pytest.lazy_fixture('hate_speech18_binary'),
    pytest.lazy_fixture('hasoc19_en_fine_grained'),
])
def test_balanced_oversample(dataset):
    sampled = balanced_random_oversample(dataset['train'], 'label', 0, False)
    counter = Counter(sampled['label'])

    assert len(set(counter.values())) == 1


def test_per_label_select(hasoc19_en_fine_grained):
    dataset = per_label_select(hasoc19_en_fine_grained['train'], 5)
    for k, v in Counter([item['label'] for item in dataset]).items():
        assert (k, v) == (k, 5)


def test_per_label_select_random_seed(hasoc19_en_fine_grained):
    dataset1 = per_label_select(hasoc19_en_fine_grained['train'], 5, 0)
    dataset2 = per_label_select(hasoc19_en_fine_grained['train'], 5, 0)

    for item1, item2 in zip(dataset1, dataset2):
        assert item1 == item2


def test_global_balanced_oversample(hate_speech18_binary, told_br_find_grained):
    data_dict = {
        'hate_speech18': hate_speech18_binary['train'],
        'told_br': told_br_find_grained['train'],
    }
    res = global_balanced_random_oversample(
        data_dict,
        ['id', 'text', 'label'],
        label_map={
            key: get_pvp(dataset).get_label_map(pid, dataset.features['label'].names)
            for (key, dataset), pid in zip(data_dict.items(), [0, 0])
        },
        label_col='label',
        random_state=0,
        load_from_cache_file=False,
    )

    assert res.keys() == data_dict.keys()
    c = Counter(
        get_pvp(dataset).get_label_map(pid, dataset.features['label'].names)[label][0]
        for (key, dataset), pid in zip(res.items(), [0, 0])
        for label in dataset['label']
    )
    assert len(set(c.values())) == 1
