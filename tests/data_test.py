import pytest

from collections import Counter
from datasets import load_dataset
from src.data.sampling import balanced_random_oversample
from src.data.utils import per_label_select


@pytest.fixture
def hate_speech18_binary():
    return load_dataset('src/data/hate_speech18', 'binary')


@pytest.fixture
def hasoc19_en_fine_grained():
    return load_dataset('src/data/hasoc19', 'en-fine_grained')


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
