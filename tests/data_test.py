import pytest

from collections import Counter
from datasets import load_dataset
from src.data.sampling import balanced_random_oversample


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
