# XXX Consider https://github.com/makcedward/nlpaug for more sampling/generation options.

from datasets import Dataset
from imblearn.over_sampling import RandomOverSampler
from .utils import apply_on_dataset


def sample_dataset(dataset, sampler, label_col, load_from_cache_file=True):
    func = lambda x: Dataset.from_pandas(
        sampler.fit_resample(
            x.to_pandas(),
            x[label_col],
        )[0]
    )

    return apply_on_dataset(
        func,
        dataset,
        load_from_cache_file=load_from_cache_file
    )


def balanced_random_oversample(dataset, label_col='label', random_state=0,
                               load_from_cache_file=True):
    sampler = RandomOverSampler(
        sampling_strategy='not majority',
        random_state=random_state
    )

    return sample_dataset(
        dataset,
        sampler,
        label_col,
        load_from_cache_file,
    )
