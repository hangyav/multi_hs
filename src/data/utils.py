import os
import logging
from datasets import load_from_disk, Dataset
from datasets.config import HF_DATASETS_CACHE
from datasets.fingerprint import Hasher

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

logger = logging.getLogger(__name__)


def cache_dataset(dataset, *args):
    fingerprint = Hasher.hash(args)
    path = os.path.join(
        HF_DATASETS_CACHE,
        'custom_data_cache',
        f'{fingerprint}.cache'
    )
    logger.info(f'Caching file to: {path}')
    dataset.save_to_disk(path)

    return load_from_disk(path)


def load_cached_dataset(*args):
    fingerprint = Hasher.hash(args)
    path = os.path.join(
        HF_DATASETS_CACHE,
        'custom_data_cache',
        f'{fingerprint}.cache'
    )
    logger.warning(f'TRYING TO LOAD FROM: {path}')
    if not os.path.exists(path):
        return None

    logger.info(f'Loading from cache: {path}')
    logger.info('WARNING: Data change not checked!')

    return load_from_disk(path)


def apply_on_dataset(func, dataset, load_from_cache_file=True):
    # FIXME for some reason the hash of func changes between the load and cache
    # functions. So this is a workaround for now.
    fingerprint = Hasher.hash([dataset, func])
    if load_from_cache_file:
        res = load_cached_dataset(
            #  dataset,
            #  func,
            fingerprint,
        )
        if res is not None:
            return res

    res = func(dataset)

    res = cache_dataset(
        res,
        #  dataset,
        #  func,
        fingerprint,
    )

    return res


def reduce_dataset_if_needed(dataset, num_samples, data_args, training_args):
    if num_samples is None:
        return dataset
    if data_args.data_selector_method == 'stratify':
        dataset = random_select(
            dataset,
            num_samples,
            training_args.seed,
        )
    elif data_args.data_selector_method == 'per_label':
        dataset = per_label_select(
            dataset,
            num_samples,
            training_args.seed,
        )
    else:
        raise NotADirectoryError(f'Data selection method not supported: {data_args.data_selector_method}')
    return dataset


def random_select(dataset, num_sample, seed=0):
    dataset, _ = train_test_split(
            dataset,
            train_size=num_sample,
            random_state=seed,
            stratify=[item['label'] for item in dataset]
    )
    return Dataset.from_dict(dataset)


def per_label_select(dataset, num_samples, seed=0):
    dataset = shuffle([item for item in dataset], random_state=seed)
    res = list()
    label_nums = {label: num_samples for label in {item['label'] for item in dataset}}
    for item in dataset:
        label = item['label']
        if label not in label_nums:
            continue

        res.append(item)
        label_nums[label] -= 1
        if label_nums[label] == 0:
            label_nums.pop(label)

        if len(label_nums) == 0:
            break
    else:
        raise ValueError(f'Not enough instances for all labels. Remaining: {label_nums}')

    return Dataset.from_dict({
        col: [item[col] for item in res]
        for col in res[0].keys()
    })
