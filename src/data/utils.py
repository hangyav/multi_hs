import os
import logging
import random


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


def reduce_dataset_if_needed(dataset, num_samples, data_selector_method,
                             seed):
    if num_samples is None:
        return dataset
    if data_selector_method == 'stratify':
        dataset = random_select(
            dataset,
            num_samples,
            seed,
        )
    elif data_selector_method == 'per_label':
        dataset = per_label_select(
            dataset,
            num_samples,
            seed,
        )
    else:
        raise NotADirectoryError(f'Data selection method not supported: {data_selector_method}')
    return dataset


def random_select(dataset, num_sample, min_per_label=1, seed=0):
    labels = set([item['label'] for item in dataset])
    res_dataset, _ = train_test_split(
            dataset,
            train_size=num_sample - len(labels)*min_per_label,
            random_state=seed,
            stratify=[item['label'] for item in dataset]
    )

    min_items = {
        label: min_per_label
        for label in labels
    }
    already_selected = set(res_dataset['id'])
    tmp_lst = [item for item in dataset if item['id'] not in already_selected]
    random.Random(seed).shuffle(tmp_lst)

    for item in tmp_lst:
        label = item['label']

        if label in min_items:
            for k, v in item.items():
                res_dataset[k].append(v)

            min_items[label] -= 1
            if min_items[label] <= 0:
                del min_items[label]

        if len(min_items) == 0:
            break
    else:
        raise ValueError(f'There are no enough elements for each label. Missing: {min_items}')

    return Dataset.from_dict(res_dataset)


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
