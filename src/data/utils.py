import os
import logging
from datasets import load_from_disk
from datasets.config import HF_DATASETS_CACHE
from datasets.fingerprint import Hasher

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
