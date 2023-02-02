# XXX Consider https://github.com/makcedward/nlpaug for more sampling/generation options.

from collections import defaultdict
from datasets import Dataset, DatasetDict
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


def global_balanced_random_oversample(
        dataset_dict,
        features,
        label_map=None,
        label_col='label',
        random_state=0,
        load_from_cache_file=True,
        joint_label_col_name='label4sample',
        dataset_name_col_name='datasetname4sample',
):
    assert len(dataset_dict) > 0
    if label_map is None:
        label_map = defaultdict(dict)
    else:
        for map in label_map.values():
            for verbs in map.values():
                assert len(verbs) == 1

    tmp_dict = dict()
    for key, dataset in dataset_dict.items():
        for feature in features:
            tmp_lst = dataset[feature]
            if feature == 'id':
                tmp_lst = [str(item) for item in tmp_lst]
            tmp_dict.setdefault(feature, list()).extend(tmp_lst)

        tmp_dict.setdefault(joint_label_col_name, list()).extend([
            label_map[key].get(label, label)[0]
            for label in dataset[label_col]
        ])
        tmp_dict.setdefault(dataset_name_col_name, list()).extend([key]*len(dataset))

    tmp_df = Dataset.from_dict(tmp_dict)
    tmp_df = balanced_random_oversample(
        dataset=tmp_df,
        label_col=joint_label_col_name,
        random_state=random_state,
        load_from_cache_file=load_from_cache_file,
    )

    res = dict()
    for item in tmp_df:
        key = item[dataset_name_col_name]
        for feature in features:
            res.setdefault(key, dict()).setdefault(feature, list()).append(item[feature])

    res_data_dict = dict()
    for key, dataset in res.items():
        ds = Dataset.from_dict(dataset)
        orig = dataset_dict[key]
        ds._info = orig._info
        res_data_dict[key] = ds

    return DatasetDict(res_data_dict)


    # res_data_dict = DatasetDict({
    #     key: Dataset.from_dict(dataset)
    #     for key, dataset in res.items()
    # })
    #
    # return res_data_dict
