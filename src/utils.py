import os
import logging
import torch
import importlib
from transformers.integrations import (
    TensorBoardCallback,
    rewrite_logs,
)

from src.data.prompting import get_pvp_by_name_and_config


logger = logging.getLogger(__name__)


PATH_NAME_MAX = 255


class TensorBoardLogitsCallback(TensorBoardCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        log_dir = None

        if state.is_hyper_param_search:
            trial_name = state.trial_name
            if trial_name is not None:
                log_dir = os.path.join(args.logging_dir, trial_name)

        if self.tb_writer is None:
            self._init_summary_writer(args, log_dir)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return

        if self.tb_writer is None:
            self._init_summary_writer(args)

        if self.tb_writer is not None:
            logs = rewrite_logs(logs)
            for k, v in logs.items():
                if isinstance(v, torch.Tensor):
                    self.tb_writer.add_histogramm(k, v, state.global_step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute."
                    )
            self.tb_writer.flush()


def import_dataset(dataset_name, package_path='src.data'):
    importlib.import_module(
        f'{package_path}.{dataset_name}.{dataset_name}'
    ).__file__,


def _get_verbs(dataset, pid, import_dataset_module=True):
    if import_dataset_module:
        import_dataset(dataset.info.splits['train'].dataset_name)

    pvp = get_pvp_by_name_and_config(
        dataset.info.splits['train'].dataset_name,
        dataset.config_name
    )
    verbalizer = pvp.prompt_verbalizers[pvp.pvps[pid][1]]
    res = list()
    for label in dataset.features['label'].names:
        verb = verbalizer.verbs[label]
        assert len(verb) == 1
        res.append(verb[0])
    return res


def save_hf_dataset(dataset, fout, pid=None):
    labels = dataset.features["label"].names
    if pid is not None:
        labels = _get_verbs(dataset, pid)
    for item in dataset:
        print(f"{item['text']}\t{labels[item['label']]}", file=fout)


def save_hf_datasets(dataset, dir, pid=None,
                     splits=['train', 'validation', 'test'],
                     names=['train.txt', 'valid.txt', 'test.txt']):
    for split, name in zip(splits, names):
        df = dataset[split]
        with open(os.path.join(dir, name), 'w') as fout:
            save_hf_dataset(df, fout, pid)


def truncate(string, length, suffix='...'):
    if len(string) > length:
        string = string[:length - len(suffix)] + suffix

    return string
