#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
# Based on run_glue.py of adapter-transformers

import logging
import os
import random
import sys
import importlib
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
from functools import partial

import datasets
from datasets import load_dataset, load_metric

from torch.nn.functional import softmax
import transformers
from transformers import (
    # AdapterConfig,
    # AutoAdapterModel,
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    # MultiLingAdapterArguments,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EarlyStoppingCallback,
    AutoModelForMaskedLM,
    BertForMaskedLM,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from src.data import sampling
from src.data.utils import reduce_dataset_if_needed
from src.modeling import (
    MultiTaskModelWrapper,
    HeadSelectionWrapper,
    AdapterSelectionWrapper,
    PromptSelectionWrapper,
)
from src.training import (
    MultitaskAdapterTrainer,
    MultitaskTrainer,
    MLMMultitaskTrainer,
    MLMMultitaskAdapterTrainer,
)


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)


def setup_max_samples(max_sample, num):
    if max_sample is None:
        max_sample = [None] * num
    elif len(max_sample) == 1:
        max_sample = max_sample * num
    elif len(max_sample) == 2 and num > 2:
        # first value for all but last
        max_sample = [max_sample[0]] * (num-1) + [max_sample[1]]
    max_sample = [None if item == -1 else item for item in max_sample]

    return max_sample


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[List[str]] = field(
        default=None,
        metadata={"help": "The name of the task. Just meta-info."},
    )
    dataset_name: Optional[List[str]] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[List[str]] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    data_selector_method_train: Optional[str] = field(
        default='stratify', metadata={"help": "If max_train/eval/predict_sample is set, how to subsample: {per_label, stratify}"}
    )
    data_selector_method_eval: Optional[str] = field(
        default='stratify', metadata={"help": "If max_train/eval/predict_sample is set, how to subsample: {per_label, stratify}"}
    )
    data_selector_method_predict: Optional[str] = field(
        default='stratify', metadata={"help": "If max_train/eval/predict_sample is set, how to subsample: {per_label, stratify}"}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    train_sampling: Optional[str] = field(
        default=None,
        metadata={
            "help": "Data sampling for label imbalance. Options: balanced_over,"
            "a global_balanced_over"
        },
    )
    prompt_ids: List[int] = field(
        default=None,
        metadata={"help": "Which prompt to use for each dataset"}
    )
    do_mlm: bool = field(
        default=False,
        metadata={"help": "Whether to do additional MLM on top of classification training."}
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    mlm_weight: float = field(
        default=1.0,
        metadata={"help": "Weight of mlm_loss in the combine loss function."}
    )

    def __post_init__(self):
        assert self.dataset_name is not None and self.dataset_config_name is not None
        assert len(self.dataset_name) == len(self.dataset_config_name)
        assert self.prompt_ids is None or len(self.prompt_ids) == len(self.dataset_name)

        if self.task_name is None:
            self.task_name = [
                f'{dataset_name}-{dataset_config_name}'
                for dataset_name, dataset_config_name in zip(self.dataset_name, self.dataset_config_name)
            ]
        else:
            assert len(self.dataset_name) == len(self.task_name)

        if self.train_file is not None:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                validation_extension = self.validation_file.split(".")[-1]
                assert (
                    validation_extension == train_extension
                ), "`validation_file` should have the same extension (csv or json) as `train_file`."

        if self.train_sampling is not None and self.train_sampling.lower() == 'none':
            self.train_sampling = None

        self.max_train_samples = setup_max_samples(
            self.max_train_samples,
            len(self.dataset_name)
        )
        assert len(self.max_train_samples) == len(self.dataset_name)
        self.max_eval_samples = setup_max_samples(
            self.max_eval_samples,
            len(self.dataset_name)
        )
        assert len(self.max_eval_samples) == len(self.dataset_name)
        self.max_predict_samples = setup_max_samples(
            self.max_predict_samples,
            len(self.dataset_name)
        )
        assert len(self.max_predict_samples) == len(self.dataset_name)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    model_type: str = field(
        default="classifier",
        metadata={"help": "Options: classifier, prompt"},
    )

    def __post_init__(self):
        if self.model_type == 'prompt' and self.use_fast_tokenizer:
            logger.info('Fast tokenizer is not supported by prompts. Ignoring')
            self.use_fast_tokenizer = False


@dataclass
class MyTrainingArguments(TrainingArguments):
    early_stopping_patience: int = field(
        default=-1,
        metadata={
            "help": ">=0 to set early stopping"
        },
    )
    early_stopping_threshold: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "Denote how much the specified metric must improve to satisfy early stopping conditions."
        },
    )
    metric_for_best_model: Optional[str] = field(
        default='eval_macro_averaged_F1_MULTIAVG',
        metadata={"help": "The metric to use to compare two models."}
    )
    freeze_model_core: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the core LM."},
    )
    do_visualize: bool = field(
        default=False,
        metadata={"help": "Visualize fusion weights"},
    )
    do_debug_predictions: bool = field(
        default=False,
        metadata={"help": ""},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.early_stopping_patience >= 0:
            self.load_best_model_at_end = True


@dataclass
class AdapterArguments:
    """
    The subset of arguments related to adapter training.
    """

    train_adapter: bool = field(default=False, metadata={"help": "Train an adapter instead of the full model."})
    load_adapter: Optional[str] = field(
        default="", metadata={"help": "Pre-trained adapter module to be loaded from Hub."}
    )
    adapter_config: Optional[str] = field(
        default="pfeiffer", metadata={"help": "Adapter configuration. Either an identifier or a path to a file."}
    )
    adapter_non_linearity: Optional[str] = field(
        default=None, metadata={"help": "Override the non-linearity of the adapter configuration."}
    )
    adapter_reduction_factor: Optional[float] = field(
        default=None, metadata={"help": "Override the reduction factor of the adapter configuration."}
    )
    language: Optional[str] = field(default=None, metadata={"help": "The training language, e.g. 'en' for English."})



@dataclass
class MultiLingAdapterArguments(AdapterArguments):
    """
    Arguemnts related to adapter training, extended by arguments for multilingual setups.
    """

    load_lang_adapter: Optional[str] = field(
        default=None, metadata={"help": "Pre-trained language adapter module to be loaded from Hub."}
    )
    lang_adapter_config: Optional[str] = field(
        default=None, metadata={"help": "Language adapter configuration. Either an identifier or a path to a file."}
    )
    lang_adapter_non_linearity: Optional[str] = field(
        default=None, metadata={"help": "Override the non-linearity of the language adapter configuration."}
    )
    lang_adapter_reduction_factor: Optional[int] = field(
        default=None, metadata={"help": "Override the reduction factor of the language adapter configuration."}
    )


@dataclass
class MyAdapterArguments(MultiLingAdapterArguments):
    load_adapter: Optional[List[str]] = field(
        default=None, metadata={"help": "Pre-trained adapter module to be loaded from Hub."}
    )
    fuse_adapters: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Adapter names to fuse."}
    )
    load_lang_adapter: Optional[List[str]] = field(
        default=None, metadata={"help": "Pre-trained language adapter module to be loaded from Hub."}
    )
    language: Optional[List[str]] = field(default=None, metadata={"help": "The training language, e.g. 'en' for English."})

    def __post_init__(self):
        if self.load_lang_adapter is not None:
            assert self.language is not None
            assert len(self.load_lang_adapter) == len(self.language)


def setup():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments, MyAdapterArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()

    assert adapter_args.load_lang_adapter is None or len(adapter_args.load_lang_adapter) == len(data_args.task_name)
    assert adapter_args.fuse_adapters is None or len(data_args.task_name) == 1

    if model_args.model_type == 'prompt':
        assert data_args.prompt_ids is not None
        training_args.label_names = ['label']

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}"
        + f", device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f", distributed training: {bool(training_args.local_rank != -1)}"
        + f", 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    np.random.seed(training_args.seed)

    return model_args, data_args, training_args, adapter_args


def get_datasets(model_args, data_args, training_args):
    # For CSV/JSON files, this script will use as labels the column called
    # 'label' and as pair of sentences the sentences in columns called
    # 'sentence1' and 'sentence2' if such column exists or the first two
    # columns not named label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does
    # single sentence classification on this single column. You can easily
    # tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only
    # one local process can concurrently download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = {
            f'{dataset_name}-{dataset_config_name}': load_dataset(
                importlib.import_module(
                    f'src.data.{dataset_name}.{dataset_name}'
                ).__file__,
                dataset_config_name,
                cache_dir=model_args.cache_dir,
            )
            for dataset_name, dataset_config_name in zip(data_args.dataset_name, data_args.dataset_config_name)
        }
    else:
        # TODO need to set the right adapter name and handle label2id conversion
        raise NotImplementedError('Not supported currently.')
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # num_labels, label_list, is_regression
    dataset_metadata = dict()

    if model_args.model_type == 'prompt':
        from src.data.prompting import get_pvp
    else:
        get_pvp = lambda x: None
    # Labels
    # Trying to have good defaults here, don't hesitate to tweak to your needs.
    for dataset_name, dataset in raw_datasets.items():
        is_regression = dataset["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
            label_list = None
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = dataset["train"].features["label"].names
            num_labels = len(label_list)

        dataset_metadata[dataset_name] = (
            num_labels,
            label_list,
            is_regression,
            get_pvp(dataset['train'])
        )

    return raw_datasets, dataset_metadata


def setup_model(model, tokenizer, dataset_metadata, model_args, data_args,
                training_args, adapter_args):
    if model_args.model_type == 'classifier':
        for i, task_name in enumerate(data_args.task_name):
            dataset_name = f'{data_args.dataset_name[i]}-{data_args.dataset_config_name[i]}'
            num_labels, label_list, _, _ = dataset_metadata[dataset_name]
            if task_name not in model.config.prediction_heads:
                # TODO model.load_head() should be used if trained head exists and
                # adapters are used. However, model.load_adapter() seems to do this if
                # head and adapter are at the same path, so I leave this as is for now.
                model.add_classification_head(
                    task_name,
                    num_labels=num_labels,
                    id2label={i: v for i, v in enumerate(label_list)} if num_labels > 0 else None,
                )

    lang_adapter_names, adapter_setup = setup_adapters_if_needed(
        model,
        model_args,
        data_args,
        training_args,
        adapter_args
    )
    model, tokenizer, wrapper = setup_propting_if_needed(
        model, tokenizer, model_args)

    wrap_model(model, tokenizer, model_args, data_args, training_args, adapter_args,
               lang_adapter_names, adapter_setup, dataset_metadata)

    return model, tokenizer, wrapper


def wrap_model(model, tokenizer, model_args, data_args, training_args, adapter_args,
               lang_adapter_names, adapter_setup, dataset_metadata):
    if model_args.model_type == 'classifier':
        if not adapter_args.train_adapter:
            head_dict = {
                f'{dataset_name}-{dataset_config_name}': task_name
                for task_name, dataset_name, dataset_config_name in zip(
                    data_args.task_name,
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                )
            }
            HeadSelectionWrapper(model, head_dict)
        else:
            import transformers.adapters.composition as ac
            if adapter_args.fuse_adapters is not None:
                # there's only one task in this case.
                model.train_adapter_fusion(adapter_setup)
                MultiTaskModelWrapper(model)
            else:
                adapter_dict = {
                    f'{dataset_name}-{dataset_config_name}': [task_name]
                    for task_name, dataset_name, dataset_config_name in zip(
                        data_args.task_name,
                        data_args.dataset_name,
                        data_args.dataset_config_name,
                    )
                }
                active_adapter_dict = None
                if lang_adapter_names is not None:
                    active_adapter_dict = {
                        f'{dataset_name}-{dataset_config_name}': ac.Stack(lang_adapter, task_name)
                        for task_name, dataset_name, dataset_config_name, lang_adapter in zip(
                            data_args.task_name,
                            data_args.dataset_name,
                            data_args.dataset_config_name,
                            lang_adapter_names,
                        )
                    }

                # We need to activate any adapter now
                # Freeze all model weights except of those of this adapter
                model.train_adapter(next(adapter_dict.values().__iter__()))
                # Set the adapters to be used in every forward pass
                tmp_dict = active_adapter_dict if active_adapter_dict else adapter_dict
                model.set_active_adapters(next(tmp_dict.values().__iter__()))

                AdapterSelectionWrapper(model, adapter_dict, active_adapter_dict,
                                        training_args.freeze_model_core)
    elif model_args.model_type == 'prompt':
        from openprompt import PromptForClassification
        prompt_model_dict = {
            f'{dataset_name}-{dataset_config}': dataset_metadata[f'{dataset_name}-{dataset_config}'][3].get_pvp(pattern_id, tokenizer, model)
            for dataset_name, dataset_config, pattern_id in zip(
                data_args.dataset_name,
                data_args.dataset_config_name,
                data_args.prompt_ids
            )
        }
        prompt_model_dict = {
            # FIXME manual moving to cuda is not nice at all
            k: PromptForClassification(
                template=v.template,
                verbalizer=v.verbalizer,
                plm=model,
                ).to('cpu' if training_args.no_cuda else 'cuda:0')
            for k, v in prompt_model_dict.items()
        }
        PromptSelectionWrapper(model, prompt_model_dict)


def setup_adapters_if_needed(model, model_args, data_args, training_args,
                             adapter_args):
    lang_adapter_names = None
    adapter_setup = None
    if adapter_args.train_adapter:
        from transformers import AdapterConfig

        for adapter in list(model.config.adapters.adapters.keys()):
            # we delete existing adapters because we load externals anyways
            # if we don't do this we get an error message that some weights are
            # not used when loading adapter.
            model.delete_adapter(adapter)

        # resolve the adapter config
        adapter_config = AdapterConfig.load(
            adapter_args.adapter_config,
            non_linearity=adapter_args.adapter_non_linearity,
            reduction_factor=adapter_args.adapter_reduction_factor,
        )
        # load a pre-trained from Hub if specified
        if adapter_args.load_adapter:
            for adapter in adapter_args.load_adapter:
                model.load_adapter(
                    adapter,
                    config=adapter_config,
                    load_as=data_args.task_name[0] if len(adapter_args.load_adapter) <=1 else None,
                    with_head=adapter_args.fuse_adapters is None,
                )
        # otherwise, add a fresh adapter
        else:
            for task_name in data_args.task_name:
                model.add_adapter(task_name, config=adapter_config)

        # optionally load a pre-trained language adapter
        if adapter_args.load_lang_adapter:
            # resolve the language adapter config
            lang_adapter_config = AdapterConfig.load(
                adapter_args.lang_adapter_config,
                non_linearity=adapter_args.lang_adapter_non_linearity,
                reduction_factor=adapter_args.lang_adapter_reduction_factor,
            )
            # load the language adapter from Hub
            lang_adapter_names = [
                model.load_adapter(
                    lang_adapter,
                    config=lang_adapter_config,
                    load_as=lang,
                )
                for lang_adapter, lang in zip(adapter_args.load_lang_adapter, adapter_args.language)
            ]

        if adapter_args.fuse_adapters is not None:
            adapter_setup = [adapter_args.fuse_adapters]
            fusion_dir = os.path.join(training_args.output_dir, ','.join(adapter_setup[0]))
            fusion_config_path = os.path.join(fusion_dir, 'adapter_fusion_config.json')
            if os.path.exists(fusion_config_path):
                model.load_adapter_fusion(fusion_dir)
            else:
                model.add_adapter_fusion(adapter_setup[0], 'dynamic')
    else:
        if adapter_args.load_adapter or adapter_args.load_lang_adapter:
            raise ValueError(
                "Adapters can only be loaded in adapters training mode."
                "Use --train_adapter to enable adapter training"
            )

    if model_args.model_type == 'classifier' and adapter_args.fuse_adapters is None:
        # things get messy if we freeze with fusion. It is frozen anyways.
        # Unfreezing with fusion doesn't make too much sense probably.
        model.freeze_model(training_args.freeze_model_core)

    return lang_adapter_names, adapter_setup


def get_openprompts_model_name(model):
    try:
        return {
            BertForMaskedLM: 'bert',
            RobertaForMaskedLM: 'roberta',
        }[type(model)]
    except Exception:
        raise ValueError(f'Model type not supported: {type(model)}')


def setup_propting_if_needed(model, tokenizer, model_args):
    wrapper = None
    if model_args.model_type == 'prompt':
        from openprompt import plms

        model_name = get_openprompts_model_name(model)
        model_class = plms.get_model_class(model_name)
        specials_to_add = None
        if 'gpt' in model_name: # add pad token for gpt
            specials_to_add = ["<pad>"]
        wrapper = model_class.wrapper

        model, tokenizer = plms.add_special_tokens(
            model,
            tokenizer,
            specials_to_add=specials_to_add
        )

        if 'opt' in model_name:
            tokenizer.add_bos_token=False

    return model, tokenizer, wrapper


def get_models(model_args, data_args, training_args, adapter_args,
               dataset_metadata):
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only
    # one local process can concurrently download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if config.finetuning_task is None:
        config.finetuning_task = data_args.task_name
    elif type(config.finetuning_task) == str:
        config.finetuning_task = [config.finetuning_task] + data_args.task_name
    else:
        config.finetuning_task.extend(data_args.task_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # We use the AutoAdapterModel class here for better adapter support.
    if model_args.model_type == 'classifier':
        from transformers import AutoAdapterModel

        model = AutoAdapterModel.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif model_args.model_type == 'prompt':
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        raise ValueError(f'Unsupperted model_type: {model_args.model_type}')

    model, tokenizer, wrapper = setup_model(
        model,
        tokenizer,
        dataset_metadata,
        model_args,
        data_args,
        training_args,
        adapter_args
    )

    return model, tokenizer, config, wrapper, last_checkpoint


def preprocess_data(raw_datasets, model, tokenizer, config, data_args,
                    training_args, model_args, dataset_metadata):
    # Preprocessing the datasets
    # Again, we try to have some nice defaults but don't hesitate to tweak
    # to your use case.
    sentence_keys_dict = dict()
    for dataset_name, dataset in raw_datasets.items():
        non_label_column_names = [name for name in dataset["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        elif 'text' in non_label_column_names:
            sentence1_key, sentence2_key = 'text', None
        else:
            if len(non_label_column_names) >= 2:
                raise NotImplementedError('This is not supported at the moment!')
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

        sentence_keys_dict[dataset_name] = [sentence1_key, sentence2_key]

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence
        # length in each batch
        padding = False

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(sentence1_key, sentence2_key, examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        #  if label_to_id is not None and "label" in examples:
        #      result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        result['label'] = examples['label']
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = {
            dataset_name: dataset.map(
                partial(preprocess_function, *sentence_keys_dict[dataset_name]),
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Running tokenizer on dataset: {dataset_name}",
            )
            for dataset_name, dataset in raw_datasets.items()
        }

    train_dataset = None
    if training_args.do_train:
        train_dataset = dict()
        for (dataset_name, dataset), max_train_samples in zip(raw_datasets.items(), data_args.max_train_samples):
            if "train" not in dataset:
                raise ValueError(f"--do_train requires a train dataset in {dataset_name}")
            train_dataset_tmp = dataset["train"]
            train_dataset_tmp = reduce_dataset_if_needed(
                train_dataset_tmp,
                max_train_samples,
                data_args.data_selector_method_train,
                training_args.seed,
            )

            if data_args.train_sampling is not None:
                if data_args.train_sampling == 'balanced_over':
                    train_dataset_tmp = sampling.balanced_random_oversample(
                        dataset=train_dataset_tmp,
                        label_col='label',
                        random_state=training_args.seed,
                        load_from_cache_file=not data_args.overwrite_cache,
                    )
                    logger.info(f'Training dataset resampled with {data_args.train_sampling}')

            train_dataset[dataset_name] = train_dataset_tmp
        if data_args.train_sampling is not None:
            if data_args.train_sampling == 'global_balanced_over':
                assert model_args.model_type == 'prompt'
                train_dataset = sampling.global_balanced_random_oversample(
                    train_dataset,
                    ['id', 'text', 'label'],
                    label_map={
                        f'{dataset_name}-{dataset_config}': dataset_metadata[f'{dataset_name}-{dataset_config}'][3].get_label_map(pattern_id, train_dataset[f'{dataset_name}-{dataset_config}'].features['label'].names)
                        for dataset_name, dataset_config, pattern_id in zip(
                            data_args.dataset_name,
                            data_args.dataset_config_name,
                            data_args.prompt_ids
                        )
                    },
                    label_col='label',
                    random_state=training_args.seed,
                    load_from_cache_file=not data_args.overwrite_cache,
                )
                logger.info(f'Training dataset resampled with {data_args.train_sampling}')

    eval_dataset = None
    if training_args.do_eval:
        eval_dataset = dict()
        for (dataset_name, dataset), max_eval_samples in zip(raw_datasets.items(), data_args.max_eval_samples):
            if "validation" not in dataset:
                raise ValueError(f"--do_eval requires a validation dataset in {dataset_name}")

            eval_dataset_tmp = dataset["validation"]
            eval_dataset_tmp = reduce_dataset_if_needed(
                eval_dataset_tmp,
                max_eval_samples,
                data_args.data_selector_method_eval,
                training_args.seed,
            )

            eval_dataset[dataset_name] = eval_dataset_tmp

    predict_dataset = None
    if training_args.do_predict:
        predict_dataset = dict()
        for (dataset_name, dataset), max_predict_samples in zip(raw_datasets.items(), data_args.max_predict_samples):
            if "test" not in dataset:
                raise ValueError(f"--do_predict requires a test dataset in {dataset_name}")

            predict_dataset_tmp = dataset["test"]
            predict_dataset_tmp = reduce_dataset_if_needed(
                predict_dataset_tmp,
                max_predict_samples,
                data_args.data_selector_method_predict,
                training_args.seed,
            )

            predict_dataset[dataset_name] = predict_dataset_tmp

    # Log a few random samples from the training set:
    if training_args.do_train:
        for dataset_name, dataset in train_dataset.items():
            for index in random.sample(range(len(dataset)), 1):
                logger.info(f"Sample {index} of the training set in {dataset_name}: {dataset[index]}.")

    return train_dataset, eval_dataset, predict_dataset


def main():
    model_args, data_args, training_args, adapter_args = setup()

    raw_datasets, dataset_metadata = get_datasets(
        model_args,
        data_args,
        training_args
    )

    model, tokenizer, config, wrapper, last_checkpoint = get_models(
        model_args,
        data_args,
        training_args,
        adapter_args,
        dataset_metadata,
    )

    train_dataset, eval_dataset, predict_dataset = preprocess_data(
        raw_datasets,
        model,
        tokenizer,
        config,
        data_args,
        training_args,
        model_args,
        dataset_metadata
    )

    # Get the metric function
    metric = [
        load_metric("accuracy"),
        #  load_metric("f1"),
        #  load_metric("precision"),
        #  load_metric("recall"),
        load_metric(
            importlib.import_module(
                "src.metrics.f1_report.f1_report"
            ).__file__
        ),
    ]

    # You can define your custom compute_metrics function. It takes an
    # `EvalPrediction` object (a namedtuple with a predictions and label_ids
    # field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction, dataset_name: str):
        _, label_list, is_regression, _ = dataset_metadata[dataset_name]
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

        if is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            res = dict()
            for m in metric:
                if m.info.metric_name == 'f1_report':
                    scores = m.compute(
                        predictions=preds,
                        references=p.label_ids,
                        label_names=label_list,
                    )
                else:
                    scores = m.compute(
                        predictions=preds,
                        references=p.label_ids,
                    )

                for k, v in scores.items():
                    res[k] = v

            return res

    # Data collator will default to DataCollatorWithPadding when the tokenizer
    # is passed to Trainer, so we change it if we already did the padding.
    if wrapper is not None:
        data_collator = (
            {
                f'{dataset_name}-{dataset_config}': dataset_metadata[f'{dataset_name}-{dataset_config}'][3].get_pvp(pattern_id, tokenizer, model).template
                for dataset_name, dataset_config, pattern_id in zip(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    data_args.prompt_ids
                )
            },
            wrapper,
        )
    elif data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    callbacks = []
    if training_args.early_stopping_patience >= 0:
        callbacks.append(EarlyStoppingCallback(
            training_args.early_stopping_patience,
            training_args.early_stopping_threshold,
        ))

    if data_args.do_mlm:
        trainer_class = MLMMultitaskAdapterTrainer if adapter_args.train_adapter else MLMMultitaskTrainer
        mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
        )
        trainer_class = partial(
            trainer_class,
            mlm_data_collator=mlm_collator,
            mlm_weight=data_args.mlm_weight,
        )
    else:
        trainer_class = MultitaskAdapterTrainer if adapter_args.train_adapter else MultitaskTrainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        # XXX there seems to be an but in AdapterTrainerCallback
        # temporary fix is to change lines in transformers/adapters/trainer.py
        # 269: fusion_models = getattr(self.trainer.model.config, "adapter_fusion_models", [])
        # 273: self.trainer.model.load_adapter_fusion(fusion_dir)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples_tmp = sum([
            max_sample if max_sample is not None else len(dataset)
            for dataset, max_sample in zip(train_dataset, data_args.max_train_samples)
        ])
        metrics["train_samples"] = min(
            max_train_samples_tmp, len(train_dataset))

        # XXX There seems to be a bug in the adapter-transformers library.
        # When --train_adapter and --load_best_model_at_end is set the
        # classification head of the reloaded model stays on the CPU.
        # The command below moves the whole model to the same device.
        trainer.model = trainer.model.to(trainer.model.device)

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        for task_name, dataset_name, dataset_config_name, max_samples in zip(
            data_args.task_name,
            data_args.dataset_name,
            data_args.dataset_config_name,
            data_args.max_eval_samples,
        ):
            task = f'{dataset_name}-{dataset_config_name}'
            dataset = eval_dataset[task]

            metrics = trainer.evaluate(eval_dataset={task: dataset})
            metrics = {k: v for k, v in metrics.items() if 'MULTIAVG' not in k}

            max_eval_samples_tmp = (
                max_samples if max_samples is not None else len(dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples_tmp, len(dataset))

            trainer.log_metrics(f'validation_{task_name}', metrics)
            trainer.save_metrics(f'validation_{task_name}', metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        for task_name, dataset_name, dataset_config_name in zip(
            data_args.task_name,
            data_args.dataset_name,
            data_args.dataset_config_name,
        ):
            task = f'{dataset_name}-{dataset_config_name}'
            dataset = predict_dataset[task]

            _, label_list, is_regression, _ = dataset_metadata[task]
            predict_res = trainer.predict({task: dataset}, metric_key_prefix="predict")
            predictions = predict_res.predictions[task]
            metrics = predict_res.metrics
            metrics = {k: v for k, v in metrics.items() if 'MULTIAVG' not in k}

            metrics["predict_samples"] = len(dataset)
            trainer.log_metrics(task_name, metrics)
            trainer.save_metrics(task_name, metrics)

            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_predict_file = os.path.join(training_args.output_dir, f"{task_name}_predictions.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task_name} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

    if training_args.do_visualize:
        from src.visualization import get_averaged_fusion_attentions, visualize_and_save

        logger.info("*** Generating figures ***")

        assert adapter_args.fuse_adapters

        model.eval()

        fusion_attentions = dict()

        for task_name, dataset_name, dataset_config_name in zip(
            data_args.task_name,
            data_args.dataset_name,
            data_args.dataset_config_name,
        ):
            task = f'{dataset_name}-{dataset_config_name}'
            dataset = eval_dataset[task]

            _, label_list, is_regression, _ = dataset_metadata[task]

            attentions = get_averaged_fusion_attentions(
                trainer,
                ','.join(adapter_args.fuse_adapters),
                eval_dataset={task: dataset}
            )
            fusion_attentions[task_name] = attentions

            visualize_and_save(
                attentions,
                os.path.join(
                    training_args.output_dir,
                    f'{task_name}_fusion_per_layer_weights.pdf'
                ),
                xticklabels=adapter_args.fuse_adapters,
                yticklabels=[f'layer_{i}' for i in range(attentions.shape[0])],
            )

        aggregated_fusion_attentions = np.array([
            fusion_attentions[task_name].mean(axis=0)
            for task_name in data_args.task_name
        ])
        visualize_and_save(
            aggregated_fusion_attentions,
            os.path.join(
                training_args.output_dir,
                'aggregated_fusion_weights.pdf'
            ),
            xticklabels=adapter_args.fuse_adapters,
            yticklabels=data_args.task_name,
        )

    if training_args.do_debug_predictions:
        assert model_args.model_type == 'prompt'
        from src.visualization import get_prediction_logits, lineplot_and_save

        np_descriptors = {
            'mean': np.mean,
            'median': np.median,
            'minimum': np.min,
            'maximum': np.max,
            'range': np.ptp,
            'variance': np.var,
            'STD': np.std,
        }
        token_list = tokenizer.convert_ids_to_tokens(range(len(tokenizer)))

        for task_name, dataset_name, dataset_config_name in zip(
            data_args.task_name,
            data_args.dataset_name,
            data_args.dataset_config_name,
        ):
            task = f'{dataset_name}-{dataset_config_name}'
            dataset = eval_dataset[task]
            _, label_list, is_regression, _ = dataset_metadata[task]
            res_dict = {}
            pooled_logits = list()
            pooled_label_logits = list()
            num_per_label = [0] * len(label_list)

            for input_ids_batch, outputs_batch, label_logits_batch, gold_labels_batch in get_prediction_logits(
                trainer,
                eval_dataset={task: dataset}
            ):
                for token_ids, output_logits, label_logits, gold_label_idx in zip(input_ids_batch, outputs_batch, label_logits_batch, gold_labels_batch):
                    output_logits = softmax(output_logits).numpy().tolist()
                    label_logits = softmax(label_logits).numpy().tolist()
                    gold_label = label_list[gold_label_idx]
                    num_per_label[gold_label_idx] += 1
                    last_idx = 0
                    for idx in range(len(token_ids)):
                        if token_ids[idx] == tokenizer.pad_token_id:
                            break
                        last_idx = idx

                    token_ids = token_ids[:last_idx]

                    print(tokenizer.convert_tokens_to_string(
                        tokenizer.convert_ids_to_tokens(token_ids))
                    )

                    for k, v in np_descriptors.items():
                        val = v(output_logits)
                        print(f'{k}: {val}')
                        res_dict.setdefault(f'output_logits_{k}', list()).append(val)

                    pooled_logits.append(output_logits)
                    pooled_label_logits.append(label_logits)

                    print(f'Gold label: {gold_label}')
                    prediction = label_list[np.argmax(label_logits)]
                    print(f'Predicted label: {prediction}')
                    print('Label logits:')
                    print_label_logits(label_list, label_logits)
                    print()

            print('Dataset statistics:')
            for k, v in res_dict.items():
                print(f'{k}: {np.mean(v)}')

            print('Averaged label logits:')
            print_label_logits(label_list, np.mean(pooled_label_logits, axis=0).tolist())
            print('Averaged & normalized label logits:')
            print_label_logits(label_list, (np.mean(pooled_label_logits, axis=0)/num_per_label).tolist())
            print('Number per label:')
            print_label_logits(label_list, num_per_label)

            # TODO topK tokens
            k = 20
            avg_logits = np.mean(pooled_logits, axis=0)
            top_k_token_ids = np.argsort(avg_logits)[-k:][::-1]
            print(f'Tok-{k} predicted tokens:')
            print_label_logits(np.array(token_list)[top_k_token_ids].tolist(), np.array(avg_logits)[top_k_token_ids].tolist() )

            lineplot_and_save(
                np.mean(pooled_logits, axis=0),
                os.path.join(
                    training_args.output_dir,
                    f'{task}_pooled_output_logits.pdf'
                ),

            )
            lineplot_and_save(
                np.mean(pooled_label_logits, axis=0),
                os.path.join(
                    training_args.output_dir,
                    f'{task}_pooled_label_logits.pdf'
                ),

            )


def print_label_logits(labels, logits):
    print(', '.join(f'{label}={logit:4f}' for label, logit in zip(labels, logits)))


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
