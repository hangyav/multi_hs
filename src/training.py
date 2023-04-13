# based on: https://medium.com/@shahrukhx01/multi-task-learning-with-transformers-part-1-multi-prediction-heads-b7001cf014bf

import os
from typing import List, Optional
import numpy as np

from transformers import Trainer, PreTrainedModel
from transformers.trainer_utils import (
    EvalLoopOutput,
    EvalPrediction,
    has_length,
    denumpify_detensorize,
)
from transformers.trainer_pt_utils import (
    find_batch_size,
    nested_concat,
    nested_numpify,
    nested_truncate,
    IterableDatasetShard,
)
from transformers.file_utils import is_torch_tpu_available, WEIGHTS_NAME
from transformers.modeling_utils import unwrap_model
from transformers.deepspeed import deepspeed_init
from transformers.utils import logging
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.distributed import DistributedSampler

from src.data.multitask import MultitaskDataloader, DataLoaderWithTaskname

try:
    from openprompt import PromptDataLoader
    from openprompt.data_utils import InputExample
except Exception:
    pass

try:
    from transformers import AdapterTrainer
except Exception:
    AdapterTrainer = Trainer

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl


logger = logging.get_logger(__name__)


class MultitaskEvalLoopOutput:

    def __init__(self, evalloop_outputs):
        self.metrics = dict()
        self.per_task_outputs = evalloop_outputs
        self.num_samples = 0
        self.label_ids = dict()
        self.predictions = dict()

        tmp_metrics = dict()
        for dataset_name, output in evalloop_outputs.items():
            self.num_samples += output.num_samples
            self.predictions[dataset_name] = output.predictions
            self.label_ids[dataset_name] = output.label_ids
            for metric, value in output.metrics.items():
                tmp_metrics.setdefault(metric, list()).append(value)
                self.metrics[f'{dataset_name}_{metric}'] = value

        for metric, values in tmp_metrics.items():
            self.metrics[f'{metric}_MULTIAVG'] = np.mean(values)


class MultitaskTrainerMixin(object):

    def _get_dataloader_for_classification(self, dataset, batch_size, sampler):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
        )

    @staticmethod
    def _convert_dataset_to_openprompt(dataset):
        return [
            InputExample(
                guid=item['id'] if 'id' in item else idx,
                text_a=item['text'],
                label=item['label'],
            )
            for idx, item in enumerate(dataset)
        ]

    def _get_dataloader_for_prompt(self, dataset, task_name, batch_size,
                                   shuffle):
        return PromptDataLoader(
            dataset=MultitaskTrainerMixin._convert_dataset_to_openprompt(dataset),
            tokenizer=self.tokenizer,
            template=self.data_collator[0][task_name],
            tokenizer_wrapper_class=self.data_collator[1],
            shuffle=shuffle,
            batch_size=batch_size,
        )

    def _get_dataloader(self, dataset, task_name, batch_size, sampler=None):
        if type(self.data_collator) == tuple:
            return self._get_dataloader_for_prompt(
                dataset,
                task_name,
                batch_size,
                sampler is not None,
            )
        return self._get_dataloader_for_classification(dataset, batch_size,
                                                       sampler)

    def _get_single_train_dataloader(self, task_name, train_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        train_sampler = (
            RandomSampler(train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(train_dataset)
        )

        data_loader = DataLoaderWithTaskname(
            task_name=task_name,
            data_loader=self._get_dataloader(
                train_dataset,
                task_name,
                self.args.train_batch_size,
                train_sampler,
            ),
        )
        return data_loader

    def get_train_dataloader(self):
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each
        task Dataloader
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        return MultitaskDataloader(
            {
                task_name: self._get_single_train_dataloader(task_name, task_dataset)
                for task_name, task_dataset in self.train_dataset.items()
            }
        )

    def _get_single_eval_dataloader(self, task_name, train_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        data_loader = DataLoaderWithTaskname(
            task_name=task_name,
            data_loader=self._get_dataloader(train_dataset, task_name,
                                             self.args.eval_batch_size)
        )
        return data_loader

    def get_eval_dataloader(self, eval_dataset=None):
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each
        task Dataloader
        """
        if eval_dataset is not None:
            if type(eval_dataset) != dict:
                raise ValueError('Only named datasets in a dictionary are supported!')
        elif self.eval_dataset is None:
            raise ValueError("Trainer: evaluation during training requires an eval_dataset.")
        else:
            eval_dataset = self.eval_dataset

        return MultitaskDataloader(
            {
                task_name: self._get_single_eval_dataloader(task_name, task_dataset)
                for task_name, task_dataset in eval_dataset.items()
            }
        )

    def get_test_dataloader(self, test_dataset):
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each
        task Dataloader
        """
        if type(test_dataset) != dict:
            raise ValueError('Only named datasets in a dictionary are supported!')

        return MultitaskDataloader(
            {
                task_name: self._get_single_eval_dataloader(task_name, task_dataset)
                for task_name, task_dataset in test_dataset.items()
            }
        )

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        outputs = {
            dataset_name: self._evaluation_loop(
                dl,
                description,
                dataset_name,
                prediction_loss_only,
                ignore_keys,
                metric_key_prefix,
            )
            for dataset_name, dl in dataloader.dataloader_dict.items()
        }

        return MultitaskEvalLoopOutput(outputs)

    def _evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        dataset_name: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        Minor modifications to the implementation in Trainer:
            * logging info of dataset name
            * and using dataset name for the metrics method
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} of {dataset_name} *****")
        if has_length(dataloader.dataset):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(
                EvalPrediction(predictions=all_preds, label_ids=all_labels),
                dataset_name,
            )
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)


class CustomLossTrainerMixin(object):
    def __init__(self, loss_fct=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fct = loss_fct
        if self.loss_fct is not None:
            self.loss_fct = self.loss_fct.to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)

        if 'label' in inputs:
            outputs = self._alter_loss(outputs, inputs['label'])

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def _alter_loss(self, model_outputs, labels):
        if self.loss_fct is None:
            return model_outputs

        if isinstance(model_outputs, dict):
            logits = model_outputs["logits"]
        else:
            logits = model_outputs[1]

        # TODO currently mulitdataset setups is not supported
        # task_name should be used from the input to select loss_fct from
        # a dict in the Multitask trainers
        loss = self.loss_fct(logits, labels)

        if isinstance(model_outputs, dict):
            model_outputs["loss"] = loss
        else:
            model_outputs = (loss,) + model_outputs[1:]

        return model_outputs


class MultitaskTrainer(CustomLossTrainerMixin, MultitaskTrainerMixin, Trainer):
    pass


class MultitaskAdapterTrainer(CustomLossTrainerMixin, MultitaskTrainerMixin, AdapterTrainer):

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                if state_dict is None:
                    state_dict = self.model.state_dict()
                unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_all_adapters(output_dir)
            if self.train_adapter_fusion:
                self.model.save_all_adapter_fusions(output_dir)
            if hasattr(self.model, "heads"):
                self.model.save_all_heads(output_dir)
            if not self.args.freeze_model_core:
                self.model.base_model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


class MLMTrainerMixin(object):

    def __init__(self, mlm_data_collator, mlm_weight=1.0, *args, **kwargs):
        self.mlm_data_collator = mlm_data_collator
        self.mlm_weight = mlm_weight
        assert 'ForMaskedLM' in kwargs['model'].__class__.__name__, 'Currently only MLM models are supported!'
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        res = super().compute_loss(model, inputs, return_outputs=return_outputs)
        if return_outputs:
            loss, outputs = res
        else:
            loss = res
            outputs = None

        # MLM loss
        # XXX This should probably go the the train function of Trainer but
        # this is less invasive
        mlm_inputs = self.mlm_data_collator([
            item for item in inputs['input_ids'].to('cpu')
        ])
        mlm_inputs = self._prepare_inputs(mlm_inputs)

        mlm_outputs = model(**mlm_inputs)
        mlm_loss = mlm_outputs["loss"] if isinstance(mlm_outputs, dict) else mlm_outputs[0]

        loss += mlm_loss * self.mlm_weight

        return (loss, outputs) if return_outputs else loss


class MLMMultitaskTrainer(MLMTrainerMixin, MultitaskTrainer):
    pass


class MLMMultitaskAdapterTrainer(MLMTrainerMixin, MultitaskAdapterTrainer):
    pass
