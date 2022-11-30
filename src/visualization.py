import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from transformers.utils import logging


logger = logging.get_logger(__name__)


def get_averaged_fusion_attentions_after_forward(model_encoder, fusion_name):
    return np.array([
        layer.output.adapter_fusion_layer[fusion_name].recent_attention.mean(axis=(0, 1))
        for layer in model_encoder.layer
    ])


def get_averaged_fusion_attentions(trainer, fusion_name, model_encoder=None,
                                   eval_dataset=None):
    from transformers import BertAdapterModel
    model = trainer.model

    if model.training:
        model.eval()

    if model_encoder is None:
        if type(model) == BertAdapterModel:
            model_encoder = model.bert.encoder
        else:
            raise NotImplementedError(f'{type(model)} is not yet supported!')

    res = list()
    for batch in trainer.get_eval_dataloader(eval_dataset=eval_dataset):
        batch = trainer._prepare_inputs(batch)

        model(**batch)
        res.append(get_averaged_fusion_attentions_after_forward(model.bert.encoder, fusion_name))

    return np.array(res).mean(axis=0)


def visualize_and_save(data, output_path, linewidth=0.5, xticklabels=None,
                       yticklabels=None, format='pdf', bbox_inches='tight'):
    sns.heatmap(
        data,
        linewidth=linewidth,
        square=True,
        xticklabels=xticklabels,
        yticklabels=yticklabels
    )
    logger.info(f'Saving figure to: {output_path}')
    plt.savefig(
        output_path,
        format=format,
        bbox_inches=bbox_inches,
    )


def get_prediction_logits(trainer, eval_dataset=None):
    model = trainer.model.forward.__self__.prompt_models_dict[list(eval_dataset.keys())[0]]
    if model.training:
        model.eval()
    prompt_model = model.prompt_model

    with torch.no_grad():
        for batch in trainer.get_eval_dataloader(eval_dataset=eval_dataset):
            batch = trainer._prepare_inputs(batch)

            # From: openprompt/pipeline_base.py:295
            outputs = prompt_model(batch)
            outputs = model.verbalizer.gather_outputs(outputs)
            if isinstance(outputs, tuple):
                outputs_at_mask = [model.extract_at_mask(output, batch) for output in outputs]
            else:
                outputs_at_mask = model.extract_at_mask(outputs, batch)
            label_words_logits = model.verbalizer.process_outputs(outputs_at_mask, batch=batch)

            yield batch['input_ids'].to('cpu'), outputs_at_mask.to('cpu'), label_words_logits.to('cpu')


def lineplot_and_save(data, output_path, x=None, y=None, format='pdf',
                      bbox_inches='tight'):
    sns.lineplot(
        data=data,
        x=x,
        y=y,
    )
    logger.info(f'Saving figure to: {output_path}')
    plt.savefig(
        output_path,
        format=format,
        bbox_inches=bbox_inches,
    )
