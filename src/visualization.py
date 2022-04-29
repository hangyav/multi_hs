import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from transformers import BertAdapterModel
from transformers.utils import logging


logger = logging.get_logger(__name__)


def get_averaged_fusion_attentions_after_forward(model_encoder, fusion_name):
    return np.array([
        layer.output.adapter_fusion_layer[fusion_name].recent_attention.mean(axis=(0, 1))
        for layer in model_encoder.layer
    ])


def get_averaged_fusion_attentions(trainer, fusion_name, model_encoder=None,
                                   eval_dataset=None):
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
