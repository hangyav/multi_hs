
from dataclasses import dataclass
from typing import List

from openprompt import Template, Verbalizer
from openprompt.prompts import ManualTemplate, ManualVerbalizer


@dataclass
class PVP():
    template: Template
    verbalizer: Verbalizer


@dataclass
class DatasetPVPs():
    prompt_templates: List
    prompt_verbalizers: List
    pvps: List  # list of (template_idx, veralizer_idx)
    classes = None

    def get_pvp(self, id, tokenizer):
        template = self.prompt_templates[self.pvps[id][0]]
        template = ManualTemplate(tokenizer, template)

        verbalizer = self.prompt_verbalizers[self.pvps[id][1]]
        verbalizer = ManualVerbalizer(
            tokenizer,
            list(range(len(self.classes))),
            label_words={
                self.classes.index(k): v
                for k, v in verbalizer.items()
            }
        )

        return PVP(template=template, verbalizer=verbalizer)


PVP_DICT = dict()

PVP_DICT['hate_speech18'] = {
    'binary': DatasetPVPs(
        prompt_templates=[
            '{"placeholder":"text_a"} It was {"mask"}'
        ],
        prompt_verbalizers=[
            {
                'noHate': ["neutral"],
                'hate': ["hate"]
            },
        ],
        pvps={
            0: (0, 0),
        },
    ),
}

PVP_DICT['ami18'] = {
    'en-misogyny': DatasetPVPs(
        prompt_templates=[
            '{"placeholder":"text_a"} It was {"mask"}',
            '{"placeholder":"text_a"} Was it sexist? {"mask"}',
        ],
        prompt_verbalizers=[
            {
                'non-misogyny': ["neutral"],
                'misogyny': ["sexist"]
            },
            {
                'non-misogyny': ["No"],
                'misogyny': ["Yes"]
            },
        ],
        pvps={
            0: (0, 0),
            1: (1, 1),
        },
    ),
}


def get_pvp(dataset):
    pvps = get_pvp_by_name_and_config(
        dataset.builder_name,
        dataset.config_name,
    )
    pvps.classes = dataset.features['label'].names

    return pvps


def get_pvp_by_name_and_config(dataset_name, dataset_config):
    return PVP_DICT[dataset_name][dataset_config]
