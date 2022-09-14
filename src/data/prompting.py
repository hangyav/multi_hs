
from dataclasses import dataclass
from typing import List

from openprompt import Template, Verbalizer
from openprompt.prompts import ManualTemplate, ManualVerbalizer, MixedTemplate


class TemplateFactory():

    def __init__(self, template):
        self.template = template

    def get_template(self, *args, **kwargs):
        raise NotImplementedError()


class ManualTemplateFactory(TemplateFactory):

    def get_template(self, tokenizer, **kwargs):
        return ManualTemplate(tokenizer, self.template)


class MixedTemplateFactory(TemplateFactory):

    def get_template(self, tokenizer, model, **kwargs):
        return MixedTemplate(model, tokenizer, self.template)


class VerbalizerFactory():

    def __init__(self, verbs):
        self.verbs = verbs

    def get_verbalizer(self, *args, **kwargs):
        raise NotImplementedError()


class ManualVerbalizerFactory(VerbalizerFactory):

    def get_verbalizer(self, tokenizer, classes, **kwargs):
        return ManualVerbalizer(
            tokenizer,
            list(range(len(classes))),
            label_words={
                classes.index(k): v
                for k, v in self.verbs.items()
            }
        )


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

    def get_pvp(self, id, tokenizer, model):
        template = self.prompt_templates[self.pvps[id][0]].get_template(
            tokenizer=tokenizer,
            model=model,
        )
        verbalizer = self.prompt_verbalizers[self.pvps[id][1]].get_verbalizer(
            tokenizer=tokenizer,
            classes=self.classes,
        )

        return PVP(template=template, verbalizer=verbalizer)


PVP_DICT = dict()


def get_pvp(dataset):
    pvps = get_pvp_by_name_and_config(
        dataset.builder_name,
        dataset.config_name,
    )
    if pvps.classes is None:
        pvps.classes = dataset.features['label'].names

    return pvps


def get_pvp_by_name_and_config(dataset_name, dataset_config):
    return PVP_DICT[dataset_name][dataset_config]
