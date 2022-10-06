# Make sure to run this code somewhere, e.g., import it in __init__.py of this
# module

from src.data.prompting import (
    PVP_DICT,
    DatasetPVPs,
    ManualTemplateFactory,
    MixedTemplateFactory,
    ManualVerbalizerFactory
)

PVP_DICT['hateval19'] = {
    'en-hate': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
            MixedTemplateFactory('{"soft": "hateval19"} {"soft": "en-hate"} {"placeholder":"text_a"} It was {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                'hate': ["hate"],
                'nohate': ["neutral"]
            }),
            ManualVerbalizerFactory({
                'hate': ["hate", "offensive", "abusive", "bad"],
                'nohate': ["abusive", "offensive", "hate", "bad"]
            }),
        ],
        pvps={
            0: (0, 0),
            1: (0, 1),
            2: (1, 0),
            3: (1, 1),
        },
    ),
    'en-aggressive': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
            MixedTemplateFactory('{"soft": "hateval19"} {"soft": "en-aggressve"} {"placeholder":"text_a"} It was {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                'aggressive': ["aggressive"],
                'nonaggressive': ["neutral"]
            }),
            ManualVerbalizerFactory({
                'aggressive': ["aggressive", "threatening", "hostile"],
                'nonaggressive': ["neutral", "innocent", "nice", "good"]
            }),
        ],
        pvps={
            0: (0, 0),
            1: (0, 1),
            2: (1, 0),
            3: (1, 1),
        },
    ),
    'en-target': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was targeted {"mask"}'),
            MixedTemplateFactory('{"soft": "hateval19"} {"soft": "en-target"} {"placeholder":"text_a"} It was targeted at {"mask"}'),
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                'individual': ["individual"],
                'generic': ["group"]
            }),
            ManualVerbalizerFactory({
                'individual': ["individual", "person", "man", "woman"],
                'generic': ["group", "women", "immigrants", "people"]
            }),
            ManualVerbalizerFactory({
                'individual': ["active"],
                'generic': ["passive"]
            }),
        ],
        pvps={
            0: (0, 0),
            1: (0, 1),
            2: (1, 0),
            3: (1, 1),
            4: (2, 2),
        },
    ),
    'es-hate': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
            MixedTemplateFactory('{"soft": "hateval19"} {"soft": "es-hate"} {"placeholder":"text_a"} It was {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                'hate': ["hate"],
                'nohate': ["neutral"]
            }),
            ManualVerbalizerFactory({
                'hate': ["hate", "offensive", "abusive", "bad"],
                'nohate': ["abusive", "offensive", "hate", "bad"]
            }),
        ],
        pvps={
            0: (0, 0),
            1: (0, 1),
            2: (1, 0),
            3: (1, 1),
        },
    ),
    'es-aggressve': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
            MixedTemplateFactory('{"soft": "hateval19"} {"soft": "es-aggressve"} {"placeholder":"text_a"} It was {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                'aggressive': ["aggressive"],
                'nonaggressive': ["neutral"]
            }),
            ManualVerbalizerFactory({
                'aggressive': ["aggressive", "threatening", "hostile"],
                'nonaggressive': ["neutral", "innocent", "nice", "good"]
            }),
        ],
        pvps={
            0: (0, 0),
            1: (0, 1),
            2: (1, 0),
            3: (1, 1),
        },
    ),
    'es-target': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was targeted {"mask"}'),
            MixedTemplateFactory('{"soft": "hateval19"} {"soft": "es-target"} {"placeholder":"text_a"} It was targeted at {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                'individual': ["individual"],
                'generic': ["group"]
            }),
            ManualVerbalizerFactory({
                'individual': ["individual", "person", "man", "woman"],
                'generic': ["group", "women", "immigrants", "people"]
            }),
        ],
        pvps={
            0: (0, 0),
            1: (0, 1),
            2: (1, 0),
            3: (1, 1),
        },
    ),
}
