# Make sure to run this code somewhere, e.g., import it in __init__.py of this
# module

from src.data.prompting import (
    PVP_DICT,
    DatasetPVPs,
    ManualTemplateFactory,
    MixedTemplateFactory,
    ManualVerbalizerFactory
)

PVP_DICT['hasoc19'] = {
    'en-binary': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
            MixedTemplateFactory('{"soft": "hasoc19"} {"soft": "en-binary"} {"placeholder":"text_a"} It was {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                'NOT': ["neutral"],
                'HOF': ["abusive"]
            }),
            ManualVerbalizerFactory({
                'NOT': ["neutral", "innocent", "nice", "good"],
                'HOF': ["abusive", "offensive", "hate", "bad"]
            }),
        ],
        pvps={
            0: (0, 0),
            1: (0, 1),
            2: (1, 0),
            3: (1, 1),
        },
    ),
    'en-fine_grained': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
            MixedTemplateFactory('{"soft": "hasoc19"} {"soft": "en-fine_grained"} {"placeholder":"text_a"} It was {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                'HATE': ["hate"],
                'OFFN': ["offensive"],
                'PRFN': ["profane"],
            }),
            ManualVerbalizerFactory({
                'HATE': ["hate", "hostile"],
                'OFFN': ["offensive", "rude"],
                'PRFN': ["profane", "obscene", "vulgar", "dirty"],
            }),
        ],
        pvps={
            0: (0, 0),
            1: (0, 1),
            2: (1, 0),
            3: (1, 1),
        },
    ),
    'en-targeted': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                'UNT': ['general'],
                'TIN': ['targeted'],
            }),
            ManualVerbalizerFactory({
                'UNT': ['general', 'normal', 'neutral'],
                'TIN': ['targeted', 'individual', 'group'],
            }),
        ],
        pvps={
            0: (0, 0),
            1: (0, 1),
        },
    ),
    'hi-targeted': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
            MixedTemplateFactory('{"soft": "hasoc19"} {"soft": "hi-targeted"} {"placeholder":"text_a"} It was {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                'UNT': ['general'],
                'TIN': ['targeted'],
            }),
            ManualVerbalizerFactory({
                'UNT': ['general', 'normal', 'neutral'],
                'TIN': ['targeted', 'focused', 'aimed'],
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
