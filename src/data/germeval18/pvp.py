# Make sure to run this code somewhere, e.g., import it in __init__.py of this
# module

from src.data.prompting import (
    PVP_DICT,
    DatasetPVPs,
    ManualTemplateFactory,
    MixedTemplateFactory,
    ManualVerbalizerFactory
)

PVP_DICT['germeval18'] = {
    'binary': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                'OTHER': ["neutral"],
                'OFFENSE': ["offensive"]
            }),
            ManualVerbalizerFactory({
                'OTHER': ["neutral"],
                'OFFENSE': ["hate"]
            }),
        ],
        pvps={
            0: (0, 0),
            1: (0, 1),
        },
    ),
    'fine_grained': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
            MixedTemplateFactory('{"soft": "What type of abusive content does the text have?"} {"placeholder":"text_a"} {"mask"}'),
            MixedTemplateFactory('{"placeholder":"text_a"} {"soft": "What type of abusive content does the text have?"} {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                'OTHER': ["neutral"],
                'PROFANITY': ["profane"],
                'INSULT': ["insulting"],
                'ABUSE': ["abusive"],
            }),
        ],
        pvps={
            0: (0, 0),
            1: (1, 0),
            2: (2, 0),
        },
    ),
}
