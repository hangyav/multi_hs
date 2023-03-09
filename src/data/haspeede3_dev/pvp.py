# Make sure to run this code somewhere, e.g., import it in __init__.py of this
# module

from src.data.prompting import (
    PVP_DICT,
    DatasetPVPs,
    ManualTemplateFactory,
    ManualVerbalizerFactory,
    MixedTemplateFactory,
)

PVP_DICT['haspeede3_dev'] = {
    'politics_text': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
            ManualTemplateFactory('{"placeholder":"text_a"} Era {"mask"}'),
            MixedTemplateFactory('{"soft": "What type of abusive content does the text have?"} {"placeholder":"text_a"} {"mask"}'),
            MixedTemplateFactory('{"placeholder":"text_a"} {"soft": "What type of abusive content does the text have?"} {"mask"}'),

            ManualTemplateFactory('{"placeholder":"text_a"} Is this hate speech? {"mask"}'),
            ManualTemplateFactory('{"placeholder":"text_a"} Is this either hate speech or abusive? {"mask"}'),
            ManualTemplateFactory('Is this hate speech? {"placeholder":"text_a"} {"mask"}'),
            ManualTemplateFactory('Is this either hate speech or abusive? {"placeholder":"text_a"} {"mask"}'),
            MixedTemplateFactory('{"soft": "Is this either hate speech or abusive?"} {"placeholder":"text_a"} {"mask"}'),
            MixedTemplateFactory('{"soft": "Is this hate speech?"} {"placeholder":"text_a"} {"mask"}'),

            ManualTemplateFactory('Questo è un discorso di odio? {"placeholder":"text_a"} {"mask"}'),
            MixedTemplateFactory('{"soft": "Questo è un discorso di odio?"} {"placeholder":"text_a"} {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                "noHate": ["neutral"],
                "Hate": ["hate"],
            }),
            ManualVerbalizerFactory({
                "noHate": ["neutro"],
                "Hate": ["odio"],
            }),
            ManualVerbalizerFactory({
                "noHate": ["No"],
                "Hate": ["Yes"],
            }),
            ManualVerbalizerFactory({
                "noHate": ["No"],
                "Hate": ["Sì"],
            }),
        ],
        pvps={
            0: (0, 0),
            1: (1, 1),
            2: (2, 0),
            3: (3, 0),

            4: (4, 2),
            5: (5, 2),
            6: (6, 2),
            7: (7, 2),
            8: (8, 2),
            9: (9, 2),

            10: (10, 3),
            11: (11, 3),
        },
    ),
}
