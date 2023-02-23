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
        ],
        pvps={
            0: (0, 0),
            1: (1, 1),
            2: (2, 0),
            3: (3, 0),
        },
    ),
}
