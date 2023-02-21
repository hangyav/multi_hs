# Make sure to run this code somewhere, e.g., import it in __init__.py of this
# module

from src.data.prompting import (
    PVP_DICT,
    DatasetPVPs,
    ManualTemplateFactory,
    ManualVerbalizerFactory
)

PVP_DICT['haspeede3_dev'] = {
    'politics_text': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
            ManualTemplateFactory('{"placeholder":"text_a"} Era {"mask"}'),
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
        },
    ),
}
