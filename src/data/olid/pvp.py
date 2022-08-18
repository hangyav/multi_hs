# Make sure to run this code somewhere, e.g., import it in __init__.py of this
# module

from src.data.prompting import (
    PVP_DICT,
    DatasetPVPs,
    ManualTemplateFactory,
    ManualVerbalizerFactory
)

PVP_DICT['olid'] = {
    'offensive': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                'NOT': ["neutral"],
                'OFF': ["offensive"]
            }),
            ManualVerbalizerFactory({
                'NOT': ["neutral", "innocent", "nice", "good"],
                'OFF': ["offensive", "abusive", "hate", "bad"]
            }),
        ],
        pvps={
            0: (0, 0),
            1: (0, 1),
        },
    ),
}
