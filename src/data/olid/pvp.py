# Make sure to run this code somewhere, e.g., import it in __init__.py of this
# module

from src.data.prompting import (
    PVP_DICT,
    DatasetPVPs,
    ManualTemplateFactory,
    MixedTemplateFactory,
    ManualVerbalizerFactory
)

PVP_DICT['olid'] = {
    'offensive': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
            MixedTemplateFactory('{"soft": "OLID"} {"soft": "offensive"} {"placeholder":"text_a"} It was {"mask"}'),
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
            ManualVerbalizerFactory({
                'NOT': ["neutral"],
                'OFF': ["hate"]
            }),
        ],
        pvps={
            0: (0, 0),
            1: (0, 1),
            2: (1, 0),
            3: (1, 1),
            4: (0, 2),
        },
    ),
    'targeted': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                'UNT': ["general"],
                'TIN': ["targeted"]
            }),
        ],
        pvps={
            0: (0, 0),
        },
    ),
    'target': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was targeted at {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                'IND': ["individual"],
                'GRP': ["group"],
                'OTH': ["other"],
            }),
        ],
        pvps={
            0: (0, 0),
        },
    ),
}
