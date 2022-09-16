# Make sure to run this code somewhere, e.g., import it in __init__.py of this
# module

from src.data.prompting import (
    PVP_DICT,
    DatasetPVPs,
    ManualTemplateFactory,
    ManualVerbalizerFactory
)

PVP_DICT['mlma'] = {
    'en-sentiment_rep': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                'abusive': ['abusive'],
                'hateful': ['hate'],
                'offensive': ['offensive'],
                'disrespectful': ['disrespectful'],
                'fearful': ['fearful'],
                'normal': ['neutral'],
            }),
        ],
        pvps={
            0: (0, 0),
        },
    ),
    'en-directness': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                'direct': ['direct'],
                'indirect': ['indirect'],
            }),
        ],
        pvps={
            0: (0, 0),
        },
    ),
    'en-target': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was because of {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                'origin': ['origin'],
                'gender': ['gender'],
                'sexual_orientation': ['sexuality'],
                'religion': ['religion'],
                'disability': ['disability'],
                'other': ['other'],
            }),
        ],
        pvps={
            0: (0, 0),
        },
    ),
    'en-group': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was targeted at {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                'individual': ['individual'],
                'women': ['women'],
                'special_needs': ['special'],
                'african_descent': ['Africans'],
                'other': ['other'],
            }),
        ],
        pvps={
            0: (0, 0),
        },
    ),
}
