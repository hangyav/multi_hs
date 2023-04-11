# Make sure to run this code somewhere, e.g., import it in __init__.py of this
# module

from src.data.prompting import (
    PVP_DICT,
    DatasetPVPs,
    ManualTemplateFactory,
    ManualVerbalizerFactory
)


PVP_DICT['ihsc'] = {
    'hate': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
            ManualTemplateFactory('Questo è un discorso di odio? {"placeholder":"text_a"} {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                'no': ["neutral"],
                'yes': ["hate"]
            }),
            ManualVerbalizerFactory({
                "no": ["No"],
                "yes": ["Sì"],
            }),
        ],
        pvps={
            0: (0, 0),
            1: (1, 1),
        },
    ),
    'stereotype': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
            ManualTemplateFactory('Questo è stereotipato? {"placeholder":"text_a"} {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                'no': ["neutral"],
                'yes': ["stereotypical"]
            }),
            ManualVerbalizerFactory({
                "no": ["No"],
                "yes": ["Sì"],
            }),
        ],
        pvps={
            0: (0, 0),
            1: (1, 1),
        },
    ),
}
