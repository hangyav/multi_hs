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
                'noHate': ["neutral"],
                'Hate': ["hate"]
            }),
            ManualVerbalizerFactory({
                "noHate": ["No"],
                "Hate": ["Sì"],
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
                'noStereotype': ["neutral"],
                'Stereotype': ["stereotypical"]
            }),
            ManualVerbalizerFactory({
                "noStereotype": ["No"],
                "Stereotype": ["Sì"],
            }),
        ],
        pvps={
            0: (0, 0),
            1: (1, 1),
        },
    ),
}
