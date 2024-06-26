# Make sure to run this code somewhere, e.g., import it in __init__.py of this
# module

from src.data.prompting import (
    PVP_DICT,
    DatasetPVPs,
    ManualTemplateFactory,
    ManualVerbalizerFactory
)


PVP_DICT['religious_hate'] = {
    'en-abusive': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
            ManualTemplateFactory('Is this hate speech? {"placeholder":"text_a"} {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                'NOT-ABUSIVE': ["neutral"],
                'ABUSIVE': ["abusive"]
            }),
            ManualVerbalizerFactory({
                'NOT-ABUSIVE': ["No"],
                'ABUSIVE': ["Yes"]
            }),
        ],
        pvps={
            0: (0, 0),
            1: (1, 1),
        },
    ),
    'en-religious_bin': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
            ManualTemplateFactory('Is this hate speech? {"placeholder":"text_a"} {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                'NOT-ABUSIVE': ["neutral"],
                'ABUSIVE': ["abusive"]
            }),
            ManualVerbalizerFactory({
                'NOT-ABUSIVE': ["No"],
                'ABUSIVE': ["Yes"]
            }),
        ],
        pvps={
            0: (0, 0),
            1: (1, 1),
        },
    ),
    'it-abusive': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
            ManualTemplateFactory(
                'Questo è un discorso di odio? {"placeholder":"text_a"} {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                'NOT-ABUSIVE': ["neutral"],
                'ABUSIVE': ["abusive"]
            }),
            ManualVerbalizerFactory({
                "NOT-ABUSIVE": ["No"],
                "ABUSIVE": ["Sì"],
            }),
        ],
        pvps={
            0: (0, 0),
            1: (1, 1),
        },
    ),
    'it-religious_bin': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
            ManualTemplateFactory(
                'Questo è un discorso di odio? {"placeholder":"text_a"} {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                'NOT-ABUSIVE': ["neutral"],
                'ABUSIVE': ["abusive"]
            }),
            ManualVerbalizerFactory({
                "NOT-ABUSIVE": ["No"],
                "ABUSIVE": ["Sì"],
            }),
        ],
        pvps={
            0: (0, 0),
            1: (1, 1),
        },
    ),
}
