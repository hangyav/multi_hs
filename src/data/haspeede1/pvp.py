# Make sure to run this code somewhere, e.g., import it in __init__.py of this
# module

from src.data.prompting import (
    PVP_DICT,
    DatasetPVPs,
    ManualTemplateFactory,
    ManualVerbalizerFactory
)


_DEFAULT_PVP = DatasetPVPs(
    prompt_templates=[
        ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
        ManualTemplateFactory(
            'Questo è un discorso di odio? {"placeholder":"text_a"} {"mask"}'),
    ],
    prompt_verbalizers=[
        ManualVerbalizerFactory({
            'Non-Hateful': ["neutral"],
            'Hateful': ["hate"]
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
)


PVP_DICT['haspeede1'] = {
    'FB': _DEFAULT_PVP,
    'TW': _DEFAULT_PVP,
}
