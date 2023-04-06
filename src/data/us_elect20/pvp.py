# Make sure to run this code somewhere, e.g., import it in __init__.py of this
# module

from src.data.prompting import (
    PVP_DICT,
    DatasetPVPs,
    ManualTemplateFactory,
    ManualVerbalizerFactory
)


PVP_DICT['us_elect20'] = {
    'binary': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                'Non-Hateful': ["neutral"],
                'Hateful': ["hate"]
            }),
        ],
        pvps={
            0: (0, 0),
        },
    ),
}
