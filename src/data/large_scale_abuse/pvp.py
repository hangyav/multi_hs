# Make sure to run this code somewhere, e.g., import it in __init__.py of this
# module

from src.data.prompting import (
    PVP_DICT,
    DatasetPVPs,
    ManualTemplateFactory,
    ManualVerbalizerFactory
)

PVP_DICT['large_scale_abuse'] = {
    'fine_grained': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                'abusive': ['abusive'],
                'hateful': ['hate'],
                'spam': ['spam'],
                'normal': ['neutral'],
            }),
        ],
        pvps={
            0: (0, 0),
        },
    ),
}
