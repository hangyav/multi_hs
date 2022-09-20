# Make sure to run this code somewhere, e.g., import it in __init__.py of this
# module

from src.data.prompting import (
    PVP_DICT,
    DatasetPVPs,
    ManualTemplateFactory,
    MixedTemplateFactory,
    ManualVerbalizerFactory
)

PVP_DICT['srw16'] = {
    'fine_grained': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                'sexism': ['sexist'],
                'racism': ['racist'],
                'none': ['neutral'],
            }),
        ],
        pvps={
            0: (0, 0),
        },
    ),
}
