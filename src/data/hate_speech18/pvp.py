# Make sure to run this code somewhere, e.g., import it in __init__.py of this
# module

from src.data.prompting import (
    PVP_DICT,
    DatasetPVPs,
    ManualTemplateFactory,
    ManualVerbalizerFactory
)


PVP_DICT['hate_speech18'] = {
    'binary': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                'noHate': ["neutral"],
                'hate': ["hate"]
            }),
            ManualVerbalizerFactory({
                'noHate': ["neutral", "innocent", "nice", "good"],
                'hate': ["hate", "offensive", "abusive", "bad"]
            }),
        ],
        pvps={
            0: (0, 0),
            1: (0, 1),
        },
    ),
}
