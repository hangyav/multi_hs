# Make sure to run this code somewhere, e.g., import it in __init__.py of this
# module

from src.data.prompting import (
    PVP_DICT,
    DatasetPVPs,
    ManualTemplateFactory,
    MixedTemplateFactory,
    ManualVerbalizerFactory
)


PVP_DICT['hate_speech18'] = {
    'binary': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
            MixedTemplateFactory('{"soft": "hate_speech18"} {"soft": "binary"} {"placeholder":"text_a"} It was {"mask"}'),
            MixedTemplateFactory('{"soft": "What type of abusive content does the text have?"} {"placeholder":"text_a"} {"mask"}'),
            MixedTemplateFactory('{"placeholder":"text_a"} {"soft": "What type of abusive content does the text have?"} {"mask"}'),
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
            2: (1, 0),
            3: (1, 1),
            4: (2, 0),
            5: (3, 1),
        },
    ),
}
