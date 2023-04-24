# Make sure to run this code somewhere, e.g., import it in __init__.py of this
# module

from src.data.prompting import (
    PVP_DICT,
    DatasetPVPs,
    ManualTemplateFactory,
    ManualVerbalizerFactory
)


_DEFAULT_BINARY_PVP = DatasetPVPs(
    prompt_templates=[
        ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
    ],
    prompt_verbalizers=[
        ManualVerbalizerFactory({
            'normal': ["neutral"],
            'hate': ["hate"]
        }),
    ],
    pvps={
        0: (0, 0),
    },
)


_DEFAULT_FINE_GRAINED_PVP = DatasetPVPs(
    prompt_templates=[
        ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
    ],
    prompt_verbalizers=[
        ManualVerbalizerFactory({
            'counter': ["supportive"],
            'normal': ["neutral"],
            'hate': ["hate"]
        }),
    ],
    pvps={
        0: (0, 0),
    },
)


PVP_DICT['measuring_hate'] = {
    'binary-full': _DEFAULT_BINARY_PVP,
    'fine_grained-full': _DEFAULT_FINE_GRAINED_PVP,
    'binary-politics': _DEFAULT_BINARY_PVP,
    'fine_grained-politics': _DEFAULT_FINE_GRAINED_PVP,
}
