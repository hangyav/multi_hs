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
        ManualTemplateFactory('Is this hate speech? {"placeholder":"text_a"} {"mask"}'),
    ],
    prompt_verbalizers=[
        ManualVerbalizerFactory({
            'normal': ["neutral"],
            'hate': ["hate"]
        }),
        ManualVerbalizerFactory({
            'normal': ["No"],
            'hate': ["Yes"]
        }),
    ],
    pvps={
        0: (0, 0),
        1: (1, 1),
    },
)


_DEFAULT_FINE_GRAINED_PVP = DatasetPVPs(
    prompt_templates=[
        ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
    ],
    prompt_verbalizers=[
        ManualVerbalizerFactory({
            'normal': ["neutral"],
            'offensive': ["offensive"],
            'hate': ["hate"]
        }),
    ],
    pvps={
        0: (0, 0),
    },
)


PVP_DICT['large_scale_xdomain'] = {
    'en-binary-full': _DEFAULT_BINARY_PVP,
    'en-fine_grained-full': _DEFAULT_FINE_GRAINED_PVP,
    'en-binary-politics': _DEFAULT_BINARY_PVP,
    'en-fine_grained-politics': _DEFAULT_FINE_GRAINED_PVP,
    'en-binary-religion': _DEFAULT_BINARY_PVP,
    'en-fine_grained-religion': _DEFAULT_FINE_GRAINED_PVP,
}
