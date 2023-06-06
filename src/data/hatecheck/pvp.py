# Make sure to run this code somewhere, e.g., import it in __init__.py of this
# module

from src.data.prompting import (
    PVP_DICT,
    DatasetPVPs,
    ManualTemplateFactory,
    ManualVerbalizerFactory
)
from .hatecheck import FUNCTIONALITIES

_DEFAULT_PVP = DatasetPVPs(
    prompt_templates=[
        ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
    ],
    prompt_verbalizers=[
        ManualVerbalizerFactory({
            'non-hateful': ["neutral"],
            'hateful': ["hate"]
        }),
    ],
    pvps={
        0: (0, 0),
    },
)

PVP_DICT['hatecheck'] = {
    func: _DEFAULT_PVP
    for func in FUNCTIONALITIES.keys()
}
