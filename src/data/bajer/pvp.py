# Make sure to run this code somewhere, e.g., import it in __init__.py of this
# module

from src.data.prompting import (
    PVP_DICT,
    DatasetPVPs,
    ManualTemplateFactory,
    ManualVerbalizerFactory
)

PVP_DICT['bajer'] = {
    'fine_grained': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} Det ver {"mask"}'),
            #
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                'SEX': ['sexistik'],
                'RAC': ['racist'],
                'OTH': ['neutral'],
            }),
            ManualVerbalizerFactory({
                'SEX': ['sexistik', 'misantrop'],
                'RAC': ['racist', 'apartheid', 'fremmedfjendsk'],
                'OTH': ["neutral", "uskyldig", "pæn", "godt"],
            }),
            #
            ManualVerbalizerFactory({
                'SEX': ['sexist'],
                'RAC': ['racist'],
                'OTH': ['neutral'],
            }),
            ManualVerbalizerFactory({
                'SEX': ['sexist', 'misanthrope'],
                'RAC': ['racist', 'apartheid', 'xenophobic'],
                'OTH': ["neutral", "innocent", "nice", "good"],
            }),
        ],
        pvps={
            0: (0, 0),
            1: (0, 1),
            2: (1, 2),
            3: (1, 3),
        },
    ),
    'sexism_binary': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} Det ver {"mask"}'),
            #
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                "SEX": ["sexistisk"],
                "NOT": ["neutral"],
            }),
            ManualVerbalizerFactory({
                "SEX": ["sexistisk", "offensiv", "ekstrem", "dårligt"],
                "NOT": ["neutral", "uskyldig", "pæn", "godt"],
            }),
            #
            ManualVerbalizerFactory({
                "SEX": ["sexist"],
                "NOT": ["neutral"],
            }),
            ManualVerbalizerFactory({
                "SEX": ["sexist", "offensive", "extreme", "bad"],
                "NOT": ["neutral", "innocent", "nice", "good"],
            }),
        ],
        pvps={
            0: (0, 0),
            1: (0, 1),
            2: (1, 2),
            3: (1, 3),
        },
    ),
}
