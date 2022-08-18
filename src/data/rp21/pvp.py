# Make sure to run this code somewhere, e.g., import it in __init__.py of this
# module

from src.data.prompting import (
    PVP_DICT,
    DatasetPVPs,
    ManualTemplateFactory,
    ManualVerbalizerFactory
)

PVP_DICT['rp21'] = {
    'fine_grained': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} Es war {"mask"}'),
            #
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                'sexism': ['sexistisch'],
                'racism': ['rassistisch'],
                'threat': ['bedrohlich'],
                'insult': ['beleidigend'],
                'profanity': ['entweihen'],
                'meta': ['Meta'],
                'advertisement': ['Anzeige'],
                'none': ['neutral'],
            }),
            #
            ManualVerbalizerFactory({
                'sexism': ['sexist'],
                'racism': ['racist'],
                'threat': ['threatening'],
                'insult': ['insulting'],
                'profanity': ['profane'],
                'meta': ['meta'],
                'advertisement': ['advertisement'],
                'none': ['neutral'],
            }),
        ],
        pvps={
            0: (0, 0),
            1: (1, 1),
        },
    ),
}
