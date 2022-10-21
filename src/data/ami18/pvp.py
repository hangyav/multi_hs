# Make sure to run this code somewhere, e.g., import it in __init__.py of this
# module

from src.data.prompting import (
    PVP_DICT,
    DatasetPVPs,
    ManualTemplateFactory,
    MixedTemplateFactory,
    ManualVerbalizerFactory
)

PVP_DICT['ami18'] = {
    'en-misogyny': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} Was it misogynisctic? {"mask"}'),
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
            ManualTemplateFactory('{"placeholder":"text_a"} Was it sexist or neutral? It was {"mask"}'),
            MixedTemplateFactory('{"soft": "AMI18"} {"soft": "en-misogyny"} {"placeholder":"text_a"} It was {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                'non-misogyny': ["No"],
                'misogyny': ["Yes"]
            }),
            ManualVerbalizerFactory({
                'non-misogyny': ["neutral"],
                'misogyny': ["sexist"]
            }),
            ManualVerbalizerFactory({
                'non-misogyny': ["neutral", "innocent", "nice", "good"],
                'misogyny': ["sexist", "offensive", "extreme", "bad"],
            }),
            ManualVerbalizerFactory({
                'non-misogyny': ["neutral"],
                'misogyny': ["sexist", "stereotypical", "dominance", "derailing", "harasment", "discrediting"]
            }),
        ],
        pvps={
            0: (0, 0),
            1: (1, 1),
            2: (1, 2),
            3: (2, 1),
            4: (3, 1),
            5: (2, 2),
            6: (1, 3),
        },
    ),
    'en-misogyny_category': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
            MixedTemplateFactory('{"soft": "AMI18"} {"soft": "en-misogyny_category"} {"placeholder":"text_a"} It was {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                "stereotype": ["stereotypical"],
                "dominance": ["dominance"],
                "derailing": ["derailing"],
                "sexual_harassment": ["harassment"],
                "discredit": ["discrediting"],
            }),
            ManualVerbalizerFactory({
                "stereotype": ["stereotypical", "stereotypic", "typical"],
                "dominance": ["dominance", "domination", "dominion"],
                "derailing": ["derailing", "deflecting", "redirecting"],
                "sexual_harassment": ["harasment", "violence", "threat"],
                "discredit": ["discrediting", "degrading", "offensive"],
            }),
        ],
        pvps={
            0: (0, 0),
            1: (0, 1),
            2: (1, 0),
            3: (1, 1),
        },
    ),
    'en-target': DatasetPVPs(
        prompt_templates=[
            # ManualTemplateFactory('{"placeholder":"text_a"} It was targeted {"mask"}'),
            ManualTemplateFactory('{"placeholder":"text_a"} It was targeted at {"mask"}'),
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                'active': ["individual"],
                'passive': ["group"]
            }),
            ManualVerbalizerFactory({
                'active': ["active"],
                'passive': ["passive"]
            }),
        ],
        pvps={
            0: (0, 0),
            1: (1, 1),
        },
    ),
    'it-misogyny': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} Era misogino? {"mask"}'),
            ManualTemplateFactory('{"placeholder":"text_a"} Era {"mask"}'),
            #
            ManualTemplateFactory('{"placeholder":"text_a"} Was it misogynisctic? {"mask"}'),
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
            ManualTemplateFactory('{"placeholder":"text_a"} Was it sexist or neutral? It was {"mask"}'),
            MixedTemplateFactory('{"soft": "AMI18"} {"soft": "it-misogyny"} {"placeholder":"text_a"} It was {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                "misogyny": ["Sì"],
                "non-misogyny": ["No"],
            }),
            ManualVerbalizerFactory({
                "misogyny": ["sessista"],
                "non-misogyny": ["neutrale"],
            }),
            ManualVerbalizerFactory({
                "misogyny": ["sessista", "offensivo", "estremo", "cattivo"],
                "non-misogyny": ["neutrale", "innocente", "simpatico", "buono"],
            }),
            #
            ManualVerbalizerFactory({
                'non-misogyny': ["No"],
                'misogyny': ["Yes"]
            }),
            ManualVerbalizerFactory({
                'non-misogyny': ["neutral"],
                'misogyny': ["sexist"]
            }),
            ManualVerbalizerFactory({
                'non-misogyny': ["neutral", "innocent", "nice", "good"],
                'misogyny': ["sexist", "offensive", "extreme", "bad"],
            }),
        ],
        pvps={
            0: (0, 0),
            1: (1, 1),
            2: (1, 2),
            3: (2, 3),
            4: (3, 4),
            5: (3, 5),
            6: (4, 4),
            7: (5, 4),
            8: (5, 5),
        },
    ),
    'it-misogyny_category': DatasetPVPs(
        prompt_templates=[
            ManualTemplateFactory('{"placeholder":"text_a"} Era {"mask"}'),
            #
            ManualTemplateFactory('{"placeholder":"text_a"} It was {"mask"}'),
            MixedTemplateFactory('{"soft": "AMI18"} {"soft": "it-misogyny_category"} {"placeholder":"text_a"} It was {"mask"}'),
        ],
        prompt_verbalizers=[
            ManualVerbalizerFactory({
                "stereotype": ["stereotipato"],
                "dominance": ["predominante"],
                "derailing": ["deragiante"],
                "sexual_harassment": ["molesto"],
                "discredit": ["screditante"],
            }),
            ManualVerbalizerFactory({
                # stereotipic is translated as stereotipato as well
                "stereotype": ["stereotipato", "tipico"],
                # dominion is also dominio
                "dominance": ["predominante", "dominio"],
                "derailing": ["deragiante", "deviazione", "reindirizzamento"],
                "sexual_harassment": ["molesto", "violenza", "minaccia"],
                "discredit": ["screditante", "degradante", "offensivo"],
            }),
            #
            ManualVerbalizerFactory({
                "stereotype": ["stereotypical"],
                "dominance": ["dominance"],
                "derailing": ["derailing"],
                "sexual_harassment": ["harasment"],
                "discredit": ["discrediting"],
            }),
            ManualVerbalizerFactory({
                "stereotype": ["stereotypical", "stereotypic", "typical"],
                "dominance": ["dominance", "domination", "dominion"],
                "derailing": ["derailing", "deflecting", "redirecting"],
                "sexual_harassment": ["harasment", "violence", "threat"],
                "discredit": ["discrediting", "degrading", "offensive"],
            }),
        ],
        pvps={
            0: (0, 0),
            1: (0, 1),
            2: (1, 2),
            3: (1, 3),
            4: (2, 2),
            5: (2, 3),
        },
    ),
}

