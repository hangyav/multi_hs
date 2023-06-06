import datasets
from datasets.utils import logging


logger = logging.get_logger(__name__)


_CITATION = """
"""

_DESCRIPTION = """
"""

FUNCTIONALITIES = {
    # Derogation
    'F1': 'derog_neg_emote_h',
    'F2': 'derog_neg_attrib_h',
    'F3': 'derog_dehum_h',
    'F4': 'derog_impl_h',
    # Threatening language
    'F5': 'threat_dir_h',
    'F6': 'threat_norm_h',
    # Slur usage
    'F7': 'slur_h',
    'F8': 'slur_homonym_nh',
    'F9': 'slur_reclaimed_nh',
    # Profanity usage
    'F10': 'profanity_h',
    'F11': 'profanity_nh',
    # Pronoun reference
    'F12': 'ref_subs_clause_h',
    'F13': 'ref_subs_sent_h',
    # Negation
    'F14': 'negate_pos_h',
    'F15': 'negate_neg_nh',
    # Phrasing
    'F16': 'phrase_question_h',
    'F17': 'phrase_opinion_h',
    # Non-hate group identity
    'F18': 'ident_neutral_nh',
    'F19': 'ident_pos_nh',
    # Counter speech
    'F20': 'counter_quote_nh',
    'F21': 'counter_ref_nh',
    # Abuse against non-protected group
    'F22': 'target_obj_nh',
    'F23': 'target_indiv_nh',
    'F24': 'target_group_nh',
    # Spelling variations
    'F25': 'spell_char_swap_h',
    'F26': 'spell_char_del_h',
    'F27': 'spell_space_del_h',
    'F28': 'spell_space_add_h',
    'F29': 'spell_leet_h',
}


class HateCheck(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name=func, version=datasets.Version("1.0.0"))
        for func in FUNCTIONALITIES.keys()
    ]

    def _info(self):
        features = datasets.Features(
            {
                'id': datasets.Value('int64'),
                'text': datasets.Value('string'),
                'label': datasets.features.ClassLabel(
                    names=[
                        'non-hateful',
                        'hateful',
                    ]
                ),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage='https://github.com/paul-rottger/hatecheck-data',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        dataset = datasets.load_dataset("Paul/hatecheck")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    'dataset': dataset['test']
                }
            )
        ]

    def _generate_examples(self, dataset):
        func = self.config.name
        for idx, item in enumerate(filter(
            lambda x: x['functionality'] == FUNCTIONALITIES[func],
            dataset
        )):
            yield idx, {
                'id': item['case_id'],
                'text': item['test_case'],
                'label': item['label_gold'],
            }
