# Make sure to run this code somewhere, e.g., import it in __init__.py of this
# module

from src.data.prompting import (
    PVP_DICT,
)
from src.data.haspeede3_dev.pvp import DEFAULT_PVP

PVP_DICT['haspeede3_st'] = {
    'politics_text': DEFAULT_PVP,
    'religious_text': DEFAULT_PVP,
}
