""" Describes a model to extract coreferential spans from a list of tokens.

  Usage example:

  model = CorefModel("config.toml", "debug")
  model.evaluate("dev")
"""

from .ner_coref_builder import NerCorefBuilder
from .ner_builder import NerBuilder
from .ner_coref_re_builder import NerCorefREBuilder

__all__ = ["NerBuilder", "NerCorefBuilder", "NerCorefREBuilder"]
