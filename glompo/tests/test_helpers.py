

from glompo.common.helpers import *


class TestHelpers:

    def test_string(self):
        assert nested_string_formatting("[TrueHunter() AND\n"
                                        "[[TrueHunter() OR\n"
                                        "[FalseHunter() AND\n"
                                        "[TrueHunter() OR\n"
                                        "FalseHunter()]]]\n"
                                        "OR\n"
                                        "FalseHunter()]]") == \
               "TrueHunter() AND\n" \
               "[\n" \
               " [\n" \
               "  TrueHunter() OR\n" \
               "  [\n" \
               "   FalseHunter() AND\n" \
               "   [\n" \
               "    TrueHunter() OR\n" \
               "    FalseHunter()\n" \
               "   ]\n" \
               "  ]\n" \
               " ]\n" \
               " OR\n" \
               " FalseHunter()\n" \
               "]"

    def test_string_with_result(self):
        assert nested_string_formatting("[TrueHunter() = None AND\n"
                                        "[[TrueHunter() = None OR\n"
                                        "[FalseHunter() = None AND\n"
                                        "[TrueHunter() = None OR\n"
                                        "FalseHunter() = None]]]\n"
                                        "OR\n"
                                        "FalseHunter() = None]]") == \
               "TrueHunter() = None AND\n" \
               "[\n" \
               " [\n" \
               "  TrueHunter() = None OR\n" \
               "  [\n" \
               "   FalseHunter() = None AND\n" \
               "   [\n" \
               "    TrueHunter() = None OR\n" \
               "    FalseHunter() = None\n" \
               "   ]\n" \
               "  ]\n" \
               " ]\n" \
               " OR\n" \
               " FalseHunter() = None\n" \
               "]"
