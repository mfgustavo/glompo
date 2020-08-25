import os
import shutil

import pytest

from glompo.common.helpers import FileNameHandler, distance, is_bounds_valid, nested_string_formatting


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

    @pytest.mark.parametrize('bnds, output', [([(0, 1)] * 5, True),
                                              ([(1, -1)] * 5, False),
                                              ([(0, float('inf'))] * 5, False)])
    def test_bounds(self, bnds, output):
        assert is_bounds_valid(bnds, raise_invalid=False) == output
        if not output:
            with pytest.raises(ValueError):
                is_bounds_valid(bnds, raise_invalid=True)

    def test_distance(self):
        assert distance([1] * 9, [-1] * 9) == 6

    def test_file_name_handler(self):
        start_direc = os.getcwd()
        with FileNameHandler('_tmp/fnh') as name:
            assert os.getcwd() == start_direc + os.sep + '_tmp'
            assert name == 'fnh'
        assert os.getcwd() == start_direc

    @classmethod
    def teardown_class(cls):
        shutil.rmtree("_tmp", ignore_errors=True)
