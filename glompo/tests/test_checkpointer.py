from pathlib import Path

import pytest
from glompo.core.checkpointing import CheckpointingControl


@pytest.fixture
def cc():
    return CheckpointingControl(naming_format='%(count)_s[p$e^c._{h}e|llo%(date)%(time)_%(weird)_]%(hour)%(year)'
                                              '%(yr)%(month)%(day)%(hour)%(min)%(sec)')


def test_findcount_nodir(cc):
    # Test dir does not exist
    cc.get_name()
    assert cc.count == 1


def test_findcount_nocontent(tmp_path, cc):
    # Test dir does exist but no contents
    cc.checkpointing_dir = tmp_path
    cc.get_name()
    assert cc.count == 1


def test_find_count_matchingcontent(tmp_path, cc):
    # Test dir exist with only matching content
    cc.checkpointing_dir = tmp_path
    for i in range(3):
        Path(tmp_path, f'00{i}_s[p$e^c._{{h}}e|llo00000000000000_%(weird)_]000000000000000000').mkdir()
    cc.get_name()
    assert cc.count == 4


def test_find_count_mixedcontent(tmp_path, cc):
    # Test dir exist with matching and non-matching content
    cc.checkpointing_dir = tmp_path
    Path(tmp_path, '000_s[p$e^c._{h}e|llo00000000000000_%(weird)_]000000000000000000').mkdir()
    Path(tmp_path, '000_s[p$e^c._{h}e|llo0_%(weird)_]000000000000000000').mkdir()
    cc.get_name()
    assert cc.count == 2


def test_matchname(cc):
    assert cc.matches_naming_format('000_s[p$e^c._{h}e|llo00000000000000_%(weird)_]000000000000000000')
    assert not cc.matches_naming_format('00_sp$e^c._{h}e|llo00000000123000_]000000000000000')
