import os

from glompo.core.checkpointing import CheckpointingControl


def test_findcount():
    os.mkdir('checkpoints')
    os.mkdir(os.path.join('checkpoints', '000_s[p$e^c._{h}e|llo00000000000000_%(weird)_]000000000000000000'))
    os.mkdir(os.path.join('checkpoints', '001_s[p$e^c._{h}e|llo00000000000000_%(weird)_]000000000000000000'))
    os.mkdir(os.path.join('checkpoints', '002_s[p$e^c._{h}e|llo00000000000000_%(weird)_]000000000000000000'))

    cc = CheckpointingControl(checkpoint_frequency=10,
                              checkpoint_at_init=False,
                              checkpoint_at_conv=False,
                              keep_past=0,
                              naming_format='%(count)_s[p$e^c._{h}e|llo%(date)%(time)_%(weird)_'
                                            ']%(hour)%(year)%(yr)%(month)%(day)%(hour)%(min)%(sec)',
                              checkpointing_dir='checkpoints')
    assert cc.count == 3
