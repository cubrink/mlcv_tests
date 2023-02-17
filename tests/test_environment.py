import torch

def test_pytorch_version():
    MAIN_VERSION = 1
    MINOR_VERSION = 13
    version, _, _ = torch.__version__.partition('+')
    version = [int(v) for v in version.split('.')]
    assert version[0] == MAIN_VERSION
    assert version[1] == MINOR_VERSION
    
