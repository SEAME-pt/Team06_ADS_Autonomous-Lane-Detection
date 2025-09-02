import torch
from linenet import create_linenet

def test_forward_shape():
    m = create_linenet(variant='small', num_classes=1)
    x = torch.randn(1, 3, 224, 224)
    y = m(x)
    assert y.shape == (1, 1, 224, 224)
    # após sigmoid, a saída deve estar em [0,1]
    assert torch.all(y.ge(0)) and torch.all(y.le(1))

def test_all_variants_forward():
    for v in ['nano', 'lite', 'small', 'medium', 'strong']:
        m = create_linenet(variant=v, num_classes=1).eval()
        x = torch.randn(2, 3, 96, 96)
        with torch.no_grad():
            y = m(x)
        assert y.ndim == 4 and y.shape[1] == 1
PY

