import torch
import pytest

from train import compute_accuracy

def test_arange_elems():
    arr = torch.arange(0, 10, dtype=torch.float)
    assert torch.allclose(arr[-1], torch.tensor([9]).float())

def test_div_zero():
    a = torch.zeros(1,dtype=torch.long)
    b = torch.ones(1,dtype=torch.long)

    assert not torch.isfinite(b/a)


def test_div_zero_python():
    with pytest.raises(ZeroDivisionError):
        1/0

def test_accuracy():
    preds = torch.randint(0,2,size=(100,))
    targets = preds.clone()

    assert compute_accuracy(preds, targets) == 1.0

    preds = torch.tensor([1,2,3,0,0,0])
    targets = torch.tensor([1,2,3,4,5,6])

    assert compute_accuracy(preds, targets) == 0.5

@pytest.mark.parametrize("preds,targets,result",[
    (torch.tensor([1,2,3]),torch.tensor([1,2,3]), 1.0),
    (torch.tensor([1,2,3]),torch.tensor([0,0,0]), 0.0),
    (torch.tensor([1,2,3]),torch.tensor([1,2,0]), 2/3),
    ])
def test_accuracy_parametrized(preds, targets, result):
    assert torch.allclose(compute_accuracy(preds, targets), torch.tensor([result]), rtol=0, atol=1e-5)
