import torch, torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.prec = torch.float64
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=self.prec) if x.dtype != self.prec else x
        return x * 2

try:
    sm = torch.jit.script(M())
    out = sm(torch.randn(2))
    print("COMPILE+RUN OK ->", out.dtype, out.shape)
except Exception as e:
    print("FAIL:", type(e).__name__, str(e).splitlines()[0][:160])
