# calculate our model's parameters.
from torch import load

xornet = load('xornet.pt', weights_only=False)

print(sum(p.numel() for p in xornet.parameters()))