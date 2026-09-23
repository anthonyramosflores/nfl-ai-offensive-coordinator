"""
Stage C: simple feedforward classifier.
Input: defensive shell features
Output: predicted best play_family (or per-family success rate)
"""
import torch.nn as nn

class CoordinatorNet(nn.Module):
    def __init__(self, input_dim, num_play_families):
        super().__init__()
        # TODO: define a couple of Linear + activation layers
        pass

    def forward(self, x):
        # TODO
        pass