import torch
import torch.nn as nn

from sparselayer import BlocksparseConv, BlocksparseLinear

__all__ = ['BlocksparseModel']

class BlocksparseModel(nn.Module):
    def __init__(self, model, block_sizes, pruning_rates, shuffle=True):
        super(BlocksparseModel, self).__init__()

        features = []
        sparse_index = 0
        for layer in model.features:
            print("=> converting layer %s" % str(layer))
            if type(layer) is nn.Conv2d:
                features.append(BlocksparseConv(layer, block_sizes[sparse_index], pruning_rates[sparse_index], shuffle))
                sparse_index += 1
            else:
                features.append(layer)

        classifier = []
        for layer in model.classifier:
            print("=> converting layer %s" % str(layer))
            if type(layer) is nn.Linear:
                classifier.append(BlocksparseLinear(layer, block_sizes[sparse_index], pruning_rates[sparse_index], shuffle))
                sparse_index += 1
            else:
                classifier.append(layer)

        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y

