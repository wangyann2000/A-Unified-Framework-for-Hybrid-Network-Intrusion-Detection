import torch.nn as nn
import torch.nn.functional as f


class EmbeddingNet(nn.Module):
    def __init__(self, opt):
        super(EmbeddingNet, self).__init__()

        self.fc1 = nn.Linear(opt.resSize, opt.embedSize)
        self.fc2 = nn.Linear(opt.embedSize, opt.outzSize)
        self.relu = nn.ReLU(True)

    def forward(self, features):
        embedding = self.relu(self.fc1(features))
        out_z = f.normalize(self.fc2(embedding), dim=1)
        return embedding, out_z