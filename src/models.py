'''
Models definition
'''

import torch
import torch.nn as nn
from torchvision import models


class CRNN(nn.Module):
    def __init__(self, num_classes=100, hidden_size=256, resnet_pretrained_weights=None):
        super(CRNN, self).__init__()
        resnet = models.resnet18(weights=resnet_pretrained_weights)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        self.feature_dim = 512
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.rnn = nn.LSTM(self.feature_dim, hidden_size, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        features = self.cnn(x)
        pooled = self.pool(features).squeeze(-1).squeeze(-1)
        seq = pooled.view(B, T, self.feature_dim)
        rnn_out, _ = self.rnn(seq)
        final = rnn_out[:, -1, :]
        return self.fc(final)