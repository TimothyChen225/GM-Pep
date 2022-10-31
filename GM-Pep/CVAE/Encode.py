import torch
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()


        self.enmbeding = nn.Embedding(21, 20)

        self.linearmu = nn.Linear(512, 350)
        self.linearlogvar = nn.Linear(512, 350)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels= 100, out_channels= 16, kernel_size=(2,2), stride= 2),
            nn.MaxPool2d((2,2),1),
            nn.Conv2d(in_channels= 16, out_channels= 1, kernel_size= (4,4), stride = 2)

        )
        self.linear = nn.Sequential(
            nn.Linear(12, 512),
            torch.nn.ReLU(),

        )

    def forward(self, x, label):

        x = torch.cat([x, label.float()], -1)

        x = x.squeeze(1)
        x = x.view(x.shape[1], x.shape[0], x.shape[2])

        h_e = self.enmbeding(x.long())

        h_e = h_e.view(h_e.shape[1],h_e.shape[0],h_e.shape[2], h_e.shape[3])

        h = self.conv(h_e)

        out = h.squeeze(1)

        out = out.view(out.shape[0], -1)

        out = self.linear(out)

        mu = self.linearmu(out)

        logvar = self.linearlogvar(out)

        z = self.reparameterize(mu, logvar)
        z = z.view(z.shape[0], -1, 350)

        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


