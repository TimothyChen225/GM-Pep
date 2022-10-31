import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super(Decoder, self).__init__()

        self.hidden_szie = hidden_size

        self.model_Lstm1 = nn.LSTM(353, 350, num_layers = 2, bidirectional= True)



        self.model_Linear = nn.Sequential(
            nn.Linear(700, self.hidden_szie),

            nn.ReLU(),
            nn.Linear(self.hidden_szie, 100 * 21),
            nn.Sigmoid(),
        )

    def forward(self, z, label):


        z = torch.cat([z, label], -1)

        z = z.permute(1,0,2)

        h, _ = self.model_Lstm1(z)


        h = h.permute(1, 0, 2)

        h = h.view(h.shape[0], -1)

        out = self.model_Linear(h)

        out = out.view(-1, 100, 21)

        return out


