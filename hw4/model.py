import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=1,
        dropout=0.0,
    ):
        super(Model, self).__init__()

        # in:  seq, batch, feature
        # out: seq, batch_size, hidden_dim
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        # in: batch_size,input_dim
        # out: batch size,output_dim
        self.output_layer = nn.Linear(
            in_features=hidden_dim,
            out_features=output_dim
        )

        # self.init_weight()

    def forward(self, x):
        batch_size, seq_len, feature = x.size()
        x = x.permute(1, 0, 2)
        gru_out, _ = self.gru(x, None)
        out = self.output_layer(gru_out[-1].view(batch_size, -1))

        return out.view(-1)

    def init_weight(self):
        ih = (param for name, param
              in self.named_parameters() if 'weight_ih' in name)
        hh = (param for name, param
              in self.named_parameters() if 'weight_hh' in name)

        b = (param for name, param
             in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t)
