"""
Lena-TRNN模型定义
包含Transformer编码器、GRU单元、能量线性头，以及能量更新步骤
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        out = self.transformer(x)
        out = self.proj(out)
        return out


class Lena_TRNN(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        self.input_proj = nn.Linear(input_dim, d_model)
        self.transformer = TransformerBlock(d_model, nhead, num_layers, dropout=dropout)
        self.gru = nn.GRUCell(d_model * 2, d_model)
        self.energy_head = nn.Linear(d_model, 1)
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x, return_energy=False):
        batch, seq_len, _ = x.shape
        x_proj = self.input_proj(x)
        c = self.transformer(x_proj)

        h = torch.zeros(batch, self.d_model, device=x.device)
        outputs = []
        energies = []
        for t in range(seq_len):
            gru_input = torch.cat([x_proj[:, t, :], c[:, t, :]], dim=-1)
            h = self.gru(gru_input, h)
            outputs.append(h.unsqueeze(1))
            energy_t = self.energy_head(h)
            energies.append(energy_t)

        h_seq = torch.cat(outputs, dim=1)
        out = self.output_proj(h_seq)

        if return_energy:
            energy = torch.cat(energies, dim=-1)
            return out, energy
        return out

    def energy_update(self, x, steps=3, alpha=0.1, return_all=False):
        x_current = x.detach().clone().requires_grad_(True)
        all_preds = []
        all_energies = []

        for step in range(steps):
            pred, energy = self.forward(x_current, return_energy=True)
            all_energies.append(energy.detach())
            if return_all:
                all_preds.append(pred.detach())

            grad = torch.autograd.grad(energy.sum(), x_current, retain_graph=True, create_graph=False)[0]
            with torch.no_grad():
                x_new = x_current - alpha * grad
            x_current = x_new.detach().requires_grad_(True)

        if return_all:
            return all_preds, all_energies
        else:
            pred, _ = self.forward(x_current, return_energy=False)
            return pred