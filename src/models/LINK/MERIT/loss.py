"""Loss module for training."""

import torch


class MultiLabelLoss(torch.nn.Module):
    def forward(self, logits):
        """Compute loss from prediction logits."""
        return -torch.log(
            torch.div(torch.exp(logits[:, 0]), torch.sum(torch.exp(logits), -1))
        ).sum()
