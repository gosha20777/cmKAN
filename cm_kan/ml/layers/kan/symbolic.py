import torch
from torch import nn
import torch.nn.functional as F


class KANSymbolic(nn.Module):
    "Defines and stores the Symbolic functions fixed / set for a KAN."

    def __init__(self, in_dim: int, out_dim: int):
        """
        We have to store a 2D array of univariate functions, one for each
        edge in the KAN layer. 
        """
        super(KANSymbolic, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fns = [[lambda x: x for _ in range(in_dim)] for _ in range(out_dim)]
    
    def forward(self, x: torch.Tensor):
        """
        Run symbolic activations over all inputs in x, where
        x is of shape (batch_size, in_dim). Returns a tensor of shape
        (batch_size, out_dim, in_dim).
        """
        
        acts = []
        # Really inefficient, try tensorizing later.
        for j in range(self.in_dim):
            act_ins = []
            for i in range(self.out_dim):
                o = torch.vmap(self.fns[i][j])(x[:,[j]]).squeeze(dim=-1)
                act_ins.append(o)
            acts.append(torch.stack(act_ins, dim=-1))
        acts = torch.stack(acts, dim=-1)

        return acts

    def set_symbolic(self, in_index: int, out_index: int, fn):
        """
        Set symbolic function at specified edge to new function.
        """
        self.fns[out_index][in_index] = fn 
