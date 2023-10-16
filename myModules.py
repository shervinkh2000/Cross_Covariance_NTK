# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import math

# def FilterFunction(h, S, x):
#     K = h.shape[0]
#     B = x.shape[0]
#     N = x.shape[1]

#     x = x.reshape([B, 1, N])
#     S = S.reshape([1, N, N])
#     z = x
#     for k in range(1, K):
#         x = torch.matmul(x, S)
#         xS = x.reshape([B, 1, N])
#         z = torch.cat((z, xS), dim = 1)
#     y = torch.matmul(z.permute(0, 2, 1).reshape([B, N, K]), h)
#     return y

    
# class GraphFilter(nn.Module):
#     def __init__(self, gso, k):
#         super().__init__()
#         self.gso = torch.tensor(gso)
#         self.n = gso.shape[0]
#         self.k = k
#         self.weight = nn.Parameter(torch.randn(self.k))
#         self.bias = nn.Parameter(torch.randn(1))
#         self.reset_parameters()
        
        
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.k)
#         self.weight.data.uniform_(-stdv, stdv)
#         self.bias.data.uniform_(-stdv, stdv)

#     def forward(self, x):
#         return FilterFunction(self.weight, self.gso, x) + self.bias




###################
#Multiple features#
###################

def FilterFunction(h, S, x):
    
    # Number of output features
    F = h.shape[0]
    
    # Number of filter taps
    K = h.shape[1]
    
    # Number of input features
    G = h.shape[2]
    
    # Number of nodes
    N = S.shape[1]
    
    # Batch size
    B = x.shape[0]

    # Create concatenation dimension and initialize concatenation tensor z
    x = x.reshape([B, 1, G, N])
    S = S.reshape([1, N, N])
    z = x
    
    # Loop over the number of filter taps
    for k in range(1, K):
        
        # S*x
        x = torch.matmul(x, S)
        
        # Reshape
        xS = x.reshape([B, 1, G, N])
        
        # Concatenate
        z = torch.cat((z, xS), dim=1)
    
    # Multiply by h
    y = torch.matmul(z.permute(0, 3, 1, 2).reshape([B, N, K*G]), 
                     h.reshape([F, K*G]).permute(1, 0)).permute(0, 2, 1)
    return y

class GraphFilter(nn.Module):
    def __init__(self, gso, k, f_in, f_out):
        super().__init__()
        self.gso = torch.tensor(gso)
        self.n = gso.shape[0]
        self.k = k
        self.f_in = f_in
        self.f_out = f_out
        self.weight = nn.Parameter(torch.randn(self.f_out, self.k, self.f_in))
        self.bias = nn.Parameter(torch.randn(1))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.f_in * self.k)

        stdv = 0.1*stdv
        self.bias.data.uniform_(-stdv, stdv)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return FilterFunction(self.weight, self.gso, x) + self.bias