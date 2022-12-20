import logging
import itertools
import torch
import torch.nn as nn
from torch.nn.functional import softmax, relu
from torch.nn import Parameter


class RGL(nn.Module):
    def __init__(self, num_layer):
        """ The current code might not be compatible with models trained with previous version
        """
        super().__init__()
        final_state_dim = 128
        self.num_layer = num_layer
        self.X_dim = 128
        self.layerwise_graph = True
        self.skip_connection = False

        self.w_a = nn.Parameter(torch.randn(self.X_dim, self.X_dim))
        # self.w_a = nn.Parameter(torch.Tensor(self.X_dim, self.X_dim))
        # TODO: try other dim size
        embedding_dim = self.X_dim
        self.Ws = torch.nn.ParameterList()
        for i in range(self.num_layer):
            if i == 0:
                self.Ws.append(nn.Parameter(torch.randn(self.X_dim, embedding_dim)))
            elif i == self.num_layer - 1:
                self.Ws.append(nn.Parameter(torch.randn(embedding_dim, final_state_dim)))
            else:
                self.Ws.append(nn.Parameter(torch.randn(embedding_dim, embedding_dim)))
        
        # for visualization
        self.A = None
        # import pdb
        # pdb.set_trace()

    def compute_similarity_matrix(self, X):
        A = torch.matmul(torch.matmul(X, self.w_a), X.permute(0, 2, 1))
        normalized_A = softmax(A, dim=2)
        return normalized_A

    def forward(self,X):
        """
        Embed current state tensor pair (robot_state, human_states) into a latent space
        Each tensor is of shape (batch_size, # of agent, features)
        :param state:
        :return:
        """

        # compute feature matrix X

        # X = torch.cat([robot_state_embedings, human_state_embedings], dim=1)

        # compute matrix A
        if not self.layerwise_graph:
            normalized_A = self.compute_similarity_matrix(X)
            self.A = normalized_A[0, :, :].data.cpu().numpy()

        next_H = H = X
        for i in range(self.num_layer):
            if self.layerwise_graph:
                A = self.compute_similarity_matrix(H)
                next_H = relu(torch.matmul(torch.matmul(A, H), self.Ws[i]))
            else:
                next_H = relu(torch.matmul(torch.matmul(normalized_A, H), self.Ws[i]))

            if self.skip_connection:
                next_H += H
            H = next_H

        return next_H


