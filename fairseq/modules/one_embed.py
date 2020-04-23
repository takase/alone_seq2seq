import math

import os
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class OneEmbed(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, one_emb_type='binary', dropout=0.5, std=1.0, codenum=64, codebooknum=8, layernum=1, interdim=0, relu_dropout=0.1, mask_file=''):
        super(OneEmbed, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.one_emb_type = one_emb_type
        self.layernum = layernum
        self.relu_dropout = relu_dropout
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.weight = Parameter(torch.Tensor(1, embedding_dim)) #embedding for all tokens
        if interdim == 0:
            interdim = embedding_dim
        self.weight_matrices = nn.ParameterList([nn.Parameter(torch.Tensor(embedding_dim, interdim)) if i+1 == self.layernum else (nn.Parameter(torch.Tensor(interdim, embedding_dim)) if i == 0 else nn.Parameter(torch.Tensor(interdim, interdim))) for i in range(self.layernum)])
        if os.path.isfile(mask_file):
            self.mask = torch.load(mask_file)
        else:
            if self.one_emb_type == 'binary':
                prob = torch.Tensor(codenum, embedding_dim)
                nn.init.constant_(prob, (1 - dropout ** (1.0 / codebooknum)))
                self.masklist = [torch.bernoulli(prob) for _ in range(codebooknum)]
            else:
                mean_m = torch.zeros(codenum, embedding_dim)
                std_m = torch.Tensor(codenum, embedding_dim)
                nn.init.constant_(std_m, std * (codebooknum ** -0.5))
                self.masklist = [torch.normal(mean_m, std_m) for _ in range(codebooknum)]
            self.hash2mask = torch.randint(0, codenum, (num_embeddings, codebooknum), dtype=torch.long)
            self.mask = self.construct_mask2each_token() #mask for each token
            dirname = '/'.join(mask_file.split('/')[:-1])
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            torch.save(self.mask, mask_file)


    def construct_mask2each_token(self):
        mask = []
        for i in range(self.hash2mask.size(1)):
            token_hash = self.hash2mask[:, i]
            mask.append(nn.functional.embedding(token_hash, self.masklist[i], padding_idx=self.padding_idx))
        mask = sum(mask)
        if self.one_emb_type == 'binary':
            mask.clamp_(0, 1)
        return mask


    def construct_matrix_for_output_layer(self):
        vocab_vec = self.mask.new(range(self.num_embeddings)).long()
        matrix = self.forward(vocab_vec, dropout=0)
        return matrix


    def forward(self, input, dropout=None):
        if input.is_cuda and not self.mask.is_cuda:
            self.mask = self.mask.cuda()
        relu_dropout = self.relu_dropout if dropout is None else dropout
        each_token_mask = nn.functional.embedding(input, self.mask, padding_idx=self.padding_idx)
        embed = each_token_mask * self.weight.expand_as(each_token_mask)
        for i in range(self.layernum):
            embed = nn.functional.linear(embed, self.weight_matrices[i])
            if i+1 != self.layernum:
                embed = nn.functional.relu(embed)
                embed = nn.functional.dropout(embed, p=relu_dropout, training=self.training)
        return embed


