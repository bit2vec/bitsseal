import math
import os
import torch
import torch.nn as nn 
from torch.nn.parameter import Parameter

class ByteSubword(nn.Module):
    """
    num_embeddings: dictionary size
    """
    def __init__(self, num_embeddings, embedding_dim, layernum=1, interdim=0,
                 std=1.0, relu_dropout=0.1, aggre='avg', subword_bytes_file='', embeddings_file='./embeddings') -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.layernum = layernum
        self.aggre = aggre
        self.relu_dropout = relu_dropout

        # print(num_embeddings)
        # print(padding_idx)

        self.byte_dict_size = 256
        self.max_byte_seq_len = 8
        

        # if padding_idx is not None:
        #     if padding_idx > 0:
        #         assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
        #     elif padding_idx == -1:
        #         padding_idx = None
        #     elif padding_idx < 0:
        #         assert padding_idx >= -self.num_embeddings, 'Padding_idx much be within num_embeddings'
        #         padding_idx = self.num_embeddings + padding_idx


        self.padding_idx = 1
        self.embedding_file = embeddings_file
        self.subword_bytes_file = subword_bytes_file

        # exit()

        # the embedding table
        # store the subword index to the byte sequences

        # if os.path.isfile(self.subword_bytes_file):
        #     self.subword_bytes_table = torch.load(self.subword_bytes_file)
        # else:
        #     print("please provide the subword to bytes table.")
        #     exit()

        # self.register_buffer('subword_bytes_table', subword_bytes_table)

        
        

        if os.path.isfile(subword_bytes_file):
            subword2bytes = torch.load(subword_bytes_file)
        else:
            subword2bytes = torch.randint(0, self.byte_dict_size, (num_embeddings, self.max_byte_seq_len), dtype=torch.long)
            dirname = '/'.join(subword_bytes_file.split('/')[:-1])
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            torch.save(subword2bytes, subword_bytes_file)
        
        self.register_buffer('subword2bytes', subword2bytes)

            # self.subword_bytes_table = self.subword_bytes_table.float()
            # max_byte_seq_len = self.subword_bytes_table.shape[1]
        
        # print(self.subword2bytes.size())
        # exit()
        # self.subword_bytes_embed = nn.Embedding(num_embeddings, self.max_byte_seq_len, _weight=self.subword2bytes)
        # self.subword_bytes_embed.requires_grad_(False)


        # the embedding table for bytes
        #  dim: [256, embedding_dim]

        if os.path.isfile(self.embedding_file):
            weight = torch.load(self.embedding_file)
            self.byte_embedding = nn.Embedding(self.byte_dict_size + 1, embedding_dim, _weight=weight, padding_idx=self.byte_dict_size)
        else:
            self.byte_embedding = nn.Embedding(self.byte_dict_size + 1, embedding_dim, padding_idx=self.byte_dict_size)

        self.weight = self.byte_embedding.weight

        if interdim == 0:
            interdim = embedding_dim


        subword = torch.tensor([i for i in range(self.num_embeddings)])
        self.register_buffer('subword', subword)

        # mean_m = torch.zeros(self.max_byte_seq_len, embedding_dim)
        # std_m = torch.Tensor(self.max_byte_seq_len, embedding_dim)
        # nn.init.constant_(std_m, std * (self.max_byte_seq_len ** -0.5))
        # self.pos_embedding = torch.normal(mean_m, std_m)

        

    
        
        # self.weight_matrices = nn.ParameterList([nn.Parameter(torch.Tensor(embedding_dim, interdim)) if i+1 == self.layernum
        #                                          else (nn.Parameter(torch.Tensor(interdim, embedding_dim)) if i == 0
        #                                          else nn.Parameter(torch.Tensor(interdim, embedding_dim))) for i in range(self.layernum)])

        self.linears = nn.ModuleList([nn.Linear(interdim, embedding_dim) if i+1 == self.layernum
                                      else (nn.Linear(embedding_dim, interdim) if i == 0
                                      else nn.Linear(interdim, interdim)) for i in range(self.layernum)])


    # def construct_code2each_token(self):
    #     byte_encodes = self.subword2bytes
    #     return byte_encodes

    def construct_matrix_for_output_layer(self):
        """
        construct the embedding matrix for the decoder output layer
        if we share the embedding between the encoder and decoder, the embedding
        matrix for decoder layer is needed.
        """
        
        matrix = self.forward(self.subword, dropout=0)
        return matrix


    def forward(self, input, dropout=None):
        # print("load table device:", self.subword_bytes_table.device)
        # exit()
        # if input.is_cuda:
        #     print("input is on cuda")
        #     exit()
        # if input.is_cuda and not self.subword_bytes_table.is_cuda:
        #     self.subword_bytes_table.cuda()
        # print("working on cuda:", self.subword_bytes_embed.weight.device)
        # print("working on cuda:", input.device)


        # bytes_input = self.subword2bytes(input)
        # bytes_input = bytes_input.int()




        bytes_input = self.subword2bytes[input]
        # if input.is_cuda and not bytes_input.is_cuda:
        #     bytes_input.cuda()
        # print("working on cuda:", self.subword_bytes_embed.weight.device)
        # exit()
        # print("input size is:", bytes_input.size())


        # padding_matrix = torch.ones_like(bytes_input) * self.padding_idx
        # mask = ~padding_matrix.eq(bytes_input)
        # num_bytes = mask.sum(dim=-1)
        # nb = num_bytes.unsqueeze(dim=-1)
        # nb[nb == 0] = 1

        embeddings = self.byte_embedding(bytes_input)
        # print("working on cuda:", embeddings.is_cuda)
        # exit()

        # print("embeddings size before average:", embeddings.size())
        # print(len(embeddings.shape))
        # exit()
        # if embeddings.shape[0] == self.num_embeddings:
        if len(embeddings.shape) == 3:
            embed = embeddings.sum(dim=1)
            # embed = embed_sum / nb
        else:  
            embed = embeddings.sum(dim=2)
            # embed = embed_sum / nb

        relu_dropout = self.relu_dropout if dropout is None else dropout

        for i in range(self.layernum):
            # embed = nn.functional.linear(embed, self.weight_matrices[i])
            # print("embedding size:", embed.size())
            embed = self.linears[i](embed)
            if i + 1 != self.layernum:
                embed = nn.functional.gelu(embed)
                # embed = nn.functional.dropout(embed, p=relu_dropout, training=self.training)
        return embed


if __name__ == "__main__":
    subword_bytes_file='subword_byte_table.pt'
    if os.path.isfile(subword_bytes_file):
        subword_bytes_table = torch.load(subword_bytes_file)
    else:
        print("please provide the subword to bytes table.")
        exit()
    subword_bytes_table = subword_bytes_table.float()
    # print(subword_bytes_table.size())
    # exit()
    subword_bytes_table.requires_grad = False
    max_byte_seq_len = subword_bytes_table.shape[1]

    if not os.path.isfile(subword_bytes_file):
        # randint(dim0, dim1, (low, high))
        bytesequence = torch.randint(0, 4, (10, 6))
        torch.save(bytesequence, subword_bytes_file)
    bytesequence = torch.load(subword_bytes_file)
    print(bytesequence)

    embed_layer = ByteSubword(10, 3, subword_bytes_file=subword_bytes_file)
    input = torch.LongTensor([[0, 0, 2, 8],[1, 3, 4, 4]])
    embed = embed_layer(input)

    print(embed.size())

    print(embed)
    
    # input = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    # input = torch.LongTensor([[0, 0, 2, 8],[1, 3, 4, 4]])
    
    # subword_bytes_sequence = subword_bytes(input)

