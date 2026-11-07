import math
import os
import torch
import torch.nn as nn 
import torch.nn.functional as F

from torch.nn.parameter import Parameter

class ByteSubwordConcateOnehot(nn.Module):
    """
    num_embeddings: dictionary size
    """
    def __init__(self, num_embeddings, embedding_dim, interdim='', relu_dropout=0.1, aggre='avg', subword_bytes_file='', embeddings_file='./embeddings') -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        self.aggre = aggre
        self.relu_dropout = relu_dropout

        # print(num_embeddings)
        # print(padding_idx)

        self.byte_dict_size = 256
        self.max_byte_seq_len = 8
        

        self.padding_idx = 1
        self.embedding_file = embeddings_file
        self.subword_bytes_file = subword_bytes_file

        self.W_a = Parameter(torch.Tensor(embedding_dim, 1))
        
        if os.path.isfile(subword_bytes_file):
            subword2bytes = torch.load(subword_bytes_file)
        else:
            subword2bytes = torch.randint(0, self.byte_dict_size, (num_embeddings, self.max_byte_seq_len), dtype=torch.long)
            dirname = '/'.join(subword_bytes_file.split('/')[:-1])
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            torch.save(subword2bytes, subword_bytes_file)

        # print(subword2bytes.shape)
        # exit()
        
        self.register_buffer('subword2bytes', subword2bytes)


        onehot_byte = F.one_hot(torch.arange(0, self.byte_dict_size)).to(torch.float32)
        # if os.path.isfile(self.embedding_file):
        #     weight = torch.load(self.embedding_file)
        #     # print(weight.shape)
        #     # exit()
        #     self.byte_embedding = nn.Embedding(self.byte_dict_size, embedding_dim, _weight=weight, padding_idx=self.byte_dict_size)
        # else:
        self.byte_embedding = nn.Embedding(self.byte_dict_size, self.byte_dict_size, _weight=onehot_byte, padding_idx=None)
        self.byte_embedding.requires_grad_(False)

        self.weight = self.byte_embedding.weight
        

        if interdim == []:
            interdim = [embedding_dim]

        
        subword = torch.tensor([i for i in range(self.num_embeddings)])
        self.register_buffer('subword', subword)

        if type(interdim) == int:
            interdim = str(interdim)

        # print(type(interdim))
        # exit()

        # print(interdim)
        # exit()
        interdim = [int(i) for i in interdim.split('_')]

        interdim = [self.byte_dict_size * self.max_byte_seq_len] + interdim + [embedding_dim]
        
        self.layernum = len(interdim) - 1

        layers = []

        # print(self.layernum)
        # exit()
        for i in range(1, len(interdim)):
            layers.append(nn.Linear(interdim[i - 1], interdim[i]))
        #     print(layers[i-1].weight.dtype)
        # exit()
        
        self.linears = nn.ModuleList(layers)
        # print(self.linears)
        # exit()


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
        # print("forward working here")


        bytes_input = self.subword2bytes[input]

        embeddings = self.byte_embedding(bytes_input)

        relu_dropout = self.relu_dropout if dropout is None else dropout
        
        emb_dim = embeddings.size()
        

        # print(bz, nw)
        # exit()

        if len(emb_dim) == 4:
            bz = emb_dim[0]
            nw = emb_dim[1]
            embed = embeddings.reshape((bz, nw, -1))
            # print(embed.size())
            # exit()
        
        else:
            nw = emb_dim[0]
            embed = embeddings.reshape((nw, -1))

        # print(embed.dtype)
        # print(self.linears[0].weight.dtype)
        # exit()
        # print(self.layernum)
        # exit()
        for i in range(self.layernum):
            # print(embed.size())
            # print(embed.dtype)
            # print(self.linears[i].weight.dtype)
            # exit()
            embed = self.linears[i](embed)
            if i + 1 != self.layernum:
                embed = nn.functional.relu(embed)
                embed = nn.functional.dropout(embed, p=relu_dropout, training=self.training)
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

    embed_layer = ByteSubwordConcateOnehot(10, 3, subword_bytes_file=subword_bytes_file)
    input = torch.LongTensor([[0, 0, 2, 8],[1, 3, 4, 4]])
    embed = embed_layer(input)

    print(embed.size())

    print(embed)
    
    # input = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    # input = torch.LongTensor([[0, 0, 2, 8],[1, 3, 4, 4]])
    
    # subword_bytes_sequence = subword_bytes(input)

