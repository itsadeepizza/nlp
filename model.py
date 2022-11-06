import torch
import torch.nn as nn

embedding_size = 50

class Embedding(nn.Module):
    def __init__(self,device,len_voc):
        super().__init__()
        self.device = device
        self.fc = nn.Linear(len_voc, embedding_size, bias=False)
    def forward(self, x):
        return self.fc(x.to(self.device))


class Context(nn.Module):
    def __init__(self,device,len_voc):
        super().__init__()
        self.device = device
        self.fc = nn.Linear(len_voc, embedding_size, bias=False)
    def forward(self, x):
        return self.fc(x.to(self.device))


class Model(nn.Module):
    def __init__(self, device=torch.device('cpu'), len_voc=None):
        if not len_voc: raise ValueError("len_voc needed")
        super().__init__()
        self.device = device
        self.context = Context(device, len_voc)
        self.embedding = Embedding(device, len_voc)
        self.sigmoid = nn.Sigmoid()
        self.to(self.device)
    def forward(self, x):
        v1 = self.embedding(x.to(self.device).select(1, 0))
        v2 = self.context(x.to(self.device).select(1, 1))
        # qui mi serve solo v1*v2 ma per il batch ho problemi con le dimensioni
        # TODO normalizzare i vettori 
        # res_matrix = self.sigmoid(torch.matmul(v1,torch.transpose(v2,0,1)))

        prods = torch.bmm(v1.unsqueeze(1), v2.unsqueeze(2))
        norm_v1 = torch.bmm(v1.unsqueeze(1), v1.unsqueeze(2))
        norm_v2 = torch.bmm(v2.unsqueeze(1), v2.unsqueeze(2))
        prod_normalised = prods/torch.sqrt(norm_v1*norm_v2)
        return torch.abs(prod_normalised).squeeze(dim=2)

        # return torch.diagonal(res_matrix,0).view(-1,1)

