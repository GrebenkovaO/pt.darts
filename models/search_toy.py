""" CNN for architecture search """
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.search_cells import SearchCell
import genotypes as gt
from torch.nn.parallel._functions import Broadcast
from torch.nn import Linear, Parameter
import logging
import torch as t

class SearchToy(nn.Module):
    """ Search CNN model """
    def __init__(self):
        super().__init__()
        self.model1 = nn.Parameter(t.ones(2))
        self.model2 = Linear(1,2)
        self.model3 = Linear(25,2)
        


    def forward(self, x,  alpha):
        r1 = self.model1
        r2 = self.model2(x[:,:1])
        r3 = self.model3(x)
        logits = r1*alpha[0]+r2*alpha[1]+r3*alpha[2]
        return logits
        


class SearchToyController(nn.Module):
    """ SearchCNN controller supporting multi-gpu """
    def __init__(self,  criterion):
        super().__init__()
        self.criterion = criterion

        self.alphas_ = nn.Parameter(1e-3*torch.randn(3))
        self.net = SearchToy()


    def forward(self, x):
        weights_normal = F.softmax(self.alphas_, dim=-1)

        return self.net(x, weights_normal)


    def loss(self, X, y):
        logits = self.forward(X)
        return self.criterion(logits, y)

    def alphas(self):
    	return [self.alphas_]
    	
    def print_alphas(self, logger):
        
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### ALPHA #######")
        logger.info(F.softmax(self.alphas_, dim=-1))


        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    
    def weights(self):
        return self.net.parameters()

    
    
