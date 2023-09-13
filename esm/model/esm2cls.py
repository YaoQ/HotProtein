# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import torch
import numpy as np
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import esm
from esm.model.esm2 import ESM2
from deepspeed.pipe import PipelineModule, TiedLayerSpec, LayerSpec
from esm.utils import PGD_classification, PGD_classification_amino
import re
# Create ESM
class ESM2CLS(nn.Module):
    # Initialize the ESM2CLS model
    def __init__(self, num_classes=2, state_dict=None):
        super().__init__()
        # Set the number of classes
        self.num_classes = num_classes
        # Set the esm2cls_state_dict
        # If the esm_state_dict is not None, set the model_data to the esm_state_dict
        # Load for esm state dict to train
        if state_dict is not None:
            self.state_dict = state_dict
            if "cfg" in self.state_dict.keys():
                self.cfg = self.state_dict["cfg"]
                self.num_layers= self.cfg['model'].encoder_layers
                self.embed_dim = self.cfg['model'].encoder_embed_dim
            else: 
                self.cfg = None
                self.num_layers= self.state_dict['model']['embed_tokens.weight'].shape[0]
                self.embed_dim = self.state_dict['model']['embed_tokens.weight'].shape[1]
        else:
            self.state_dict = None
            print("None pretrain model found.")
            return

        self.alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        self.encoder_attention_heads = 20 
        self.token_dropout = True
        self.repr_layers = [self.num_layers]

        # Create an ESM2 model
        self.esm2 = ESM2(
            num_layers=self.num_layers,
            embed_dim=self.embed_dim,
            attention_heads=self.encoder_attention_heads,
            alphabet=self.alphabet,
            token_dropout=self.token_dropout,
        )

        # Create a linear layer
        self.linear = nn.Sequential( nn.Linear(self.embed_dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Linear(512, self.num_classes))
        

        # Set all esm2 parameters to not require gradients
        for param in self.esm2.parameters():
            param.requires_grad = False
        
        self.load_state_dict()

    def forward(self, tokens, labels=None, args= None):
        out = self.esm2(tokens, repr_layers=self.repr_layers, return_contacts=False, return_temp=True)
        hidden = out['hidden']

        ## addd Adversarial Feature Augmentation
        if args is not None:
            if args.mix:
                lam = np.random.beta(0.2, 0.2)
                rand_index = torch.randperm(hidden.size()[0]).cuda()
                labels_all_a = labels
                labels_all_b = labels[rand_index]
                hiddens_a = hidden
                hiddens_b = hidden[rand_index]
                hiddens = lam * hiddens_a + (1 - lam) * hiddens_b
                result = self.linear(hiddens)
                return result, labels_all_a, labels_all_b, lam
            elif args.adv:
                hidden_adv = PGD_classification(hidden, self.linear, labels, steps=1, eps=3/255, num_classes=self.num_classes, gamma=1e-3)
                hiddens_adv = self.linear(hidden_adv)
                hiddens_clean = self.linear(hidden)
                return hiddens_adv, hiddens_clean 
            elif args.aadv:
                hidden_adv = PGD_classification_amino(hidden, self.linear, labels, steps=1, eps=3/255, num_classes=self.num_classes, gamma=1e-3)
                hiddens_adv = self.linear(hidden_adv)
                hiddens_clean = self.linear(hidden)
                return hiddens_adv, hiddens_clean 
        
        result = self.linear(hidden)
        return result
        
    def upgrade_state_dict(self, state_dict):
        """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
        prefixes = ["encoder.sentence_encoder.", "encoder."]
        pattern = re.compile("^" + "|".join(prefixes))
        state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
        return state_dict

    def save_state_dict(self, path):
        # Create a dictionary containing the state of the model and configs.
        state_dict = {  
            'esm': self.esm2.state_dict(),  
            'linear': self.linear.state_dict(),
            'cfg': self.cfg
        }  
        # Save the state dictionary to the specified path
        torch.save(state_dict, path)
        # Return the state dictionary
        return state_dict 

    def load_state_dict(self):
        # Load the state dictionary from the specified path
        if self.state_dict is not None:
            if 'linear' in self.state_dict.keys():
                self.esm2.load_state_dict(self.state_dict['esm'], strict=False)
                self.linear.load_state_dict(self.state_dict['linear'], strict=False)
            else:
                self.esm2.load_state_dict(self.state_dict, strict=False)
