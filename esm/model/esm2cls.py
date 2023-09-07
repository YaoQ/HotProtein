# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import esm
from esm.model.esm2 import ESM2
from deepspeed.pipe import PipelineModule, TiedLayerSpec, LayerSpec
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
            self.cfg = self.state_dict["cfg"]
        else:
            self.state_dict = None
            print("None pretrain model found.")
            return

        self.alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        self.num_layers= self.cfg['model'].encoder_layers
        self.embed_dim = self.cfg['model'].encoder_embed_dim
        self.encoder_attention_heads = self.cfg['model'].encoder_attention_heads
        self.token_dropout = self.cfg['model'].token_dropout
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
        
        # Load the state_dict of the model
        self.load_state_dict()

    def forward(self, tokens):
        out = self.esm2(tokens, repr_layers=self.repr_layers, return_contacts=False, return_temp=True)
        hidden = out['hidden']
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
