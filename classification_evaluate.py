#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pathlib
import random
from sched import scheduler
import numpy as np
import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, CSVBatchedDataset, creating_ten_folds, PickleBatchedDataset, FireprotDBBatchedDataset
from esm.modules import TransformerLayer, SparseMultiheadAttention
from tqdm import tqdm
from esm.utils import PGD_classification, PGD_classification_amino

def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract per-token representations and model outputs for sequences in a FASTA file"  # noqa
    )

    parser.add_argument(
        "model_location",
        type=str,
        help="PyTorch model file OR name of pretrained model to download (see README for models)",
    )
    parser.add_argument(
        "fasta_file",
        type=pathlib.Path,
        help="FASTA file on which to extract representations",
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        help="output directory for extracted representations",
    )

    parser.add_argument("--toks_per_batch", type=int, default=4096, help="maximum batch size")
    parser.add_argument(
        "--repr_layers",
        type=int,
        default=[-1],
        nargs="+",
        help="layers indices from which to extract representations (0 to num_layers, inclusive)",
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="+",
        choices=["mean", "per_tok", "bos", "contacts"],
        help="specify which representations to return",
        required=True,
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate sequences longer than 1024 to match the training setup",
    )

    parser.add_argument(
        "--split_file",
        type=str,
        help="fold",
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        help="num_classes",
        default=2, 
    )

    parser.add_argument(
        "--lr",
        type=float,
        help="learning rates",
        default=1e-6, 
    )

    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mix", action="store_true")
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--adv", action="store_true")
    parser.add_argument("--aadv", action="store_true")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--sparse", type=int, default=64)
    parser.add_argument("--lr-factor", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=1e-3)
    parser.add_argument("--wandb-name", type=str, default='protein')
    parser.add_argument("--output-name", type=str, default='protein')
    parser.add_argument("--load-pretrained", type=str, default=None)
    return parser

def pruning_model(model, px):
    

    print('start unstructured pruning for all conv layers')
    parameters_to_prune =[]
    for name, m in model.named_modules():
        #if 'self_attn' in name and (not 'k' in name) and isinstance(m, nn.Linear):
        #    print(f"Pruning {name}")
        #    parameters_to_prune.append((m,'weight'))
        if isinstance(m, TransformerLayer):
            print(f"Pruning {name}.fc1")
            parameters_to_prune.append((m.fc1,'weight'))
            print(f"Pruning {name}.fc2")
            parameters_to_prune.append((m.fc2,'weight'))

    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )

def set_seed(args):
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def main(args):

    # Load pretiraned model
    model, alphabet = pretrained.load_model_and_alphabet(args.model_location)
    embed_dim = model.embed_tokens.embedding_dim
    linear = nn.Sequential( nn.Linear(embed_dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Linear(512, args.num_classes)).cuda()
    checkpoint = torch.load(f"{args.output_dir}/{args.output_name}.pt", map_location='cpu')
    print(checkpoint.keys())
    linear.load_state_dict(checkpoint['linear'])

    # Try to make HotProtein and ESM2CLS two networks compatible.
    if 'ems' in checkpoint.keys():
        model.load_state_dict(checkpoint['esm']) 
    else:
        model.load_state_dict(checkpoint['model'])
    
    model.eval()
    linear.eval()

    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        linear = linear.cuda()
        print("Transferred model to GPU")

    return_contacts = "contacts" in args.include

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in args.repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers]
    
    # Evaluate on test set for 10 folds
    mean_accuracy = 0;
    mean_precision = 0;

    # d1 : s2c2
    # d2 : s2c5
    dataset_type = args.split_file.split("/")[0]

    FOLDS_NUM = 10
    for i in range(FOLDS_NUM):
        split_file= "%s/%s_%d_classification.pkl"%(dataset_type, dataset_type, i)
        print("\nEvaluating %s ..."%(split_file))
        test_set = PickleBatchedDataset.from_file(split_file, False, args.fasta_file)
        test_data_loader = torch.utils.data.DataLoader(
            test_set, collate_fn=alphabet.get_batch_converter(), batch_size=4, num_workers=8 
        )

        accuracy, precision = evaluate(model, linear, test_data_loader, repr_layers, return_contacts)
        mean_accuracy += accuracy
        mean_precision += precision

    print("\n%d folds evaluation result"%(FOLDS_NUM))
    print("mean accuracy: ", mean_accuracy.item()/FOLDS_NUM)
    #print("mean precision: ", mean_precision.item()/FOLDS_NUM)

def evaluate(model, linear, test_data_loader, repr_layers, return_contacts):
    with torch.no_grad():
        outputs = []
        tars = []
        for batch_idx, (labels, strs, toks) in tqdm(enumerate(test_data_loader)):
            if torch.cuda.is_available() and not args.nogpu:
                toks = toks.to(device="cuda", non_blocking=True)
            # The model is trained on truncated sequences and passing longer ones in at
            # infernce will cause an error. See https://github.com/facebookresearch/esm/issues/21
            if args.truncate:
                toks = toks[:, :1022]
            out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts, return_temp=True)
            hidden = out['hidden']
            logits = linear(hidden)
            labels = torch.tensor(labels).cuda().long()
            outputs.append(torch.topk(logits.reshape(-1, args.num_classes), 1)[1].view(-1))
            tars.append(labels.reshape(-1))
        
        outputs = torch.cat(outputs, 0)
        tars = torch.cat(tars, 0)
        acc = (outputs == tars).float().sum() / tars.nelement()
        precision = ((outputs == tars).float() * (outputs == 1).float()).sum() / (outputs == 1).float().sum()
        print(f"Precision: {precision}")
        print(f"Accuracy: {acc}")
        return acc, precision

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
