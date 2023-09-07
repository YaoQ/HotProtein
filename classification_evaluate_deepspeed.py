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
from esm.model.esm2cls import ESM2CLS
import deepspeed
import logging 
from deepspeed.accelerator import get_accelerator
import numpy as np
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, CSVBatchedDataset, creating_ten_folds, PickleBatchedDataset, FireprotDBBatchedDataset
from esm.modules import TransformerLayer, SparseMultiheadAttention
from tqdm import tqdm
from esm.utils import PGD_classification, PGD_classification_amino

def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract per-token representations and model outputs for sequences in a FASTA file"  # noqa
    )

    parser.add_argument(
        "--model_location",
        type=str,
        help="PyTorch model file OR name of pretrained model to download (see README for models)",
    )
    parser.add_argument(
        "--fasta_file",
        type=pathlib.Path,
        help="FASTA file on which to extract representations",
    )
    parser.add_argument(
        "--output_dir",
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
    parser.add_argument('-e',
         '--epochs',
         default=30,
         type=int,
         help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank',
        type=int,
        default=-1,
        help='local rank passed from distributed launcher'
    )
    parser.add_argument(
        '--dtype',
        default='fp32',
        type=str,
        choices=['bf16', 'fp16', 'fp32'],
        help=
        'Datatype used for training'
    )
    parser.add_argument(
        '--stage',
        default=3,
        type=int,
        choices=[0, 1, 2, 3],
        help=
        'Datatype used for training'
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

ds_config = {
    "kernel_inject": True,
    "tensor_parallel": {"tp_size": 4},
    "dtype": "fp32",
    "enable_cuda_graph": False
}

def main(args):

    # Load pretiraned model
    model, alphabet = pretrained.load_model_and_alphabet(args.model_location)
    model_name = args.model_location
    embed_dim = model.embed_tokens.embedding_dim

    #deepspeed.init_distributed()

    pretrain_path = "%s/%s.pt"%(args.output_dir, args.output_name)
    pretrain_data = torch.load(pretrain_path, map_location="cpu")
    mse_model = ESM2CLS(
        num_classes=args.num_classes,
        state_dict=pretrain_data
    )    
    engine = deepspeed.init_inference(model=mse_model,mp_size=1)
    local_device = torch.device(f'cuda:0')
    # Evaluate on test set for 10 folds
    mean_accuracy = 0;
    mean_precision = 0;
    FOLDS_NUM = 10
    for i in range(FOLDS_NUM):
        split_file= "d1/d1_%d_classification.pkl"%(i)
        print("\nEvaluating %s ..."%(split_file))
        test_set = PickleBatchedDataset.from_file(split_file, False, args.fasta_file)
        test_data_loader = torch.utils.data.DataLoader(
            test_set, collate_fn=alphabet.get_batch_converter(), batch_size=4, num_workers=8 
        )
        accuracy, precision = evaluate(engine, test_data_loader, local_device, args)

        mean_accuracy += accuracy
        mean_precision += precision

    print("\n%d folds evaluation result"%(FOLDS_NUM))
    print("mean accuracy: ", mean_accuracy.item()/FOLDS_NUM)
    print("mean precision: ", mean_precision.item()/FOLDS_NUM)

def evaluate(engine, test_data_loader, local_device, args):
    with torch.no_grad():
        outputs = []
        tars = []
        for batch_idx, (labels, strs, toks) in enumerate(test_data_loader):
            toks = toks.to(local_device)
            out = engine(toks)
            labels = torch.tensor(labels).cuda().long()
            outputs.append(torch.topk(out.reshape(-1, args.num_classes), 1)[1].view(-1))
            tars.append(labels.reshape(-1))
        
        outputs = torch.cat(outputs, 0)
        tars = torch.cat(tars, 0)
        accuracy = (outputs == tars).float().sum() / tars.nelement()
        precision = ((outputs == tars).float() * (outputs == 1).float()).sum() / (outputs == 1).float().sum()
        logger.info('Accuracy: {:.4f}'.format(accuracy * 100))
        #logger.info('Precision: {:.4f}'.format(precision * 100))
        return accuracy, precision

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
