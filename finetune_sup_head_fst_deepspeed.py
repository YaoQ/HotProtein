#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pathlib
import random
import numpy as np
import torch
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import numpy as np
import time
import logging 
import deepspeed
from deepspeed.accelerator import get_accelerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, CSVBatchedDataset, creating_ten_folds, PickleBatchedDataset, FireprotDBBatchedDataset
from esm.modules import TransformerLayer, SparseMultiheadAttention
from esm.model.esm2cls import ESM2CLS
import esm
from tqdm import tqdm
from esm.utils import PGD_classification, PGD_classification_amino


# Initialize DeepSpeed to use the following features
# 1) Distributed model
# 2) Distributed data loader
# 3) DeepSpeed optimizer
ds_config = {
  "train_batch_size": 4,
  "steps_per_print": 1000000,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 1e-7,
      "betas": [
        0.8,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 5e-2
    }
  },
  "comms_logger": {
  "enabled": False,
  "verbose": False,
  "prof_all": False,
  "debug": False
},
  "scheduler": {
    "type": "OneCycle",
    "params": {
        "cycle_first_step_size": 100,
        "cycle_first_stair_count": 50,
        "cycle_second_step_size": 100,
        "cycle_second_stair_count": 50,
        "decay_step_size": 100,
        "cycle_min_lr": 0.0001,
        "cycle_max_lr": 1e-6,
        "decay_lr_rate": 0.001,
        "cycle_min_mom": 0.85,
        "cycle_max_mom": 0.99,
        "decay_mom_rate": 0.0
    }
  },
  "gradient_clipping": 1.0,
  "prescale_gradients": False,
  "bf16": {
      "enabled": False
  },
  "fp16": {
      "enabled": False ,
      "fp16_master_weights_and_grads": False,
      "loss_scale": 0,
      "loss_scale_window": 500,
      "hysteresis": 2,
      "min_loss_scale": 1,
      "initial_scale_power": 15
  },
  "wall_clock_breakdown": False,
  "zero_optimization": {
      "stage": 0,
      "allgather_partitions": True,
      "reduce_scatter": True,
      "allgather_bucket_size": 50000000,
      "reduce_bucket_size": 50000000,
      "overlap_comm": True,
      "contiguous_gradients": True,
      "offload_optimizer": {
        "device": "none"
      }
  }
}

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
         default=6,
         type=int,
         help='number of total epochs (default: 6)')
    parser.add_argument('--local_rank',
        type=int,
        default=-1,
        help='local rank passed from distributed launcher'
    )
    parser.add_argument(
        '--dtype',
        default='fp16',
        type=str,
        choices=['bf16', 'fp16', 'fp32'],
        help=
        'Datatype used for training'
    )
    parser.add_argument(
        '--stage',
        default=1,
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
    parser.add_argument("--save-freq", type=int, default=200)
    parser.add_argument("--sparse", type=int, default=64)
    parser.add_argument("--lr-factor", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=1e-3)
    parser.add_argument("--wandb-name", type=str, default='protein')
    parser.add_argument("--output-name", type=str, default='protein')
    parser.add_argument("--load-pretrained", type=str, default='sap.pt')
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

def main(parser):

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    set_seed(args)

    deepspeed.init_distributed()
    
    # wandb.config.update(vars(args))
    best_accuray = 0
    best_precision = 0
    best_epoch = 0

    # Load the model and alphabet
    model_name = args.model_location
    model_path = f"{torch.hub.get_dir()}/checkpoints/{model_name}.pt"
    model_data = torch.load(model_path, map_location="cpu")
    mse_model = ESM2CLS(
        num_classes=args.num_classes,
        state_dict=model_data
    )    

    alphabet = esm.data.Alphabet.from_architecture("ESM-1b")

    train_set = PickleBatchedDataset.from_file(args.split_file, True, args.fasta_file)
    test_set = PickleBatchedDataset.from_file(args.split_file, False, args.fasta_file)
    train_data_loader = torch.utils.data.DataLoader(
        train_set, collate_fn=alphabet.get_batch_converter(), batch_size=4, shuffle=True, num_workers=8,
        pin_memory=False
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_set, collate_fn=alphabet.get_batch_converter(), batch_size=4, num_workers=16, #batch_sampler=test_batches
        pin_memory=False
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, m in mse_model.named_modules():
        if isinstance(m, SparseMultiheadAttention):
            Q_weight = m.q_proj.weight
            V_weight = m.v_proj.weight
            Q_weight = Q_weight.detach().cpu()
            V_weight = V_weight.detach().cpu()
            U_Q = torch.randn((Q_weight.shape[0], 1)).to(Q_weight.device)
            V_Q = torch.randn((1, Q_weight.shape[1])).to(Q_weight.device)
            S_Q = torch.zeros_like(Q_weight)

            U_V = torch.randn((V_weight.shape[0], 1)).to(V_weight.device)
            V_V = torch.randn((1, V_weight.shape[1])).to(V_weight.device)
            S_V = torch.zeros_like(V_weight)
            last_S_Q = torch.zeros_like(Q_weight)

            for rank in tqdm(range(20)):
                S_Q = torch.zeros_like(Q_weight)
                S_V = torch.zeros_like(Q_weight)
                for _ in range(10):
                    U_Q = torch.qr((Q_weight - S_Q) @ V_Q.T)[0]
                    V_Q = U_Q.T @ (Q_weight - S_Q)
                    S_Q = Q_weight - U_Q @ V_Q
                    q = 0.01
                    S_Q[S_Q.abs() < q] = 0
                    U_V = torch.qr((V_weight - S_V) @ V_V.T)[0]
                    V_V = U_V.T @ (V_weight - S_V)
                    S_V = V_weight - U_V @ V_V
                    S_V[S_V.abs() < q] = 0

                E_Q = Q_weight - U_Q @ V_Q - S_Q
                E_V = V_weight - U_V @ V_V - S_V
                
                E_Q_vector = torch.qr(E_Q)[1][:1]
                E_V_vector = torch.qr(E_V)[1][:1]
                
                V_Q = torch.cat([V_Q, E_Q_vector], 0)
                V_V = torch.cat([V_V, E_V_vector], 0)
            
            q, _ = torch.kthvalue(S_Q.abs().view(-1), S_Q.numel() - args.sparse)
            S_Q = (S_Q.abs() >= q).float()
            #print(S_Q)
            v, _ = torch.kthvalue(S_V.abs().view(-1), S_V.numel() - args.sparse)
            S_V = (S_V.abs() >= v).float()
            prune.custom_from_mask(m.q_proj_sparse, 'weight', S_Q.to(m.q_proj.weight.device))
            prune.custom_from_mask(m.v_proj_sparse, 'weight', S_V.to(m.v_proj.weight.device))


    train_data_loader = iter(deepspeed.utils.RepeatingLoader(train_data_loader))

    engine, _, _, _ = deepspeed.initialize(model=mse_model,
                                            model_parameters=[p for p in mse_model.parameters() if p.requires_grad], 
                                            config=ds_config)

    local_device = get_accelerator().device_name(engine.local_rank)
    local_rank = engine.local_rank

    step = 0
    epoch_num = 6
    #whole_data = epoch_num * len(train_data_loader)
    for epoch in range(epoch_num):
        for batch_idx, data in enumerate(train_data_loader):
            labels, strs, toks = data
            step += 1
            data = toks.clone().to(local_device)
            labels = torch.tensor(labels).to(local_device).long()

            if args.mix:
                hiddens, labels_all_a, labels_all_b,lam  = engine(data,labels, args)
                loss = F.cross_entropy(hiddens.view(hiddens.shape[0], args.num_classes), labels_all_a) * lam + \
                    F.cross_entropy(hiddens.view(hiddens.shape[0], args.num_classes), labels_all_b) * (1 - lam)
            elif args.adv:
                hiddens_adv, hiddens_clean = engine(data, labels, args)
                loss = (F.cross_entropy(hiddens_adv.view(hiddens_adv.shape[0], args.num_classes), labels) + F.cross_entropy(hiddens_clean.view(hiddens_clean.shape[0], args.num_classes), labels)) / 2
            elif args.aadv:
                hiddens_adv, hiddens_clean = engine(data, labels, args)
                loss = (F.cross_entropy(hiddens_adv.view(hiddens_adv.shape[0], args.num_classes), labels) + F.cross_entropy(hiddens_clean.view(hiddens_clean.shape[0], args.num_classes), labels)) / 2
            else:
                hiddens = engine(data)
                loss = F.cross_entropy(hiddens.view(hiddens.shape[0], args.num_classes), labels)

            engine.backward(loss)
            engine.step()
            if (step + 1) % args.save_freq == 0:
                logger.info('Loss: {:.4f}'.format(loss))
                engine.eval()
                accuracy, precision = evaluate(engine,test_data_loader, local_device, args)
                if accuracy > best_accuray:
                    engine.module.save_state_dict("{}/{}.pt".format(args.output_dir, args.output_name))
                    #engine.save_checkpoint("result.pt")
                    #engine.save_checkpoint(f"{args.output_dir}/{args.output_name}.pt")
                    logger.info('Saving model to {}'.format("{}/{}.pt".format(args.output_dir, args.output_name)))
                    best_accuray = accuracy

    logger.info('Best Accuracy: {:.4f}'.format(best_accuray * 100))
    #logger.info('Best Precision: {:.4f}'.format(best_precision * 100))

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
    main(parser)
