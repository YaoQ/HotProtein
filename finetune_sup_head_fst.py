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
import logging 
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

    set_seed(args)
    
    # wandb.config.update(vars(args))
    best_accuray = 0
    best_precision = 0
    best_epoch = 0

    model, alphabet = pretrained.load_model_and_alphabet(args.model_location)
    if args.load_pretrained is not None and args.load_pretrained.strip() != "" :
        print(f'loading {args.load_pretrained}')
        state_dict = torch.load(args.load_pretrained, map_location='cpu')['state_dict']
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")

    train_set = PickleBatchedDataset.from_file(args.split_file, True, args.fasta_file)
    test_set = PickleBatchedDataset.from_file(args.split_file, False, args.fasta_file)
    train_data_loader = torch.utils.data.DataLoader(
        train_set, collate_fn=alphabet.get_batch_converter(), batch_size=4, shuffle=True, num_workers=8,
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_set, collate_fn=alphabet.get_batch_converter(), batch_size=4, num_workers=16, #batch_sampler=test_batches
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    return_contacts = "contacts" in args.include

    # Get embed dim to create linear layer
    embed_dim = model.embed_tokens.embedding_dim

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in args.repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers]
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))


    linear = nn.Sequential( nn.Linear(embed_dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Linear(512, args.num_classes)).cuda()
    for name, p in model.named_parameters():
        if 'adapter' in name or 'sparse' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
    
    #for name, p in model.named_parameters():
    #    print(name, p.requires_grad)
    for name, m in model.named_modules():
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
    optimizer1 = torch.optim.AdamW(linear.parameters(), lr=args.lr, weight_decay=5e-2)
    optimizer2 = torch.optim.AdamW(model.parameters(), lr=args.lr / args.lr_factor, weight_decay=5e-2)
    lr_scheduler1 = torch.optim.lr_scheduler.OneCycleLR(optimizer1, max_lr=args.lr, steps_per_epoch=1, epochs=int(20))
    lr_scheduler2 = torch.optim.lr_scheduler.OneCycleLR(optimizer2, max_lr=args.lr / args.lr_factor, steps_per_epoch=1, epochs=int(20))
    step = 0
    log_freq = int(len(train_data_loader)/360)
    for epoch in range(6):
        logger.info('Epoch {}'.format(epoch))
        model.eval()
        for batch_idx, (labels, strs, toks) in enumerate(train_data_loader):
            step += 1

            toks = toks.cuda()
            if args.truncate:
                toks = toks[:, :1022]
            
            out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts, return_temp=True)

            hidden = out['hidden']

            labels = torch.tensor(labels).cuda().long()
            if args.mix:
                lam = np.random.beta(0.2, 0.2)
                rand_index = torch.randperm(hidden.size()[0]).cuda()
                labels_all_a = labels
                labels_all_b = labels[rand_index]
                hiddens_a = hidden
                hiddens_b = hidden[rand_index]
                hiddens = lam * hiddens_a + (1 - lam) * hiddens_b
                hiddens = linear(hiddens)
                loss = F.cross_entropy(hiddens.view(hiddens.shape[0], args.num_classes), labels_all_a) * lam + \
                    F.cross_entropy(hiddens.view(hiddens.shape[0], args.num_classes), labels_all_b) * (1 - lam)
            elif args.adv:
                hidden_adv = PGD_classification(hidden, linear, labels, steps=args.steps, eps=3/255, num_classes=args.num_classes, gamma=args.gamma)
                hiddens_adv = linear(hidden_adv)
                hiddens_clean = linear(hidden)
                loss = (F.cross_entropy(hiddens_adv.view(hiddens_adv.shape[0], args.num_classes), labels) + F.cross_entropy(hiddens_clean.view(hiddens_clean.shape[0], args.num_classes), labels)) / 2
            elif args.aadv:
                hidden_adv = PGD_classification_amino(hidden, linear, labels, steps=args.steps, eps=3/255, num_classes=args.num_classes, gamma=args.gamma)
                hiddens_adv = linear(hidden_adv)
                hiddens_clean = linear(hidden)
                loss = (F.cross_entropy(hiddens_adv.view(hiddens_adv.shape[0], args.num_classes), labels) + F.cross_entropy(hiddens_clean.view(hiddens_clean.shape[0], args.num_classes), labels)) / 2
            else:
                hiddens = linear(hidden)
                loss = F.cross_entropy(hiddens.view(hiddens.shape[0], args.num_classes), labels)

            if batch_idx % log_freq == 0:
                logger.info('Epoch {} {}/{}, Training Loss: {:.4f}'.format(epoch, batch_idx, len(train_data_loader), loss.item()))

            loss.backward()
            optimizer1.step()
            optimizer2.step()
            linear.zero_grad()
            model.zero_grad()

            # Save internal model
            if (step + 1) % args.save_freq == 0:
                model.eval()
                accuracy, precision = evaluate(model, linear, test_data_loader, repr_layers, return_contacts, step)
                if accuracy > best_accuray:
                    torch.save({'linear': linear.state_dict(), 'model': model.state_dict()}, f"{args.output_dir}/{args.output_name}.pt")
                    best_accuray = accuracy
            model.train()

        lr_scheduler1.step()
        lr_scheduler2.step()
        model.eval()
        accuracy, precision = evaluate(model, linear, test_data_loader, repr_layers, return_contacts, step)
        if accuracy > best_accuray:
            torch.save({'linear': linear.state_dict(), 'model': model.state_dict()}, f"{args.output_dir}/{args.output_name}.pt")
            logger.info('Saving model to {}'.format("{}/{}.pt".format(args.output_dir, args.output_name)))
            best_accuray = accuracy
            best_precision = precision
    logger.info('Best Accuracy: {:.4f}'.format(best_accuray * 100))
    logger.info('Best Precision: {:.4f}'.format(best_precision * 100))

def evaluate(model, linear, test_data_loader, repr_layers, return_contacts, step):
    with torch.no_grad():
        outputs = []
        tars = []
        for batch_idx, (labels, strs, toks) in enumerate(test_data_loader):
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
        accuracy = (outputs == tars).float().sum() / tars.nelement()
        precision = ((outputs == tars).float() * (outputs == 1).float()).sum() / (outputs == 1).float().sum()
        logger.info('Accuracy: {:.4f}'.format(accuracy * 100))
        logger.info('Precision: {:.4f}'.format(precision * 100))
        return accuracy, precision

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
