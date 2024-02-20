import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

# =====================================================================
# from seqeval.metrics import accuracy_score
# from seqeval.metrics import classification_report
# from seqeval.scheme import IOB2
# import numpy as np

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def oneEpochTrain(model, optimizer, train_loader):
    model.train()
    total_train_step = 0

    loss_sum = 0
    loss_count = 0

    correctTokenNum = 0
    allCorrectNum = 0
    TokenSize = 0
    allSize = 0

    bar = tqdm(train_loader)
    for i, batch in enumerate(bar):
        batch['tokens'] = batch['tokens'].to(args.device) # [batch_size * max_len]
        batch['tags'] = batch['tags'].to(args.device) # [batch_size]
        batch['NotPad'] = batch['NotPad'].to(args.device) # [batch_size * seq_len]
        output = model(batch)
        # ===================================================================== training mertic
        thisLoss = output['loss']
        # 是算整個batch的loss
        thisLossCount = batch['tokens'].size(0)
        loss_sum += thisLoss
        loss_count += thisLossCount
        maxLen = output['pred'].shape[1]
        batch['tags'] = batch['tags'][:, :maxLen]
        batch['NotPad'] = batch['NotPad'][:, :maxLen]
        # 在長度內的預測對的數量
        correctBatchNum = (batch['tags'].eq(output['pred'].view_as(batch['tags'])) * batch['NotPad']).sum(-1)
        # 實際有東西的長度
        seq_len = batch['len'] # [batch_size]
        correctTokenNum += correctBatchNum.sum().item()
        # 全對的個數
        allCorrectNum += correctBatchNum.eq(seq_len.to(args.device)).sum().item()
        TokenSize += batch['NotPad'].sum().item()
        # batch數
        allSize += batch['tokens'].shape[0]
        bar.set_postfix(loss=output['loss'].item(), iter=i, lr=optimizer.param_groups[0]['lr'])

       # ===========================================================================
        loss = output['loss']
        optimizer.zero_grad()
        loss.backward()
        # if args.grad_clip > 0.0:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()
        total_train_step += 1
    accRate = correctTokenNum / (TokenSize + 1E-8)
    joiAccRate = allCorrectNum / (allSize + 1E-8)
    avg_loss = loss_sum / loss_count
    print('Train Loss: {:6.6f} All Acc: {:6.6f} ({}/{}) Token Acc: {:6.6f} ({}/{})'.format(avg_loss, joiAccRate, allCorrectNum, allSize,
                                                                                                accRate, correctTokenNum, TokenSize))
 

def oneEpochVal(model, dev_loader, ytrueList, ypredList):
    model.eval()
    totalAccCount = 0
    totalSize = 0
    correctTokenNum = 0
    allCorrectNum = 0
    TokenSize = 0
    allSize = 0
    loss_sum = 0
    loss_count = 0


    with torch.no_grad():
        for batch in dev_loader:
            batch['tokens'] = batch['tokens'].to(args.device)
            batch['tags'] = batch['tags'].to(args.device)
            batch['NotPad'] = batch['NotPad'].to(args.device)
            output = model(batch)
            # ====================================================================
            # ytrueList += output['y_true']
            # ypredList += output['y_pred']
            # =====================================================================

            thisLoss = output['loss']
            # 是算整個batch的loss
            thisLossCount = batch['tokens'].size(0)
            loss_sum += thisLoss
            loss_count += thisLossCount
            # ========================================================================
            maxLen = output['pred'].shape[1]
            batch['tags'] = batch['tags'][:, :maxLen]
            batch['NotPad'] = batch['NotPad'][:, :maxLen]
            # ========================================================================for seq

            # predList.append(output['pred'][i][:batch['len'][i]] for i in range(len(batch['len'])))
            # tagList.append(batch['tags'][i][:batch['len'][i]]for i in range(len(batch['len'])))

            # ========================================================================for seq
            # 在長度內的預測對的數量
            correctBatchNum = (batch['tags'].eq(output['pred'].view_as(batch['tags'])) * batch['NotPad']).sum(-1)
            # 實際有東西的長度
            seq_len = batch['len'] # [batch_size]
            correctTokenNum += correctBatchNum.sum().item()
            # 全對的個數
            allCorrectNum += correctBatchNum.eq(seq_len.to(args.device)).sum().item()
            TokenSize += batch['NotPad'].sum().item()
            # batch數
            allSize += batch['tokens'].shape[0]
            # ===========================================================================
            # 問題-> 填補的0也會被算為預測正確，應紀錄實際長度並只用實際長度判斷準確率
        accRate = correctTokenNum / (TokenSize + 1E-8)
        joiAccRate = allCorrectNum / (allSize + 1E-8)
        avg_loss = loss_sum / loss_count

    print('Val Loss: {:6.6f} All Acc: {:6.6f} ({}/{}) Token Acc: {:6.6f} ({}/{})'.format(avg_loss, joiAccRate, allCorrectNum, allSize,
                                                                                                accRate, correctTokenNum, TokenSize))
    print('Joint Acc: {:6.6f} ({}/{}) Token Acc: {:6.6f} ({}/{})'.format(joiAccRate, allCorrectNum, allSize,
                                                                                                accRate, correctTokenNum, TokenSize))

    return joiAccRate, avg_loss

def saveCKPT(model, epoch, best):

    if best:
        ckpt_path = args.ckpt_dir / "bestModelSlot.pth"
        torch.save(model, ckpt_path)
        print("Best Model Save")
    else:
        ckpt_path = args.ckpt_dir / "{} ModelSlot.pth".format(epoch)
        torch.save(model, ckpt_path)
        print("{} Model Save:".format(epoch))


def main(args):
    # TODO: implement main function
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }
    dataLoader: Dict[str, SeqTaggingClsDataset] = {
        split: DataLoader(dataset, args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
        for split, dataset in datasets.items()
    }
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqTagger(embeddings=embeddings, hidden_size=args.hidden_size, num_layers=args.num_layers,
                      dropout=args.dropout, bidirectional=args.bidirectional, num_class=datasets[TRAIN].num_classes
                      ).to(args.device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor=0.1, patience=2)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    best_acc = 0

    for epoch in epoch_pbar:
        oneEpochTrain(model, optimizer, dataLoader[TRAIN])
        ytrueList = []
        ypredList = []
        this_acc, avg_loss = oneEpochVal(model, dataLoader[DEV], ytrueList, ypredList)
        # print(ytrueList)
        # print(ypredList)

        # print(0)
        # print(ytrueList[0])
        # print(1)
        # print(ytrueList[1])
        # ytrueList3 = []
        # ypredList3 = []
        # for ytrueSmallList, ypredSmallList in zip(ytrueList, ypredList):
        #     ytrueList2 = []
        #     ypredList2 = []
        #     for ytrue, ypred in zip(ytrueSmallList, ypredSmallList):
        #         # print(ytrue)
        #         ytrueList2.append(datasets[TRAIN].idx2label(int(ytrue)))
        #         ypredList2.append(datasets[TRAIN].idx2label(int(ypred)))
        #     ytrueList3.append(ytrueList2)
        #     ypredList3.append(ypredList2)
        # print(ypredList3)

        # ytrueList = [[datasets[TRAIN].idx2label(int(ytrue))for ytrue in ytrueSmallList] for ytrueSmallList in ytrueList]
        # ypredList = [[datasets[TRAIN].idx2label(int(ypred))for ypred in ypredSmallList] for ypredSmallList in ypredList]
        # print(ytrueList)
        scheduler.step(avg_loss)
        saveCKPT(model, epoch, False)
        if this_acc > best_acc:
            best_acc = this_acc
            saveCKPT(model, epoch, True)
            # f1_score(tagList, predList)
            # print("seqeval classification report: ")
            # print(classification_report(ytrueList3, ypredList3,scheme=IOB2, mode="strict"))

    




def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=50)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # # loss
    # parser.add_argument('--grad_clip', default = 5., type=float, help='max gradient norm')

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=50)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)