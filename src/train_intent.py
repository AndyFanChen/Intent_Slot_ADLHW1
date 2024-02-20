# DL環境
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate DLEnv

import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange

from dataset import SeqClsDataset
from utils import Vocab

# ================================================================
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from model import SeqClassifier

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

"""
取資料轉torch進模型訓練
"""

def oneEpochTrain(model, optimizer, train_dataloader):
    model.train()
    total_train_step = 0
    for batch in train_dataloader:
        batch['text'] = batch['text'].to(args.device)
        batch['intent'] = batch['intent'].to(args.device)
        outputs = model(batch)
        loss = outputs['loss']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        # print('{} batch_end'.format(total_train_step))
        if total_train_step % 50 == 0:
            print("訓練次數：{}, Loss: {}".format(total_train_step, loss.item()))

def oneVal(model, dev_dataloader):
    model.eval()
    with torch.no_grad():
        totalCount = 0
        for batch in dev_dataloader:
            batch['text'] = batch['text'].to(args.device)
            batch['intent'] = batch['intent'].to(args.device)
            outputs = model(batch)
            # 算準確率，要算有幾個相等的
            # thisCount = 0

            # torch.view -> reshape成指定形式
            # torch.view_as -> reshape成別人樣子
            # torch.eq -> 比較每個元素是否相等
            # torch.sum -> 把元素全部加起來
            thisCount = outputs['pred'].eq(batch['intent'].view_as(outputs['pred'])).sum()
            # for i in range(args.batch_size):
            #     if outputs['pred'][i] == batch['intent'][i]:
            #         thisCount += 1
            totalCount += thisCount
        devDataSize = len(dev_dataloader.dataset)
        acc = totalCount/devDataSize
        print('thisTimeACC: {}'.format(acc))
        return acc

def saveCKPT(model):
    ckpt_path = args.ckpt_dir / 'bestIntentModel.pth'
    torch.save(model.state_dict(), ckpt_path)
    print("save_model")


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    dataLoader: Dict[str, DataLoader] = {
        split: DataLoader(split_datasets, batch_size=args.batch_size, shuffle=True, collate_fn=split_datasets.collate_fn)
        for split, split_datasets in datasets.items()
    }

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)

    model = SeqClassifier(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, datasets[TRAIN].num_classes).to(args.device)

    # TODO: init optimizer
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    bestAcc = 0.0
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        oneEpochTrain(model, optimizer, dataLoader[TRAIN])
        # TODO: Evaluation loop - calculate accuracy and save model weights
        
        acc = oneVal(model, dataLoader[DEV])
        if acc > bestAcc:
            saveCKPT(model)
            acc = bestAcc

    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
