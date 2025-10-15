import torch
import argparse
import os
from model import get_model
from dataset import get_balanced_loader
from losses import FocalLoss
from utils import compute_metrics
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args.model).to(device)
    train_loader = get_balanced_loader(args.data_dir, args.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = FocalLoss()

    best_auc = 0
    os.makedirs('runs', exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        total_loss, all_outputs, all_targets = 0, [], []
        for imgs, labels in tqdm(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            all_outputs.append(outputs)
            all_targets.append(labels)
        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)
        auc, f1 = compute_metrics(all_outputs, all_targets)
        print(f"Epoch {epoch+1}/{args.epochs} Loss: {total_loss/len(train_loader.dataset):.4f} AUC: {auc:.4f} F1: {f1:.4f}")
        # Save best model
        if auc > best_auc:
            torch.save(model.state_dict(), f'runs/best_model.pth')
            best_auc = auc

if __name__ == '__main__':
    main()
