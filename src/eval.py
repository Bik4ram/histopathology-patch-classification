import torch
import argparse
from model import get_model
from dataset import get_balanced_loader
from utils import compute_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--weights', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = get_balanced_loader(args.data_dir, batch_size=32)
    model = get_model(args.model).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    all_outputs, all_targets = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            all_outputs.append(outputs)
            all_targets.append(labels)
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    auc, f1 = compute_metrics(all_outputs, all_targets)
    print(f"AUC: {auc:.4f} F1: {f1:.4f}")

if __name__ == '__main__':
    main()
