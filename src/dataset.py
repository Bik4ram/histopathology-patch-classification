import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

class PatchDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.labels = []
        self.class_to_idx = {"benign": 0, "malignant": 1}
        for cls in self.class_to_idx:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.endswith('.png') or fname.endswith('.jpg'):
                    self.samples.append(os.path.join(cls_dir, fname))
                    self.labels.append(self.class_to_idx[cls])
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        img = self.transform(img)
        label = self.labels[idx]
        return img, label

def get_balanced_loader(root_dir, batch_size=32, shuffle=True):
    dataset = PatchDataset(root_dir)
    class_counts = [dataset.labels.count(0), dataset.labels.count(1)]
    weights = [1.0/class_counts[label] for label in dataset.labels]
    sampler = WeightedRandomSampler(weights, len(weights))
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return loader
