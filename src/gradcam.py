import torch
import argparse
import cv2
import numpy as np
from model import get_model
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

def preprocess(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    return transform(img).unsqueeze(0)

def show_gradcam_on_image(img_path, model, target_layer, device):
    img = preprocess(img_path).to(device)
    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    def forward_hook(module, inp, out):
        activations.append(out.detach())

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_backward_hook(backward_hook)

    output = model(img)
    pred_class = output.argmax(dim=1).item()
    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients[0]
    acts = activations[0]
    weights = grads.mean(dim=[2,3], keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = cam.squeeze().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224,224))
    cam = cam / cam.max()

    orig_img = cv2.imread(img_path)
    orig_img = cv2.resize(orig_img, (224,224))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_img, 0.5, heatmap, 0.5, 0)
    plt.imshow(overlay[...,::-1])
    plt.title(f"Grad-CAM: Predicted Class {pred_class}")
    plt.axis('off')
    plt.show()
    handle_f.remove()
    handle_b.remove()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--weights', type=str, required=True)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args.model).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    # Pick last conv layer
    if args.model.startswith('resnet'):
        target_layer = model.layer4[-1]
    elif args.model.startswith('efficientnet'):
        target_layer = model._blocks[-1]
    else:
        raise ValueError("Unknown model for Grad-CAM")
    show_gradcam_on_image(args.img_path, model, target_layer, device)

if __name__ == '__main__':
    main()
