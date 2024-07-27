from typing import Tuple
from tqdm import tqdm
import torch

import config
from data import load_data, TRANSFORM
from model import CNN

def test(ckpt, device, testloader, classes, K):

    metrics = {classname: {'top1_pred': 0, 
                        'topk_pred': 0, 
                        'total_pred': 0} 
                        for classname in classes}

    wrong_samples = list[Tuple]()
    # Load model
    model = CNN(in_channels=3, num_classes=10).to(device)
    model.load_state_dict(torch.load(ckpt))
    model.eval().to(device)
    # again no gradients needed
    with torch.no_grad():
        for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc="[TEST]"):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.topk(outputs, k=K, dim=1)  # Get top-K prediction

            total = labels.size(0)
            top1 = torch.sum(labels.eq(predictions[:, 0])).item()        
            topk = torch.any(labels.unsqueeze(1).eq(predictions), dim=1).sum().item()

            metrics[classes[labels[0]]]['top1_pred'] += top1
            metrics[classes[labels[0]]]['topk_pred'] += topk
            metrics[classes[labels[0]]]['total_pred'] += total
            
            for label, prediction in zip(labels, predictions):
                if label not in prediction:
                    wrong_samples.append((images[label], label, prediction))
        

    overall_top1 = sum([metrics[classname]['top1_pred'] for classname in classes]) / sum([metrics[classname]['total_pred'] for classname in classes])
    print(f"Overall Top-1 Accuracy: {overall_top1*100:.2f}%")

    overall_topk = sum([metrics[classname]['topk_pred'] for classname in classes]) / sum([metrics[classname]['total_pred'] for classname in classes])
    print(f"Overall Top-{K} Accuracy: {overall_topk*100:.2f}%")

    # print accuracy for each class
    for classname in classes:
        top1 = metrics[classname]['top1_pred'] / metrics[classname]['total_pred']
        topk = metrics[classname]['topk_pred'] / metrics[classname]['total_pred']
        print(f"Class {classname} - Top-1 Accuracy: {top1*100:.2f}% - Top-{K} Accuracy: {topk*100:.2f}%")

if __name__ == "__main__":
    trainloader, testloader = load_data(root=config.DATAPATH, transform=TRANSFORM, batchsize=config.BATCHSIZE)
    test(config.CKPTPATH, config.DEVICE, testloader, config.CLASSES, 5)

    