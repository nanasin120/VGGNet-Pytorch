import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import VGGNet
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
learning_rate = 0.0001
Epochs = 10

def plot_comparison(model_results):
    epochs = range(1, len(next(iter(model_results.values()))['train_loss']) + 1)
    
    plt.figure(figsize=(18, 5))

    # 1. Loss 그래프
    plt.subplot(1, 3, 1)
    for model_name, hist in model_results.items():
        plt.plot(epochs, [loss.cpu().detach().item() for loss in hist['train_loss']], label=f'{model_name} Loss')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 2. Accuracy 그래프
    plt.subplot(1, 3, 2)
    for model_name, hist in model_results.items():
        plt.plot(epochs, hist['test_acc'], label=f'{model_name} Acc')
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    for model_name, hist in model_results.items():
        cumulative_time = np.cumsum(hist['duration'])
        plt.plot(epochs, cumulative_time, marker='s', label=f'{model_name} Time')
        
    plt.title('Cumulative Training Time')
    plt.xlabel('Epochs')
    plt.ylabel('Total Seconds')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('vgg_training_result_high_res.png', dpi=300, bbox_inches='tight')
    plt.show()

def train(mode):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    history = {
        'train_loss': [],
        'test_acc': [],
        'duration': []
    }

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

    model = VGGNet(mode).to(device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

    start_time = time.time()

    for epoch in range(Epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0

        for data, target in tqdm(train_loader, desc=f'Epoch {epoch+1}/{Epochs}'):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss

        model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in test_loader:

                data, target = data.to(device), target.to(device)
                output = model(data)

                loss = criterion(output, target)
                test_loss += loss
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        print(f"\nAverage Loss: {train_loss/len(train_loader):.4f}, Accuracy: {correct}/{len(test_dataset)} ({100. * correct / len(test_dataset):.2f}%)\n")

        epoch_end_time = time.time()
        duration = epoch_end_time - epoch_start_time

        history['duration'].append(duration)
        history['train_loss'].append(train_loss / len(train_loader))
        history['test_acc'].append(100. * correct / len(test_dataset))
        print(f"Epoch {epoch+1} took {duration:.2f} seconds.\n")


    torch.save(model.state_dict(), f"VGGNet_cifar10_{mode}.pth")

    total_duration = time.time() - start_time
    print(f"\nTotal Training Time: {total_duration/60:.2f} minutes")
    print("Model saved to VGGNet_cifar10.pth")

    return history

if __name__ == "__main__":
    history_A = train('A')
    history_B = train('B')
    history_C = train('C')
    history_D = train('D')
    history_E = train('E')
    results = {
        'VGG-A' : history_A,
        'VGG-B' : history_B,
        'VGG-C' : history_C,
        'VGG-D' : history_D,
        'VGG-E' : history_E,
    }
    plot_comparison(results)