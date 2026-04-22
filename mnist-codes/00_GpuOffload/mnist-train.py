import time 
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class MnistModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 7 * 7, output_shape),
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x

def train(model, train_loader, loss_fn, optimizer, epoch, device):

    start_time = time.time()

    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0

    for x_train, y_train in train_loader:
        x_train, y_train = x_train.to(device), y_train.to(device)

        y_pred = model(x_train)  # Forward pass
        loss = loss_fn(y_pred, y_train)  # Compute loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted_labels = y_pred.argmax(dim=1)
        correct += (predicted_labels == y_train).sum().item()
        total_samples += y_train.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total_samples

    epoch_time = time.time() - start_time

    print(f"Epoch [{epoch}] | Train-Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% | Time: {epoch_time:.2f}s")


def test(model, test_loader, loss_fn, epoch, device):

    start_time = time.time()

    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test, y_test = x_test.to(device), y_test.to(device)

            y_pred = model(x_test)
            loss = loss_fn(y_pred, y_test)
            total_loss += loss.item()

            predicted_labels = y_pred.argmax(dim=1)
            correct += (predicted_labels == y_test).sum().item()
            total_samples += y_test.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total_samples

    epoch_time = time.time() - start_time

    print(f"Epoch [{epoch}] | Test Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}% | Time: {epoch_time:.2f}s")

def main():

    batch_size = 32
    epochs = 10

    torch.manual_seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST("../../datasets/data/", train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST("../../datasets/data/", train=False, download=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)

    model = MnistModel(input_shape=1, hidden_units=10, output_shape=10).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, epochs + 1):
        train(model, train_loader, loss_fn, optimizer, epoch, device)
        test(model, test_loader, loss_fn, epoch, device)

if __name__ == "__main__":
    main()

