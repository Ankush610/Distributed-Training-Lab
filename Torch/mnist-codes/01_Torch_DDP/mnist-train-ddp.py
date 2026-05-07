import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse

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

def train(rank, model, train_loader, loss_fn, optimizer, epoch, device):

    start_time = time.time()

    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0

    for x_train, y_train in train_loader:
        x_train, y_train = x_train.to(device), y_train.to(device)

        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)
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

    if rank == 0:
        print(f"Epoch [{epoch}] | Train-Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% | Time: {epoch_time:.2f}s")


def test(rank, model, test_loader, loss_fn, epoch, device):

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

    if rank == 0:
        print(f"Epoch [{epoch}] | Test Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}% | Time: {epoch_time:.2f}s")

def setup():
    dist.init_process_group(backend="nccl")

def cleanup():
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="PyTorch DDP MNIST Example")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    # --batch-size is now the GLOBAL batch size; it will be divided by world_size per GPU
    parser.add_argument("--batch-size", type=int, default=32, help="Global batch size (divided evenly across GPUs)")
    parser.add_argument("--lr", type=float, default=0.01, help="Base learning rate (scaled linearly with world size)")
    parser.add_argument("--save-model", action="store_true", help="Save trained model")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank (assigned by torchrun)")
    args = parser.parse_args()

    torch.manual_seed(42)

    setup()

    rank       = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    # --- Batch size: divide global batch size across GPUs so optimizer steps stay constant ---
    per_gpu_batch_size = args.batch_size // world_size
    assert per_gpu_batch_size > 0, (
        f"Global batch size ({args.batch_size}) must be >= world_size ({world_size})"
    )

    if rank == 0:
        print(f"[INFO] Using {world_size} GPUs")
        print(f"[INFO] Global batch size: {args.batch_size} | Per-GPU: {per_gpu_batch_size}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST("../../datasets/data", train=True, download=False, transform=transform)
    test_dataset  = datasets.MNIST("../../datasets/data", train=False, download=False, transform=transform)

    train_sampler = DistributedSampler(train_dataset)
    test_sampler  = DistributedSampler(test_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=per_gpu_batch_size, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  sampler=test_sampler,  batch_size=per_gpu_batch_size, pin_memory=True)

    model = MnistModel(input_shape=1, hidden_units=10, output_shape=10).to(device)
    model = DDP(model, device_ids=[local_rank])

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        train(rank, model, train_loader, loss_fn, optimizer, epoch, device)
        test(rank, model, test_loader, loss_fn, epoch, device)

    if rank == 0 and args.save_model:
        torch.save(model.module.state_dict(), "mnist_ddp_model.pth")

    cleanup()

if __name__ == "__main__":
    main()
