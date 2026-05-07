import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from accelerate import Accelerator
import argparse


def build_model():
    model = models.resnet50(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def train(accelerator, model, train_loader, loss_fn, optimizer, epoch):
    model.train()
    total_loss, correct, total_samples = 0, 0, 0
    start = time.time()

    for x, y in train_loader:
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        accelerator.backward(loss)
        optimizer.step()

        total_loss += loss.item()
        correct += (pred.argmax(1) == y).sum().item()
        total_samples += y.size(0)

    if accelerator.is_main_process:
        print(f"Epoch [{epoch}] | Train Loss: {total_loss/len(train_loader):.4f} "
              f"| Acc: {100*correct/total_samples:.2f}% "
              f"| Time: {time.time()-start:.2f}s")


def test(accelerator, model, test_loader, loss_fn, epoch):
    model.eval()
    total_loss, correct, total_samples = 0, 0, 0
    start = time.time()

    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x)
            total_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).sum().item()
            total_samples += y.size(0)

    if accelerator.is_main_process:
        print(f"Epoch [{epoch}] | Test  Loss: {total_loss/len(test_loader):.4f} "
              f"| Acc: {100*correct/total_samples:.2f}% "
              f"| Time: {time.time()-start:.2f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--batch-size",  type=int,   default=512)
    parser.add_argument("--lr",          type=float, default=0.1)
    parser.add_argument("--num-workers", type=int,   default=8)
    parser.add_argument("--data-dir",    type=str,   default="../../datasets/data-cifar")
    args = parser.parse_args()

    torch.manual_seed(42)

    # Create an Accelerator object — handles device placement and distributed setup
    accelerator = Accelerator()

    # Divide batch size so effective batch stays constant regardless of GPU count.
    # accelerator.prepare() handles the DistributedSampler automatically,
    # but it does NOT divide the batch size — that's our responsibility.
    batch_size_per_gpu = args.batch_size // accelerator.num_processes

    if accelerator.is_main_process:
        print(f"[INFO] Num processes:    {accelerator.num_processes}")
        print(f"[INFO] Batch per GPU:    {batch_size_per_gpu} | Effective batch: {args.batch_size}")
        print(f"[INFO] Data dir:         {args.data_dir}")

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(args.data_dir, train=True,  download=False, transform=train_tf)
    test_dataset  = datasets.CIFAR10(args.data_dir, train=False, download=False, transform=test_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size_per_gpu,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size_per_gpu * 2,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    model     = build_model()
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Accelerator wraps model, optimizer, and dataloaders — DDP is handled automatically
    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )

    total_start = time.time()
    for epoch in range(1, args.epochs + 1):
        train(accelerator, model, train_loader, loss_fn, optimizer, epoch)
        test(accelerator, model, test_loader, loss_fn, epoch)
        scheduler.step()

    if accelerator.is_main_process:
        total = time.time() - total_start
        print(f"\nTotal time: {total:.2f}s ({total/60:.2f} min)")


if __name__ == "__main__":
    main()
