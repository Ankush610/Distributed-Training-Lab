# download_data.py
# Run on login node: python download_data.py --data-dir /path/to/shared/data

import argparse
import torchvision.datasets as datasets
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, default="../datasets/data-cifar")
args = parser.parse_args()

print(f"[*] Downloading CIFAR-10 to {args.data_dir} ...")
datasets.CIFAR10(args.data_dir, train=True,  download=True, transform=transforms.ToTensor())
datasets.CIFAR10(args.data_dir, train=False, download=True, transform=transforms.ToTensor())
print("[✓] CIFAR-10 download complete.")
