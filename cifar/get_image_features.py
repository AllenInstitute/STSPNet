from __future__ import print_function
import os
import argparse
import pickle
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class ChangeDetectionDataset(Dataset):
    """Change detection dataset (set of eight images)."""

    def __init__(self, pkl_file, transform=None):
        """
        Args:
            pkl_file (string): Path to the pkl with images stored as a dict.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # Load pickle file
        with open(pkl_file, 'rb') as f:
            images = pickle.load(f, encoding='latin1')

        # Create new image dictionary
        self.images = {}
        for key in images.keys():
            self.images[key] = images[key][key]

        # Add blank image
        self.images['blank'] = 127 * np.ones((918, 1174), dtype=np.uint8)

        # Create list of keys
        self.keys = list(self.images.keys())
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[self.keys[idx]]
        image = Image.fromarray(image)
        label = self.keys[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # CIFAR10 pre-processing
    transform = transforms.Compose(
        [transforms.Resize(32),
         transforms.CenterCrop(32),
         transforms.ToTensor(),
         transforms.Normalize((0.479,), (0.239,))])
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    model = Net().to(device)
    model.load_state_dict(torch.load('models/cifar_cnn_' +
                                     str(args.seed)+'.pt'), strict=False)
    model.eval()

    input_files = ['training', 'ophys_3', 'ophys_6', 'ophys_5']
    output_files = ['A', 'B', 'C', 'D']

    for ifile, ofile in zip(input_files, output_files):
        # Only use the luminance matched image sets
        pkl_file = '../data/Natural_Images_Lum_Matched_set_'+ifile+'_2017.07.14.pkl'
        dataset = ChangeDetectionDataset(
            pkl_file=pkl_file, transform=transform)
        dataloader = DataLoader(dataset, batch_size=9,
                                shuffle=False, **kwargs)

        # Get the features
        images, _ = next(iter(dataloader))
        output = model(images.to(device))

        # Save output as numpy file
        if not os.path.exists('features'):
            os.makedirs('features')
        np.save('features/image_set_cnn_'+ofile+'_seed_' +
                str(args.seed)+'.npy', output.data.cpu().numpy())


if __name__ == '__main__':
    main()
