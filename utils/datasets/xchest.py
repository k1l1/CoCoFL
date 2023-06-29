import torchvision
from torchvision.datasets.utils import download_and_extract_archive
import os
import torch


class XCHEST(torchvision.datasets.VisionDataset):
    base_folder = "xchest"
    url = "https://bwsyncandshare.kit.edu/s/fEwKeDoHKDtnzF7/download/xchest.tar.gz"
    filename = "xchest2.tar.gz"

    tgz_md5 = "3197a0bb69cb699cf390b4fd71671ee2"

    classes = { 0: "No Finding", 1: "Finding"}

    def __init__(self, root, train=True, transform=None, target_transform=None, download=True):
        super(XCHEST, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train

        self.suffix = "/train_" if train else "/test_"
        batches = 10 if train else 1

        if download:
            self.download()

        self.data = torch.cat([torch.load(self.root + self.base_folder + self.suffix + f"data_batch_{i}.pt") for i in range(batches)])
        self.targets = torch.cat([torch.load(self.root + self.base_folder + self.suffix + f"targets_batch_{i}.pt") for i in range(batches)])[:, -1]
        if self.train:
            self.data = self.data[:12700, :]
            self.targets = self.targets[:12700]

    def download(self):
        if not os.path.isfile(self.root + self.base_folder + self.suffix + "data_batch_0.pt"):
            download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def __len__(self):
        return int(self.targets.shape[0])

    def __getitem__(self, index: int):
        img, target = self.data[index, :], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img.float(), target