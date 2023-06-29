import torchvision
from torchvision.datasets.utils import download_and_extract_archive
import os
from PIL import Image
import torch


class CINIC10(torchvision.datasets.VisionDataset):
    base_folder = "CINIC-10"
    url = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz"
    filename = "CINIC-10.tar.gz"

    tgz_md5 = "6ee4d0c996905fe93221de577967a372"

    paths = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    classes = {
        0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse",
        8: "ship", 9: "truck"}

    def __init__(self, root, train=True, transform=None, target_transform=None, download=True):
        super(CINIC10, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train

        suffix = "train/" if train else "test/"

        if download:
            self.download()

        self.data = []
        self.targets = []

        for path_idx, path in enumerate(self.paths):
            file_names = os.listdir(self.root + "CINIC-10/" + suffix + path)
            for file_name in file_names:
                with open(self.root + "CINIC-10/" + suffix + path + "/" + file_name, "rb") as fd:
                    img = Image.open(fd)
                    img = img.convert("RGB")
                    self.data.append(img)
                    self.targets.append(path_idx)

    def download(self):
        if not os.path.isfile(self.root + "/CINIC-10" + "/README.md"):
            download_and_extract_archive(self.url, self.root, extract_root=self.root + "CINIC-10",
                                            filename=self.filename, md5=self.tgz_md5)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, torch.tensor(target, dtype=int)
