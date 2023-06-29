import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt
import copy
import logging


class split_iid():
    def __init__(self, run_path, is_plot, seed):
        self._run_path = run_path
        self._is_plot = is_plot
        self._seed = seed

    def __call__(self, dataset, n_device):
        length = len(dataset)

        # Uses specific random generator (with seed)
        rng = np.random.default_rng(self._seed)
        idxs = np.array_split(rng.permutation(length), n_device)

        dev_targets = []
        for i in idxs:
            dev_targets.append(torch.tensor(dataset.dataset.targets)[i])

        if self._is_plot:
            self.plot(len(dataset.dataset.classes), n_device, dev_targets, self._run_path)

        idxs = [torch.tensor(idx) for idx in idxs]

        return idxs

    @staticmethod
    def plot(n_classes, n_device, dev_targets, path_prefix):
        dev_targets = copy.deepcopy(dev_targets)

        # preprocessing for visualization purpose
        min_sample = np.inf
        for item in dev_targets:
            min_sample = len(item) if len(item) < min_sample else min_sample
        for i in range(len(dev_targets)):
            dev_targets[i] = dev_targets[i][0:(min_sample - 1)]

        x, y = np.meshgrid([i for i in range(n_classes)], [i for i in range(n_device)])
        dev_targets = np.array(torch.stack(dev_targets))
        s = np.apply_along_axis(np.bincount, 1, dev_targets, minlength=n_classes)

        fig = plt.figure()
        ax = fig.subplots(1)

        ax.scatter(y, x, s=s, color="tab:red", alpha=0.5)
        ax.set_xlabel("dev_idx")
        ax.set_ylabel("class_idx")
        plt.savefig(path_prefix + "/data_distribution.png", )
        return


class split_noniid():
    def __init__(self, alpha, run_path, is_plot, seed):
        self._alpha = alpha
        self._run_path = run_path
        self._is_plot = is_plot
        self._seed = seed

    def __call__(self, dataset, n_device):

        rng = np.random.default_rng(self._seed)

        n_classes = len(dataset.dataset.classes)
        len_dataset = len(dataset.dataset.targets)
        n_data_per_device = int(len_dataset/n_device)

        idx = [torch.where(torch.tensor(dataset.dataset.targets) == i)[0] for i in range(n_classes)]
        idx = [[int(i) for i in item] for item in idx]

        n_samples_per_class = [len(item) for item in idx]

        s = np.stack([rng.multinomial(int(n_samples_per_class[i]),
                                      rng.dirichlet([1*self._alpha for _ in range(n_device)])) for i in range(n_classes)]).tolist()
        new_ordered_idxs = []
        ii = list(range(n_classes))
        jj = list(range(n_device))

        for j in jj:
            for i in ii:
                new_ordered_idxs.append(idx[i][0:s[i][j]])
                idx[i] = idx[i][s[i][j]:]

        new_ordered_idxs = list(itertools.chain(*new_ordered_idxs))
        dev_idxs = []
        dev_targets = []
        targets = torch.tensor(dataset.dataset.targets)[new_ordered_idxs]
        for i in range(n_device):
            dev_idxs.append(new_ordered_idxs[(i*n_data_per_device):((i+1)*n_data_per_device)])
            dev_targets.append(targets[(i*n_data_per_device):((i+1)*n_data_per_device)])
        dev_idxs = [torch.tensor(idxs) for idxs in dev_idxs]

        if self._is_plot:
            dev_targets = [torch.tensor(item) for item in dev_targets]
            self.plot(n_classes, n_device, dev_targets, self._run_path)

        return dev_idxs

    @staticmethod
    def plot(n_classes, n_device, dev_targets, path_prefix):
        x, y = np.meshgrid([i for i in range(n_classes)], [i for i in range(n_device)])
        dev_targets = np.array(torch.stack(dev_targets))
        s = np.apply_along_axis(np.bincount, 1, dev_targets, minlength=n_classes)

        fig = plt.figure()
        ax = fig.subplots(1)

        ax.scatter(y, x, s=s, color="tab:red", alpha=0.5)
        ax.set_xlabel("dev_idx")
        ax.set_ylabel("class_idx")
        plt.savefig(path_prefix + "/data_distribution.png", )
        return


class split_rcnoniid():
    def __init__(self, alpha, run_path, is_plot, seed, n_groups=3):
        self._alpha = alpha
        self._run_path = run_path
        self._is_plot = is_plot
        self._seed = seed
        self._n_groups = n_groups

    def __call__(self, dataset, n_device):

        rng = np.random.default_rng(self._seed)

        n_classes = len(dataset.dataset.classes)
        len_dataset = len(dataset.dataset.targets)
        n_data_per_device = int(len_dataset/n_device)

        idx = [torch.where(torch.tensor(dataset.dataset.targets) == i)[0] for i in range(n_classes)]
        idx = [[int(i) for i in item] for item in idx]

        n_samples_per_class = [len(item) for item in idx]

        s = np.stack([rng.multinomial(int(n_samples_per_class[i]),
                                      rng.dirichlet([1*self._alpha for _ in range(self._n_groups)])) for i in range(n_classes)]).tolist()
        new_ordered_idxs = []

        ii = list(range(n_classes))
        jj = list(range(self._n_groups))

        for j in jj:
            for i in ii:
                new_ordered_idxs.append(idx[i][0:s[i][j]])
                idx[i] = idx[i][s[i][j]:]
        cache = copy.deepcopy(new_ordered_idxs)
        new_ordered_idxs = []
        for i in range(self._n_groups*n_classes):
            new_item = []
            for item in cache[i*n_classes:(i+1)*n_classes]:
                new_item += item
            rng.shuffle(new_item)
            new_ordered_idxs.append(new_item)

        new_ordered_idxs = list(itertools.chain(*new_ordered_idxs))
        dev_idxs = []
        dev_targets = []
        targets = torch.tensor(dataset.dataset.targets)[new_ordered_idxs]
        for i in range(n_device):
            dev_idxs.append(new_ordered_idxs[(i*n_data_per_device):((i+1)*n_data_per_device)])
            dev_targets.append(targets[(i*n_data_per_device):((i+1)*n_data_per_device)])
        dev_idxs = [torch.tensor(idxs) for idxs in dev_idxs]

        if self._is_plot:
            self.plot(n_classes, n_device, dev_targets, self._run_path, self._n_groups)

        return dev_idxs

    @staticmethod
    def plot(n_classes, n_device, dev_targets, path_prefix, n_groups):

        fig = plt.figure()
        ax = fig.subplots(1, n_groups)

        occ_per_class = []

        for j in range(n_groups):

            x, y = np.meshgrid([i for i in range(n_classes)], [i for i in range(j*int(n_device/n_groups), (j+1)*int(n_device/n_groups))])
            dev_trg = np.array(torch.stack(dev_targets[j*int(n_device/n_groups):(j+1)*int(n_device/n_groups)]))
            s = np.apply_along_axis(np.bincount, 1, dev_trg, minlength=n_classes)
            occ_per_class.append(np.sum(s, axis=0))
            ax[j].scatter(y, x, s=s, color="tab:red", alpha=0.5)
            ax[j].set_xlabel(f"dev_idx group {j}")
            ax[j].set_ylabel("class_idx")
        plt.savefig(path_prefix + "/data_distribution.png")

        fig = plt.figure()
        ax = fig.subplots(1, n_groups)
        for j in range(n_groups):
            ax[j].plot(occ_per_class[j]/np.sum(occ_per_class[j]))
            logging.info(f"[DISTRIBUTION]: group {j+1}: {occ_per_class[j]/np.sum(occ_per_class[j])}")
        plt.savefig(path_prefix + "/data_distribution_per_group.png")
        return


class split_SHAKESPEARE_rcnoniid():
    def __init__(self, run_path, is_plot, seed, n_groups=3):

        self._seed = seed
        self._n_groups = n_groups
        self._is_plot = is_plot
        self._run_path = run_path

    def __call__(self, dataset, n_device):

        rng = np.random.default_rng(self._seed)

        n_classes = len(dataset.dataset.class_idxs)

        class_idxs = copy.deepcopy(dataset.dataset.class_idxs)

        val = [[i]*len(class_idxs[i]) for i in range(len(class_idxs))]
        val = np.array(list(itertools.chain(*val)))

        rng.shuffle(class_idxs)
        class_idxs = np.array(list(itertools.chain(*class_idxs)))

        class_idxs_splits = np.array_split(class_idxs, self._n_groups)

        class_idxs = np.concatenate(class_idxs_splits)
        dev_idxs = np.array_split(class_idxs, n_device)

        dev_targets = []
        for i in dev_idxs:
            dev_targets.append(torch.tensor(val[i]))

        if self._is_plot:
            self.plot(n_classes, n_device, dev_targets, self._run_path, self._n_groups)

        for i in range(len(dev_idxs)):
            dev_idxs[i] = torch.tensor(dev_idxs[i])

        return dev_idxs

    @staticmethod
    def plot(n_classes, n_device, dev_targets, path_prefix, n_groups):
        dev_targets = copy.deepcopy(dev_targets)

        # preprocessing for visualization purpose
        min_sample = np.inf
        for item in dev_targets:
            min_sample = len(item) if len(item) < min_sample else min_sample
        for i in range(len(dev_targets)):
            dev_targets[i] = dev_targets[i][0:(min_sample - 1)]

        fig = plt.figure()
        ax = fig.subplots(1, n_groups)

        occ_per_class = []

        for j in range(n_groups):

            x, y = np.meshgrid([i for i in range(n_classes)], [i for i in range(j*int(n_device/n_groups), (j+1)*int(n_device/n_groups))])
            dev_trg = np.array(torch.stack(dev_targets[j*int(n_device/n_groups):(j+1)*int(n_device/n_groups)]))
            s = np.apply_along_axis(np.bincount, 1, dev_trg, minlength=n_classes)
            occ_per_class.append(np.sum(s, axis=0))
            ax[j].scatter(y, x, s=s, color="tab:red", alpha=0.5)
            ax[j].set_xlabel(f"dev_idx group {j}")
            ax[j].set_ylabel("class_idx")
        plt.savefig(path_prefix + "/data_distribution.png")
        return