import json
import hashlib
import os
import argparse
import copy
import logging
import sys


def dict_hash(dictionary):
    dhash = hashlib.md5()
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FL Training')
    parser.add_argument("--session_tag", type=str, default="default_session", help="Sets name of subfolder for experiments")
    parser.add_argument("--algorithm", type=str, default="FedAvg",
                        choices=['CoCoFL', 'Centralized', 'FedAvg', 'FedAvgDropDevices'], help="Choice of algorithm")
    parser.add_argument("--dataset", type=str, choices=["CIFAR10", "CIFAR100", "CINIC10",
                                                        "FEMNIST", "IMDB", "XCHEST", "SHAKESPEARE"], default="CIFAR10", help="Datasets (not all dataset/network combinations are possible)")
    parser.add_argument("--seed", type=int, default=11, help="Sets random seed for experiment")
    parser.add_argument("--network", choices=['MobileNet', 'MobileNetGroupNorm', 'ResNet18', 'ResNet50', 'DenseNet',
                                              'Transformer', 'TransformerSeq2Seq', 'MobileNetLarge'], default="MobileNet", help="NN for experiement")
    parser.add_argument("--n_devices", type=int, default=100, help="Number of total FL devices")
    parser.add_argument("--n_devices_per_round", type=int, default=10, help="Number of FL devices active in one round")
    parser.add_argument("--data_distribution", type=str, choices=['IID', 'NONIID', 'RCNONIID'], default="IID")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--lr_schedule", nargs="+", type=int, default=[600, 800], help="Learning Rate Schedule (LR is reduced by 10x per step)")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay for SGD optimizer")
    parser.add_argument("--noniid_dirichlet", type=float, default=0.1, help="Dirichlet alpha to control non-iidnes")

    parser.add_argument("--n_rounds", type=int, default=1000, help="Number of total FL rounds")
    parser.add_argument("--torch_device", type=str, default="cuda:0", help="PyTorch device (cuda or cpu)")
    parser.add_argument("--progress_bar", type=bool, default=True, help="Progress bar printed in stdout")
    parser.add_argument("--plot", type=bool, default=True, help="Plots are generated every 25 FL rounds")

    parser.add_argument("--dry_run", help="Dry_run loads the NN, datasets, but does not apply training and does not create any files")

    args = parser.parse_args()
    print(args)

    assert args.n_devices >= args.n_devices_per_round, "Cannot be more active devices than overall devices"
    # compatability checks
    if args.network in ['MobileNet', 'MobileNetGroupNorm', 'ResNet18', 'MobileNet', 'DenseNet']:
        if args.dataset not in ['CIFAR10', 'CIFAR100', 'FEMNIST', 'CINIC10']:
            raise ValueError(args.dataset)
    if args.network == "Transformer":
        if args.dataset != "IMDB":
            raise ValueError(args.dataset)
    if args.network == 'TransformerSeq2Seq':
        if args.dataset != "SHAKESPEARE":
            raise ValueError(args.dataset)
    if args.network == 'MobileNetLarge':
        if args.dataset != "XCHEST":
            raise ValueError(args.dataset)

    settings = vars(copy.deepcopy(args))

    settings.pop('torch_device')
    settings.pop('n_rounds')
    settings.pop('plot')
    settings.pop('dry_run')
    settings.pop('progress_bar')

    if args.algorithm == 'Centralized':
        settings.pop('n_devices')
        settings.pop('n_devices_per_round')
        settings.pop('data_distribution')
        settings.pop('noniid_dirichlet')
        args.n_devices_per_round = 1
        args.n_devices = 1

    if args.data_distribution == 'IID':
        try:
            settings.pop('noniid_dirichlet')
        except KeyError:
            pass

    if args.dataset == 'SHAKESPEARE':
        if args.data_distribution == 'RCNONIID':
            try:
                settings.pop('noniid_dirichlet')
            except KeyError:
                pass

    if args.torch_device.startswith('cuda'):
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.torch_device.split('cuda:')[1]
            args.torch_device = 'cuda'
        except IndexError:
            pass

    run_hash = dict_hash(settings)
    run_path = "runs/" + args.session_tag + "/run_" + run_hash

    import torch
    torch.manual_seed(args.seed)
    torch.set_num_threads(12)
    torch.set_num_interop_threads(12)

    if args.algorithm == 'FedAvg':
        from algorithms.fedavg import FedAvgServer
        flserver = FedAvgServer(run_path)
    elif args.algorithm == 'FedAvgDropDevices':
        from algorithms.fedavg_drop_devices import FedAvgDropDevicesServer
        flserver = FedAvgDropDevicesServer(run_path)
    elif args.algorithm == 'CoCoFL':
        from algorithms.cocofl import CoCoFLServer
        flserver = CoCoFLServer(run_path)
    elif args.algorithm == 'Centralized':
        from algorithms.centralized import CentralizedServer
        flserver = CentralizedServer(run_path)

    from utils.resources import DeviceResources, Constant, Uniform

    # Implementation of strong/medium/weak scheme
    device_constraints = [DeviceResources() for _ in range(args.n_devices)]
    if args.algorithm in ['CoCoFL', 'FedAvgDropDevices']:
        for resource in device_constraints[0:int(0.33*args.n_devices)]:
            resource.set_all(Constant(1.0), Constant(1.0), Constant(1.0))
        for resource in device_constraints[int(0.33*args.n_devices):int(0.66*args.n_devices)]:
            resource.set_all(Constant(0.66), Uniform(0.5, 1.0), Constant(0.66))
        for resource in device_constraints[int(0.66*args.n_devices):]:
            resource.set_all(Constant(0.33), Uniform(0.5, 1.0), Constant(0.33))
    else:
        for resource in device_constraints:
            resource.set_all(Constant(1.0), Constant(1.0), Constant(1.0))
    flserver.set_device_constraints(device_constraints)

    flserver.n_devices_per_round = args.n_devices_per_round
    flserver.n_devices = args.n_devices
    flserver.torch_device = args.torch_device
    flserver.n_rounds = args.n_rounds
    flserver.lr = args.lr
    flserver.set_seed(args.seed)
    flserver.lr_schedule = [[args.lr_schedule[i], args.lr/10**(i+1)] for i in range(len(args.lr_schedule))]
    flserver.progress_output = args.progress_bar

    flserver.set_optimizer(torch.optim.SGD, {'weight_decay': args.weight_decay})

    from torchvision import datasets, transforms

    # Dataset selection
    if 'CIFAR' in args.dataset:
        tf = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        kwargs = {'download': True, 'transform': tf}
        if args.dataset.endswith('100'):
            flserver.set_dataset(datasets.CIFAR100, "/tmp/", **kwargs)
            cnn_args = {'num_classes': 100}
        elif args.dataset.endswith('10'):
            flserver.set_dataset(datasets.CIFAR10, "/tmp/", **kwargs)
            cnn_args = {'num_classes': 10}
    elif 'CINIC10' in args.dataset:
        from utils.datasets.cinic10 import CINIC10
        tf = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.478, 0.472, 0.430), (0.242, 0.238, 0.258))])
        kwargs = {'download': True, 'transform': tf}
        flserver.set_dataset(CINIC10, "/tmp/", **kwargs)
        cnn_args = {'num_classes': 10}
    elif 'XCHEST' in args.dataset:
        from utils.datasets.xchest import XCHEST
        tf = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        kwargs = {'download': True, 'transform': tf}
        flserver.set_dataset(XCHEST, "/tmp/", **kwargs)
        flserver.is_unbalanced = True
        cnn_args = {'num_classes': 2}
    elif 'FEMNIST' in args.dataset:
        from utils.datasets.femnist import FEMNIST, femnist_to_cifar_format_transform
        tf = transforms.Compose([femnist_to_cifar_format_transform()])
        kwargs = {'transform': tf}
        flserver.set_dataset(FEMNIST, "data/", **kwargs)
        cnn_args = {'num_classes': 62}
    elif 'IMDB' in args.dataset:
        from utils.datasets.imdb import IMDB
        seq_len = 512
        kwargs = {'seq_len': seq_len}
        cnn_args = {}
        flserver.set_dataset(IMDB, "data/", **kwargs)
    elif 'SHAKESPEARE' in args.dataset:
        from utils.datasets.shakespeare import SHAKESPEARE
        kwargs = {}
        cnn_args = {}
        flserver.set_dataset(SHAKESPEARE, "data/", **kwargs)

    # Model selection
    if args.network == 'ResNet18':
        from nets.Baseline.ResNet.resnet import ResNet18 as baseline_ResNet18
        net_eval = baseline_ResNet18
        if args.algorithm in ['Centralized', 'FedAvg', 'FedAvgDropDevices']:
            from nets.Baseline.ResNet.resnet import ResNet18
            net = ResNet18
            net_eval = ResNet18
        elif args.algorithm in ['CoCoFL']:
            net_eval = baseline_ResNet18
            from nets.QuantizedNets.ResNet.resnet import QResNet18
            net = QResNet18
    elif args.network == 'ResNet50':
        from nets.Baseline.ResNet.resnet import ResNet50 as baseline_ResNet50
        net_eval = baseline_ResNet50
        if args.algorithm in ['Centralized', 'FedAvg', 'FedAvgDropDevices']:
            from nets.Baseline.ResNet.resnet import ResNet50
            net = ResNet50
            net_eval = ResNet50
        elif args.algorithm in ['CoCoFL']:
            net_eval = baseline_ResNet50
            from nets.QuantizedNets.ResNet.resnet import QResNet50
            net = QResNet50
    elif args.network == 'MobileNetGroupNorm':
        from nets.Baseline.MobileNet.mobilenet_norm import MobileNetV2GroupNorm
        if args.algorithm in ['Centralized', 'FedAvg', 'FedAvgDropDevices']:
            net_eval = MobileNetV2GroupNorm
            net = MobileNetV2GroupNorm
        elif args.algorithm in ['CoCoFL']:
            from nets.QuantizedNets.MobileNet.mobilenet_norm import QMobileNetGroupNorm
            net_eval = MobileNetV2GroupNorm
            net = QMobileNetGroupNorm
    elif args.network == 'MobileNetLarge':
        from nets.Baseline.MobileNet.mobilenet import MobileNetV2Large
        if args.algorithm in ['Centralized', 'FedAvg', 'FedAvgDropDevices']:
            net_eval = MobileNetV2Large
            net = MobileNetV2Large
        elif args.algorithm in ['CoCoFL']:
            from nets.QuantizedNets.MobileNet.mobilenet import QMobileNetLarge
            net_eval = MobileNetV2Large
            net = QMobileNetLarge
    elif args.network == 'MobileNet':
        from nets.Baseline.MobileNet.mobilenet import MobileNetV2 as baseline_MobileNetV2
        net_eval = baseline_MobileNetV2
        if args.algorithm in ['Centralized', 'FedAvg', 'FedAvgDropDevices']:
            net = baseline_MobileNetV2
            net_eval = baseline_MobileNetV2
        elif args.algorithm in ['CoCoFL']:
            net_eval = baseline_MobileNetV2
            from nets.QuantizedNets.MobileNet.mobilenet import QMobileNet
            net = QMobileNet
    elif args.network == 'DenseNet':
        from nets.Baseline.DenseNet.densenet import DenseNet40 as baseline_DenseNet40
        net_eval = baseline_DenseNet40
        if args.algorithm in ['Centralized', 'FedAvg', 'FedAvgDropDevices']:
            net = baseline_DenseNet40
            net_eval = baseline_DenseNet40
        elif args.algorithm in ['CoCoFL']:
            net_eval = baseline_DenseNet40
            from nets.QuantizedNets.DenseNet.densenet import QDenseNet40
            net = QDenseNet40
    elif args.network == 'TransformerSeq2Seq':
        from nets.Baseline.Transformer.transformer import TransformerSeq2Seq
        net_eval = TransformerSeq2Seq
        if args.algorithm in ['Centralized', 'FedAvg', 'FedAvgDropDevices']:
            net = TransformerSeq2Seq
        elif args.algorithm in ['CoCoFL']:
            from nets.QuantizedNets.Transformer.transformer import QTransformerSeq2Seq
            net = QTransformerSeq2Seq
    elif args.network == 'Transformer':
        from nets.Baseline.Transformer.transformer import Transformer as baseline_Transformer
        net_eval = baseline_Transformer
        if args.algorithm in ['Centralized', 'FedAvg', 'FedAvgDropDevices']:
            if args.network != 'Transformer': raise ValueError(args.algorithm)
            net = baseline_Transformer
        elif args.algorithm in ['CoCoFL']:
            from nets.QuantizedNets.Transformer.transformer import QTransformer
            net = QTransformer

    from utils.split import split_iid, split_noniid, split_rcnoniid, split_SHAKESPEARE_rcnoniid


    if args.data_distribution == 'IID':
        flserver.split_function = split_iid(run_path, False if args.dry_run else args.plot, args.seed)
    elif args.data_distribution == 'NONIID':
        flserver.split_function = split_noniid(args.noniid_dirichlet, run_path, False if args.dry_run else args.plot, args.seed)
    elif args.data_distribution == 'RCNONIID':
        if args.dataset == 'SHAKESPEARE':
             flserver.split_function = split_SHAKESPEARE_rcnoniid(run_path, False if args.dry_run else args.plot, args.seed)
        else:
            flserver.split_function = split_rcnoniid(args.noniid_dirichlet, run_path, False if args.dry_run else args.plot, args.seed)

    cnn_args_list = [cnn_args for _ in range(args.n_devices)]
    flserver.set_model([net for _ in range(args.n_devices)], cnn_args_list)
    flserver.set_model_evaluation(net_eval, cnn_args_list[0])

    if args.dry_run:
        print("DRY RUN PERFORMED SUCCESSFULLY")
        print("Run HASH ", run_hash)
        flserver.initialize()
        sys.exit(0)

    try:
        os.makedirs(run_path)
        with open(run_path + "/" + "fl_setting.json", "w") as fd:
            json.dump(settings, fd, indent=4)
    except FileExistsError:
        pass

    logging.basicConfig(format='%(asctime)s - %(message)s',
                            filename=run_path + '/run.log', level=logging.INFO, filemode='w')
    logging.info("Started")
    logging.info(f"Settings Hash: {run_hash}")
    logging.info("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in settings.items()) + "}")
    flserver.initialize()

    if args.plot:
        import utils.plots as plots
        flserver.set_plotting_callback(plots.plot_config, run_path)

    try:
        flserver.run()
    except KeyboardInterrupt:
        pass

    if args.plot:
        plots.plot_config(run_path)

    try:
        os.unlink("latest_run")
    except FileNotFoundError:
        pass
    os.symlink(run_path, "latest_run")