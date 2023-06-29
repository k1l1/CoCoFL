import argparse
import random
import copy
import json
import tqdm
import numpy as np
from multiprocessing import Process, Queue
import gc

import resource
import torch
from timeit import default_timer as timer

# filter for trained tensors (with changes)


def filter_state_dict(state_dict, model):
    for name, param in model.named_parameters():
        if not param.requires_grad:
            state_dict.pop(name)
    return state_dict


def sizeof_state_dict(state_dict):
    size_in_bytes = 0
    for key in state_dict:
        tensor_dim = 1
        for dim in state_dict[key].shape:
            tensor_dim *= dim
        size_in_bytes += 4*tensor_dim  # conversion to bytes
    return size_in_bytes


def profile(model_class, model_kwargs, model_state_dict_path, q, n_batches=16):
    # set number of threads torch can use
    torch.set_num_threads(4)

    # depending on ARM/X64 qnnpack or fbgemm backend is chosen
    torch.backends.quantized.engine = "fbgemm" if "fbgemm" in torch.backends.quantized.supported_engines else "qnnpack"
    lr = 0.1
    torch_device = torch.device("cpu")
    time_per_epoch_forward = []
    time_per_epoch_bw = []

    # previous memory is considered overhead
    max_mem_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(10**6)

    model = model_class(**model_kwargs)
    model.to(torch_device)
    model.load_state_dict(torch.load(model_state_dict_path), strict=False)
    inputs, targets = torch.load("profiling/inputs.pt"), torch.load("profiling/targets.pt")

    optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=lr)
    model.train()

    for _ in range(n_batches):
        inputs, targets = inputs.to(torch_device), targets.to(torch_device)

        t_start = timer()
        outputs = model(inputs)
        t_end = timer()

        time_per_epoch_forward.append(t_end - t_start)

        t_start = timer()
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        t_end = timer()

        # ensure that tmp object do not stay in memory throughout several batches
        del loss
        del outputs
        gc.collect()

        time_per_epoch_bw.append(t_end - t_start)

    max_mem_end = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(10**6)

    data_down = sizeof_state_dict(torch.load(model_state_dict_path))
    data_up = sizeof_state_dict(filter_state_dict(model.state_dict(), model))

    q.put([time_per_epoch_forward, time_per_epoch_bw, data_up, data_down, max_mem_end - max_mem_start])
    return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some integers.")

    parser.add_argument("--network", default="MobileNet", type=str, choices=["MobileNet", "MobileNetGroupNorm", "MobileNetLarge", "DenseNet", "ResNet18", "ResNet50", "Transformer"], help="NN model selection")
    parser.add_argument("--architecture", default="x64", type=str, choices=["x64", "arm"], help="Selects hardware architecture")
    parser.add_argument("--epochs", type=int, default=1, help="Selects how many profiling epochs are performed (more epochs reduce noise but require more time)")

    args = parser.parse_args()
    print(args)

    model = None
    model_state_dict_path = "profiling/state_dict.pt"
    torch.save(torch.rand(32, 3, 32, 32), "profiling/inputs.pt")
    torch.save(torch.ones(32, dtype=int), "profiling/targets.pt")

    from nets.QuantizedNets.MobileNet.mobilenet import QMobileNet, QMobileNetLarge
    from nets.QuantizedNets.MobileNet.mobilenet_norm import QMobileNetGroupNorm

    if "ResNet18" in args.network:
        from nets.QuantizedNets.ResNet.resnet import QResNet18
        if args.network == "QResNet18": model = QResNet18
        else: raise ValueError(args.network)
    elif args.network == "MobileNet": model = QMobileNet
    elif args.network == "MobileNetGroupNorm": model = QMobileNetGroupNorm
    elif args.network == "MobileNetLarge":
        model = QMobileNetLarge
        torch.save(torch.rand(32, 3, 221, 221), "profiling/inputs.pt")
    elif "ResNet50" in args.network:
        from nets.QuantizedNets.ResNet.resnet import QResNet50
        model = QResNet50
    elif "DenseNet" in args.network:
        from nets.QuantizedNets.DenseNet.densenet import QDenseNet40
        model = QDenseNet40
    elif "Transformer" in args.network:
        torch.save(torch.ones(32, 512, dtype=int), "profiling/inputs.pt")
        torch.save(torch.ones(32, dtype=int), "profiling/targets.pt")
        from nets.QuantizedNets.Transformer.transformer import QTransformer
        model = QTransformer

    model_state_dict_path = "profiling/state_dict.pt"
    torch.save(model().state_dict(), model_state_dict_path)

    print(model)
    print(model_state_dict_path)
    file_string_prefix = f"profiling/profiling__CoCoFL_{args.architecture}_{args.network}"
    print(file_string_prefix)

    list_of_configs = []

    for k in range(1, model.n_freezable_layers()):
        for i in range(0, model.n_freezable_layers()):
            config = list(range(0, model.n_freezable_layers()))

            try:
                for _ in range(k):
                    config.pop(i)
            except IndexError:
                continue
            list_of_configs.append(({"freeze": config}))

    # Ensure that full training has a low noise measurement
    for i in range(10):
        list_of_configs.append(({"freeze": []}))

    # Shuffle configurations
    random.shuffle(list_of_configs)

    # Repeat configurations based on input args
    list_of_configs *= int(args.epochs)

    # Prepare freezing configurations
    for kwargs in tqdm.tqdm(reversed(list_of_configs), total=len(list_of_configs)):
        kwargs = copy.deepcopy(kwargs)

        file_str = file_string_prefix + ".json"
        corrupt = False
        try:
            with open(file_str, "r") as fd:
                try:
                    res = json.load(fd)
                except json.decoder.JSONDecodeError:
                    corrupt = True
        except FileNotFoundError:
            corrupt = True

        if corrupt:
            with open(file_str, "w") as fd:
                json.dump([], fd)

        # Do measurements in isolated process
        q = Queue()
        p = Process(target=profile, args=(model, kwargs, model_state_dict_path, q))
        p.start()
        p.join()

        t_fw, t_bw, data_up, data_down, memory = q.get()

        try:
            with open(file_str, "r") as fd:
                res = json.load(fd)
        except FileNotFoundError:
            pass

        already_there = False

        for item in res:
            if item["freeze"] == kwargs["freeze"]:
                item["time_forward"] += t_fw
                item["time_backward"] += t_bw
                item["memory"] += [memory]
                already_there = True
        if not already_there:
            res.append({"freeze": list(sorted(kwargs["freeze"])),
                        "time_forward": t_fw,
                        "time_backward": t_bw,
                        "max": model.n_freezable_layers(),
                        "data_down": data_down,
                        "data_up": data_up,
                        "memory": [memory]})

        # Save new item in file
        with open(file_str, "w") as fd:
            json.dump(res, fd, indent=4)

    # Create a table with all memory/time/data values relative to full-training
    with open(file_str, "r") as fd:
        profiling_json = json.load(fd)

    table_string = f"profiling/table__CoCoFL_{args.architecture}_{args.network}.json"

    out = []
    max_time, max_up = 0.0, 0
    data = profiling_json
    data = list(sorted(data, key=lambda x: np.mean(x["time_forward"]) + np.mean(x["time_backward"]), reverse=True))

    max_config = list(filter(lambda x: x["freeze"] == [], data))
    if not max_config: assert False, "Maxium Configuration is missing. Abort!"
    max_config = max_config[0]

    max_time = np.mean(max_config["time_forward"]) + np.mean(max_config["time_backward"])
    max_up = max_config["data_up"]
    max_mem = np.mean(max_config["memory"])
    print("max_time: ", max_time, "max_up: ", max_up, "max_memory: ", max_mem)

    for config in data:
        res = {
            "freeze": config["freeze"],
            "time": round((np.mean(config["time_forward"]) + np.mean(config["time_backward"]))/max_time, 5),
            "data": round(config["data_up"]/max_up, 5),
            "memory": round(np.mean(config["memory"])/max_mem, 5),
        }
        if config["freeze"] == []:
            res.update({"__debug_max_time_in_s": round(max_time, 5),
                        "__debug_max_data_in_bytes": round((max_up*4)/(10**9), 5),
                        "__debug_max_mem_in_gb": round(max_mem, 5)})

        out.append(copy.deepcopy(res))

    with open(table_string, "w") as fd:
        json.dump(out, fd, indent=4)