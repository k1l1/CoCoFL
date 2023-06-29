from abc import ABC
import torch
import random
import copy
import json
import logging
from sklearn.metrics import f1_score
import numpy as np
import tqdm
import sys


class FedAvgDevice():
    def __init__(self, device_id):
        self._device_id = device_id

        # model related
        self._model = None
        self._model_kwargs = None
        self._model_class = None

        # data related
        self._test_data = None
        self._train_data = None
        self._batch_size_train = 32

        # training related
        self._optimizer = None
        self._optimizer_kwargs = None
        self.lr = None
        self.resources = None

        self._torch_device = None

    def set_model(self, model_class, kwargs):
        self._model_kwargs = kwargs
        self._model_class = model_class

    def init_model(self):
        self._model = self._model_class(**self._model_kwargs)

    def del_model(self):
        self._model = None

    def set_train_data(self, dataset):
        self._train_data = dataset

    def set_torch_device(self, torch_device):
        self._torch_device = torch_device

    def set_optimizer(self, optimizer, optimizer_kwargs):
        self._optimizer = optimizer
        self._optimizer_kwargs = optimizer_kwargs

    def get_model_state_dict(self):
        assert self._model is not None, "Device has not NN model"
        return self._model.state_dict()

    def set_model_state_dict(self, model_dict, strict=True):
        self._model.load_state_dict(copy.deepcopy(model_dict), strict=strict)

    def _check_trainable(self):
        assert self._model is not None, "device has no NN model"
        assert self._train_data is not None, "device has no training dataset"
        assert self._torch_device is not None, "No torch_device is set"
        assert self._optimizer is not None, "No optimizer is set"

    @staticmethod
    def assert_if_nan(*tensors):
        for tensor in tensors:
            if torch.isnan(tensor).any():
                print("Error: loss got NaN")
                assert False, ""

    def device_training(self):
        self._check_trainable()
        self._train()
        self._model.to('cpu')

    def _train(self, n_epochs=1):
        self._model.to(self._torch_device)
        self._model.train()

        trainloader = torch.utils.data.DataLoader(self._train_data,
                                                  batch_size=self._batch_size_train,
                                                  shuffle=True, pin_memory=True)

        optimizer = self._optimizer(self._model.parameters(), lr=self.lr, **self._optimizer_kwargs)
        for _ in range(n_epochs):
            for _, (inputs, labels) in enumerate(trainloader):
                inputs, labels = inputs.to(self._torch_device), labels.to(self._torch_device)

                optimizer.zero_grad()
                output = self._model(inputs)
                loss = torch.nn.functional.cross_entropy(output, labels)
                self.assert_if_nan(loss)
                loss.backward()
                optimizer.step()


class FedAvgEvaluationDevice(FedAvgDevice):
    def __init__(self, device_id):
        super().__init__(device_id)

        # evaluation related
        self._test_data = None
        self._accuracy_test = None
        self.is_unbalanced = False
        self.batch_size_test = None

    def device_training(self):
        raise NotImplementedError("Evaluation Device does not have a round function.")

    def set_test_data(self, dataset):
        self._test_data = dataset
        return

    @staticmethod
    def correct_predictions(labels, outputs):
        res = (torch.argmax(outputs.cpu().detach(), axis=1) ==
               labels.cpu().detach()).sum()
        return res

    @staticmethod
    def correct_predictions_f1(labels, output):
        res = f1_score(labels.cpu(), torch.argmax(output.cpu(), axis=1), average='macro')*output.shape[0]
        return res

    def _check_testable(self):
        assert self._model is not None, "device has no NN model"
        assert self._test_data is not None, "device has no test dataset"
        assert self._torch_device is not None, "No torch_device is set"

    def test(self):
        self._model.to(self._torch_device)
        self._model.eval()

        self._check_testable()
        assert not self._test_data.dataset.train, "Wrong dataset for testing."

        testloader = torch.utils.data.DataLoader(self._test_data, shuffle=True,
                                                 batch_size=self.batch_size_test, pin_memory=True)

        correct_predictions = 0

        # per-class accuracy calculations
        n_classes = len(self._test_data.dataset.classes)
        confusion_matrix = torch.zeros((n_classes, n_classes))

        with torch.no_grad():
            for _, (inputs, labels) in enumerate(testloader):
                inputs, labels = inputs.to(self._torch_device), labels.to(self._torch_device)

                output = self._model(inputs)

                if not self.is_unbalanced:
                    correct_predictions += self.correct_predictions(labels, output)
                else:
                    correct_predictions += self.correct_predictions_f1(labels, output)

                # per-class accuracy calculations
                predictions = torch.argmax(output.cpu().detach(), axis=1)
                for l, p in zip(labels, predictions):
                    confusion_matrix[int(l), int(p)] += 1

            self._accuracy_test = correct_predictions/len(self._test_data)

            # per-class accuarcy calculations
            per_class_accuracy = confusion_matrix.diag()/confusion_matrix.sum(1)
            logging.info(f'EVAL: {per_class_accuracy}')


class FedAvgServer(ABC):
    _device_class = FedAvgDevice
    _device_evaluation_class = FedAvgEvaluationDevice
    is_unbalanced = False

    def __init__(self, storage_path) -> None:
        super().__init__()
        self._devices_list = []
        self._storage_path = storage_path

        # general parameters
        self.torch_device = None
        self.n_rounds = 0
        self.n_devices = None
        self.n_devices_per_round = None

        self._device_constraints = None
        self._global_model = None

        # plotting
        self._plotting_function = None
        self._plotting_args = None

        # measurements
        self._measurements_dict = {}
        self._measurements_dict['accuracy'] = []
        self._measurements_dict['data_upload'] = []

        # optimizer
        self._optimizer = None
        self._optimizer_kwargs = None
        self.lr = None
        self.lr_schedule = None

        # data related
        self._test_data = None
        self._train_data = None
        self.split_function = None

        self._seed = None

    def set_seed(self, seed):
        torch.manual_seed(seed)
        random.seed(seed)
        self._seed = seed

    def set_optimizer(self, optimizer, optimizer_kwargs):
        self._optimizer = optimizer
        self._optimizer_kwargs = optimizer_kwargs

    def set_plotting_callback(self, function, args):
        self._plotting_function = function
        self._plotting_args = args

    def set_device_constraints(self, device_constraints):
        self._device_constraints = device_constraints

    @staticmethod
    def random_device_selection(n_devices, n_devices_per_round, rng):
        return rng.permutation(n_devices)[0:n_devices_per_round].tolist()

    @staticmethod
    def count_data_footprint(state_dict):
        counted_bytes = 0
        for key in state_dict:
            param = state_dict[key]
            if isinstance(param, torch.Tensor):
                val = 4
                for i in range(len(param.shape)):
                    val *= param.shape[i]
                counted_bytes += val
        return counted_bytes

    def save_dict_to_json(self, filename, input_dict):
        with open(self._storage_path + "/" + filename, "w") as fd:
            json.dump(input_dict, fd, indent=4)

    def set_dataset(self, dataset, path, *args, **kwargs):
        train_data = dataset(path, train=True, *args, **kwargs)
        test_data = dataset(path, train=False, *args, **kwargs)

        self._train_data = torch.utils.data.Subset(train_data, torch.arange(0, len(train_data)))
        self._test_data = torch.utils.data.Subset(test_data, torch.arange(0, len(test_data)))
        return

    def set_model(self, model_list, kwargs_list):
        assert all(x == model_list[0] for x in model_list), "FedAvg requires all NN models to have the same type"
        self._model = model_list
        self._model_kwargs = kwargs_list

    def set_model_evaluation(self, model, kwargs):
        self._model_evaluation = model
        self._model_evaluation_kwargs = kwargs

    def learning_rate_scheduling(self, round_n):
        if self.lr_schedule is not None:
            for schedule in self.lr_schedule:
                assert schedule[1] < 1.0
                assert isinstance(schedule[0], int)
                if round_n == schedule[0]:
                    for device in self._devices_list:
                        device.lr = schedule[1]
                    log = f'[FEDAVG]: learning_rate reduction at ' + \
                        f'round {round_n} to {schedule[1]}'
                    logging.info(log)

    def check_device_data(self):
        for i in range(len(self._devices_list)):
            for j in range(len(self._devices_list)):
                if i != j:
                    assert set(self._devices_list[i]._train_data.indices.tolist()).isdisjoint(set(
                            self._devices_list[j]._train_data.indices.tolist())), "Devices do not exclusivly have access to their data!"

    @staticmethod
    def model_averaging(list_of_state_dicts, eval_device_dict=None):
        averaging_exceptions = ['num_batches_tracked', 'frozen']

        averaged_dict = copy.deepcopy(list_of_state_dicts[0])
        for key in averaged_dict:
            if all(module_name not in key for module_name in averaging_exceptions):
                averaged_dict[key] = torch.mean(torch.stack([state_dict[key]
                                                             for state_dict in list_of_state_dicts]), dim=0)
        averaged_dict = {k: v for k, v in averaged_dict.items() if all(module_name not in k for module_name in averaging_exceptions)}
        return averaged_dict

    # FedAvg performs random device selection
    def pre_round(self, round_n, rng):
        return self.random_device_selection(self.n_devices, self.n_devices_per_round, rng)

    def post_round(self, round_n, idxs):
        used_devices = [self._devices_list[i] for i in idxs]

        byte_count = 0
        for device in used_devices:
            byte_count += self.count_data_footprint(device.get_model_state_dict())

        self._measurements_dict['data_upload'].append([byte_count, round_n])

        averaged_model = self.model_averaging([dev.get_model_state_dict() for dev in used_devices],
                                                eval_device_dict=self._global_model)

        self._evaluation_device.set_model_state_dict(copy.deepcopy(averaged_model), strict=False)
        self._evaluation_device.test()
        acc = round(float(self._evaluation_device._accuracy_test), 4)
        logging.info(f"[FEDAVG]: Round: {round_n} Test accuracy: {acc}")

        self._measurements_dict['accuracy'].append([acc, round_n])
        self._global_model = averaged_model

    def initialize(self):
        assert not any([self.split_function,
                        self._train_data, self._test_data]) is None, "Uninitialized Values"

        idxs_list = self.split_function(self._train_data, self.n_devices)
        self._evaluation_device = self._device_evaluation_class(0)
        self._evaluation_device.set_model(self._model_evaluation, self._model_evaluation_kwargs)
        self._evaluation_device.init_model()
        self._evaluation_device.set_test_data(self._test_data)
        self._evaluation_device.set_torch_device(self.torch_device)
        self._evaluation_device.is_unbalanced = self.is_unbalanced
        self._evaluation_device.batch_size_test = 512

        self._devices_list = [self._device_class(i) for i in range(self.n_devices)]

        self._devices_list = [self._device_class(i) for i in range(self.n_devices)]

        for i, device in enumerate(self._devices_list):
            device.set_model(self._model[i], self._model_kwargs[i])
            device.set_train_data(torch.utils.data.Subset(self._train_data.dataset, idxs_list[i]))
            device.lr = self.lr
            device.set_optimizer(self._optimizer, self._optimizer_kwargs)
            device.set_torch_device(self.torch_device)

            if self._device_constraints is not None:
                device.resources = self._device_constraints[i]

        self._devices_list[0].init_model()
        self._global_model = copy.deepcopy(self._devices_list[0]._model.state_dict())
        self._devices_list[0].del_model()

    def run(self):
        self.check_device_data()
        print(f"#Samples on devices: {[len(dev._train_data) for dev in self._devices_list]}")
        logging.info(f"[FL_BASE]: #Samples on devices: {[len(dev._train_data) for dev in self._devices_list]}")

        rng = np.random.default_rng(self._seed)

        tbar = tqdm.tqdm(iterable=range(self.n_rounds), total=self.n_rounds, file=sys.stdout, disable=not self.progress_output)
        for round_n in tbar:

            # learning rate scheduling
            self.learning_rate_scheduling(round_n)

            # selection of devices
            idxs = self.pre_round(round_n, rng)

            # init NN models
            for dev_idx in idxs:
                self._devices_list[dev_idx].init_model()
                self._devices_list[dev_idx].set_model_state_dict(self._global_model)

            # perform training
            for dev_idx in idxs:
                self._devices_list[dev_idx].device_training()

            # knwoledge aggregation // global model gets set
            self.post_round(round_n, idxs)

            # del models
            for dev_idx in idxs:
                self._devices_list[dev_idx].del_model()

            # save accuracy dict
            self.save_dict_to_json('measurements.json', self._measurements_dict)

            if self.progress_output:
                tbar.set_description(f"round_n {round_n}, acc: {self._measurements_dict['accuracy'][round_n][0]}")
            else:
                print(f"round_n {round_n}, acc={self._measurements_dict['accuracy'][round_n][0]}")

            # plotting
            if (round_n % 25) == 0 and round_n != 0:
                if self._plotting_function is not None:
                    try:
                        self._plotting_function(self._plotting_args)
                    except:
                        print("Error plotting!")