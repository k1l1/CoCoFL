from .fedavg import FedAvgEvaluationDevice, FedAvgServer, FedAvgDevice


class FedAvgDropDevicesServer(FedAvgServer):
    _device_class = FedAvgDevice
    _device_evaluation_class = FedAvgEvaluationDevice

    #!overrides
    def pre_round(self, round_n, rng):
        rand_selection_idxs_all = self.random_device_selection(self.n_devices, self.n_devices, rng)
        selection_idxs = [idx for idx in rand_selection_idxs_all
                          if any(self._device_constraints[idx].is_heterogeneous()) == False]
        assert len(selection_idxs) >= self.n_devices_per_round, "Error: Cant run since too many devices have to be dropped"
        return selection_idxs[0:self.n_devices_per_round]
