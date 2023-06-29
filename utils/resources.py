import torch


class Constant():
    def __init__(self, const_value):
        self.value = const_value
        pass

    def is_heterogeneous(self):
        if self.value != 1.0:
            return True
        else:
            return False

    def __call__(self):
        return self.value

    def __repr__(self):
        return f"c:{self.value}"


class Uniform():
    def __init__(self, low, high):
        self.low = low
        self.high = high
        assert self.high > self.low, "Error in uniform"

    def is_heterogeneous(self):
        return True

    def __call__(self):
        res = float((self.high - self.low)*torch.rand(1) + self.low)
        return res

    def __repr__(self):
        return f"u:{self.low}/{self.high}"


class DeviceResources():
    def __init__(self):
        self._time_function = None
        self._data_function = None
        self._memory_function = None

    def set_all(self, function_time, function_data, function_memory):
        self.set_time_selection_F(function_time)
        self.set_data_selection_F(function_data)
        self.set_memory_selection_F(function_memory)

    def set_time_selection_F(self, function):
        self._time_function = function

    def set_data_selection_F(self, function):
        self._data_function = function

    def set_memory_selection_F(self, function):
        self._memory_function = function

    def get_time(self):
        return self._time_function()

    def get_data(self):
        return self._data_function()

    def get_memory(self):
        return self._memory_function()

    def __repr__(self) -> str:
        out = "[" + "t->" + self._time_function.__str__()
        out += " d->" + self._data_function.__str__()
        out += " m->" + self._memory_function.__str__() + "]"
        return out

    def is_heterogeneous(self):
        res = [self._time_function.is_heterogeneous(),
                self._data_function.is_heterogeneous(),
                self._memory_function.is_heterogeneous()]
        return res
