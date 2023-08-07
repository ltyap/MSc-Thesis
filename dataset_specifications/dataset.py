import collections

LabelledData = collections.namedtuple("LabelledData",["x","y"])

class Dataset():
    def __init__(self) -> None:
        self.name = 'dataset name'
        self.require_path = True
        self.synthetic = True
        self.x_dim = 1
        self.y_dim = 1
    def load_data():
        return None
    def plot_test_data():
        return None
        