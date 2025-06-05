import copy
import json


class Config:
    """
    A class for loading and accessing configuration parameters from a JSON file.
    """

    def __init__(
        self,
        model_name="RxnPredictor",
        batch_size=32,
        num_epochs=100,
        learning_rate=0.001,
        num_gconv_layers=6,
        num_gconv_units=256,
        num_dense_layers=3,
        num_dense_units=256,
        weight_decay=1e-5,
        dense_dropout=0,
        save_path="./experiments/checkpoints/"
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.num_gconv_layers = num_gconv_layers
        self.num_gconv_units = num_gconv_units
        self.num_dense_layers = num_dense_layers
        self.num_dense_units = num_dense_units
        self.weight_decay = weight_decay
        self.dense_dropout = dense_dropout
        self.save_path = save_path

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    @classmethod
    def load(cls, filepath):
        """
        Loads the configuration parameters from the specified JSON file.
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f)


if __name__ == '__main__':
    config = Config()
    config = config.load('./default_configs.json')
    print(config.to_dict())
    config.save('./default_configs.json')
