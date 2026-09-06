import numpy as np
import torch

from src.models.architectures.aw_best.experiments_helpers import train_val_test_split
from src.models.architectures.aw_best.torch_lightning_modules import CancerNet
from src.models.base_model import BaseModel
from numpy import typing as npt


class AW_Best(BaseModel):
    def __init__(
        self,
        training_data_path: str = "data/data.csv",
        ckpt_path: str = "data/models/aw_best.ckpt",
    ):
        sub_config = {
            "n_h": 32,
            "n_l": 3,
        }
        
        self.net_config = {
            "name": "MultiHeadTaskRegressor",
            "config_list": [sub_config, sub_config, sub_config],
            "losses": ["L1", "L1", "L1"],
            "lr": 0.005,
            "main_loss": "L1",
            "margin_loss": True,
            "margin_loss_w": 5,
            "mode": "cnn_lstm",
            "series_len": 20,
            "w_losses": [1, 1, 1, 1],
        }
        
        self.train, self.val, self.test = train_val_test_split(training_data_path, "series")
        self.network = CancerNet.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            train_df=self.train,
            val_df=self.val,
            test_df=self.test,
            **self.net_config,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.eval().to(self.device)
        

    def predict(self, raw_protocol: npt.NDArray[np.float64]) -> float:
        columns = ["time", "dose", "time_gap"]
        scaled = raw_protocol.copy()
        for i, col in enumerate(columns):
            scaled[:, i] = (
                self.network.scalers[col]
                .transform(raw_protocol[:, i].reshape(-1, 1))
                .flatten()
            )

        x = torch.FloatTensor(scaled).unsqueeze(0).to(self.device)
        dummy_y = torch.zeros((1, 1), device=self.device)
        batch = (x, dummy_y, x, dummy_y)

        with torch.no_grad():
            output = self.network.model(batch)
            prediction = output[0, 0, 0].item() * self.network.target_scale

        return np.array([prediction], dtype=np.float64)[0]



