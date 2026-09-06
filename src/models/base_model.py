from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def load(self,
             ckpt_path: str) -> None:
        pass

    @abstractmethod
    def predict(self,
                raw_protocol):
        pass