from abc import abstractmethod

from luxonis_ml.data.loaders import BaseLoader

from luxonis_eval.registry import DATALOADERS_REGISTRY


class BaseEvalLoader(BaseLoader, register=False):
    REGISTRY = DATALOADERS_REGISTRY

    @abstractmethod
    def get_class_mapping(self) -> tuple[dict, dict, dict]:
        """Returns the class mapping for the dataset.

        @rtype: tuple[dict, dict, dict]
        @return: Tuple of class mapping dictionaries (name_to_id, id_to_name, id_to_color).
        """
