from abc import abstractmethod
from collections.abc import Callable
from functools import wraps

from luxonis_ml.data.loaders import BaseLoader
from luxonis_ml.typing import LoaderOutput

from luxonis_eval.registry import DATALOADERS_REGISTRY
from luxonis_eval.utils.utils import check_loader_output


def validate_loader_output(func: Callable) -> Callable:
    """
    Decorator to validate the output of a loader's __getitem__ method.

    Parameters
    ----------
    func : Callable
        The function to be decorated.

    Returns
    -------
    Callable
        The wrapped function with validation.
    """

    @wraps(func)
    def wrapper(self: BaseEvalLoader, idx: int) -> LoaderOutput:
        result = func(self, idx)
        try:
            check_loader_output(result)
        except TypeError as e:
            raise TypeError(
                f"Invalid loader output for {self.__class__.__name__} at index {idx}: {e}"
            ) from e
        return result

    return wrapper


class BaseEvalLoader(BaseLoader, register=False):
    REGISTRY = DATALOADERS_REGISTRY

    def __init_subclass__(cls, **kwargs):
        """
        Initialize subclass with validation for __getitem__ method.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to the parent class.
        """
        super().__init_subclass__(**kwargs)
        cls.__getitem__ = validate_loader_output(cls.__dict__["__getitem__"])

    @abstractmethod
    def get_class_mapping(self, **kwargs) -> tuple[dict, dict, dict]:
        """Returns the class mapping for the dataset.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments that may be used to customize the class mapping.

        Returns
        -------
        tuple[dict, dict, dict | None]
            LDF class map, native class map and class index map (if available).
        """
