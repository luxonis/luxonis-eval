from abc import abstractmethod
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING

from luxonis_ml.data.loaders import BaseLoader
from luxonis_ml.typing import LoaderOutput

from luxonis_eval.engines.base_engine import ModelSpec
from luxonis_eval.registry import DATALOADERS_REGISTRY
from luxonis_eval.utils.utils import check_loader_classes, check_loader_output

if TYPE_CHECKING:
    from luxonis_eval.config.nn_archive import NNArchiveConfig


class BaseEvalLoader(BaseLoader, register=False):
    """Base class for custom evaluation loaders.

    Custom loaders are expected to return image arrays in image layout:
    `HWC` for color images and `HW` for grayscale images. Engines own any
    backend-specific tensor conversion such as `HWC -> NCHW`.
    """

    REGISTRY = DATALOADERS_REGISTRY

    def __init__(
        self,
        model_spec: ModelSpec,
        nn_archive_cfg: "NNArchiveConfig | None" = None,
        **kwargs,
    ):
        self.model_spec = model_spec
        self.nn_archive_cfg = nn_archive_cfg
        self.classes = self.load_classes()
        try:
            check_loader_classes(self.classes)
        except TypeError as e:
            raise TypeError(
                f"Invalid loader classes for {self.__class__.__name__}: {e}"
            ) from e
        super().__init__(**kwargs)

    def __init_subclass__(cls, **kwargs):
        """Initialize subclass with validation for __getitem__ method.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to the parent class.
        """
        super().__init_subclass__(**kwargs)
        cls.__getitem__ = validate_loader_output(cls.__dict__["__getitem__"])

    @abstractmethod
    def load_classes(self) -> dict[str, int]:
        """Loads and returns the class mapping for the dataset. This
        method is called once during __init__ and its return value is
        assigned to self.classes. Subclasses must implement this method
        to provide a mapping of class names to their integer indices.

        Returns
        -------
        dict[str, int]
            A mapping of class name to class index, e.g. {"cat": 0, "dog": 1}.
        """

    @abstractmethod
    def get_class_mapping(
        self, **kwargs
    ) -> tuple[dict[int, str], dict[int, str], dict[int, int]]:
        """Returns the LDF class map, native class map, and class index
        map.

        The LDF class map reflects how classes are indexed within LuxonisML's
        data format (LDF). The native class map reflects the original
        class-to-index mapping the model was trained on (e.g. COCO ordering).
        The class index map bridges the two by mapping each LDF index to its
        corresponding native index, allowing correct alignment of predictions
        against ground-truth annotations.

        When implementing this method for a LuxonisLoader-backed dataset,
        the LDF and native class maps may differ if the model was trained with
        a different class order than the dataset metadata. In that case, the
        class index map must explicitly encode the remapping (e.g.
        {0: 3, 1: 0, ...}). If the model was trained using the same LDF class
        order, the native class map may be identical to the LDF class map and
        the class index map can be an identity mapping.

        When implementing this method for a custom dataset that inherits
        directly from the BaseEvalLoader class, the LDF and native class maps
        should be identical — both derived from self.classes — and the
        class index map should be an identity mapping ({0: 0, 1: 1, ...}).

        Parameters
        ----------
        **kwargs
            Additional keyword arguments that may be used to customize the
            class mapping.

        Returns
        -------
        tuple[dict[int, str], dict[int, str], dict[int, int]]
            A 3-tuple of:
            - LDF class map (dict[int, str]): LDF index to class name.
            - Native class map (dict[int, str]): original index the
              model was trained on to class name.
            - Class index map (dict[int, int]): mapping from each LDF
              index to its corresponding native index.
        """


def validate_loader_output(func: Callable) -> Callable:
    """Decorator to validate the output of a loader's __getitem__
    method.

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
