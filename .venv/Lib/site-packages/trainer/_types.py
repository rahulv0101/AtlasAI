from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeAlias, TypedDict, TypeVar

import torch

if TYPE_CHECKING:
    import matplotlib
    import numpy.typing as npt
    import plotly

    from trainer.trainer import Trainer

_T = TypeVar("_T")

ValueListDict: TypeAlias = _T | list[_T] | dict[str, _T]

Audio: TypeAlias = "npt.NDArray[Any]"
Figure: TypeAlias = "matplotlib.figure.Figure | plotly.graph_objects.Figure"
LRScheduler: TypeAlias = torch.optim.lr_scheduler._LRScheduler

Callback: TypeAlias = Callable[["Trainer"], None]


class LossDict(TypedDict):
    train_loss: float
    eval_loss: float | None
