import numpy as np
import pytest
from depthai_nodes.message.creators import create_segmentation_message

from luxonis_eval.core.context import EvalContext
from luxonis_eval.engines.io import ModelSpec, TensorSpec
from luxonis_eval.metrics import F1Score, JaccardIndex


@pytest.mark.parametrize("metric_type", [F1Score, JaccardIndex])
@pytest.mark.parametrize(
    ("background", "include_background", "target", "prediction", "expected"),
    [
        (0, False, [0] * 9 + [1], [0] * 10, (0.0, 0.0)),
        (0, True, [0] * 9 + [1], [0] * 10, (0.9, 0.45)),
        (0, False, [0, 0, 1, 1], [1, 1, 1, 1], (1.0, 1.0)),
        (1, False, [1, 1, 0, 0], [0, 0, 0, 0], (1.0, 1.0)),
        (None, False, [0, 0, 1, 1], [1, 1, 1, 1], (0.5, 0.25)),
        (0, False, [0, 0, 0, 0], [1, 1, 1, 1], (0.0, float("nan"))),
    ],
)
def test_multiclass_background_exclusion(
    metric_type: type[F1Score] | type[JaccardIndex],
    background: int | None,
    include_background: bool,
    target: list[int],
    prediction: list[int],
    expected: tuple[float, float],
):
    metric = metric_type(num_classes=2, include_background=include_background)
    metric.attach_context(make_context(background))
    predictions = create_segmentation_message(
        np.array([prediction], dtype=np.uint8)
    )
    targets = {"/segmentation": np.array([target])}
    metric.update(predictions, targets)
    expected_score = expected[0 if metric_type is F1Score else 1]
    assert metric.compute()[metric_type.__name__] == pytest.approx(
        expected_score, nan_ok=True
    )
    metric.reset()
    metric.update(predictions, targets)
    assert metric.compute()[metric_type.__name__] == pytest.approx(
        expected_score, nan_ok=True
    )


@pytest.mark.parametrize("metric_type", [F1Score, JaccardIndex])
def test_background_only_sample_does_not_change_accumulated_score(
    metric_type: type[F1Score] | type[JaccardIndex],
):
    metric = metric_type(num_classes=2, include_background=False)
    metric.attach_context(make_context(0))
    predictions = create_segmentation_message(
        np.array([[1, 1]], dtype=np.uint8)
    )
    metric.update(predictions, {"/segmentation": np.array([[0, 0]])})
    metric.update(predictions, {"/segmentation": np.array([[1, 1]])})
    assert metric.compute()[metric_type.__name__] == pytest.approx(1.0)


@pytest.mark.parametrize("metric_type", [F1Score, JaccardIndex])
def test_binary_background_still_counts_false_positives(
    metric_type: type[F1Score] | type[JaccardIndex],
):
    metric = metric_type(include_background=False)
    metric.attach_context(make_context(0))
    # Native binary masks use class 0 for foreground and 255 for unassigned.
    metric.update(
        create_segmentation_message(np.array([[0, 0]], dtype=np.uint8)),
        {"/segmentation": np.array([[[0, 1]]])},
    )
    expected = 2 / 3 if metric_type is F1Score else 0.5
    assert metric.compute()[metric_type.__name__] == pytest.approx(expected)


def make_context(background: int | None) -> EvalContext:
    classes = {0: "class_0", 1: "class_1"}
    if background is not None:
        classes[background] = "background"
    return EvalContext(
        model_spec=ModelSpec(
            input=TensorSpec("image", (1, 3, 1, 10), layout="NCHW"),
            outputs=(TensorSpec("mask"),),
        ),
        class_map=classes,
        target_class_map=classes,
        class_index_map={0: 0, 1: 1},
        category_ids=(0, 1),
        target_background_index=background,
    )
