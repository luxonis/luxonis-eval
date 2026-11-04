import numpy as np
import pytest
import torch
from depthai_nodes.message.creators import create_segmentation_message
from torchmetrics.segmentation import DiceScore, MeanIoU

from luxonis_eval.core.context import EvalContext
from luxonis_eval.engines.io import ModelSpec, TensorSpec
from luxonis_eval.engines.onnx_engine import ONNXEngineOutput
from luxonis_eval.metrics import DiceCoefficient, F1Score, JaccardIndex, MIoU
from luxonis_eval.metrics.utils import prepare_segmentation_metric_inputs
from luxonis_eval.parsers import SegmentationParser


@pytest.mark.parametrize("metric_type", [F1Score, JaccardIndex])
@pytest.mark.parametrize(
    ("background", "ignore_index", "target", "prediction", "expected"),
    [
        (0, 0, [0] * 9 + [1], [0] * 10, (0.0, 0.0)),
        (0, None, [0] * 9 + [1], [0] * 10, (0.9, 0.45)),
        (0, 0, [0, 0, 1, 1], [1, 1, 1, 1], (1.0, 1.0)),
        (1, 1, [1, 1, 0, 0], [0, 0, 0, 0], (1.0, 1.0)),
        (None, None, [0, 0, 1, 1], [1, 1, 1, 1], (0.5, 0.25)),
        (0, 0, [0, 0, 0, 0], [1, 1, 1, 1], (0.0, float("nan"))),
    ],
)
def test_multiclass_ignore_index(
    *,
    metric_type: type[F1Score] | type[JaccardIndex],
    background: int | None,
    ignore_index: int | None,
    target: list[int],
    prediction: list[int],
    expected: tuple[float, float],
):
    metric = metric_type(num_classes=2, ignore_index=ignore_index)
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
    metric = metric_type(num_classes=2, ignore_index=0)
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
    metric = metric_type()
    metric.attach_context(make_context(0))
    # Native binary masks use class 0 for foreground and 255 for unassigned.
    metric.update(
        create_segmentation_message(np.array([[0, 0]], dtype=np.uint8)),
        {"/segmentation": np.array([[[0, 1]]])},
    )
    expected = 2 / 3 if metric_type is F1Score else 0.5
    assert metric.compute()[metric_type.__name__] == pytest.approx(expected)


@pytest.mark.parametrize("metric_type", [F1Score, JaccardIndex])
def test_binary_ignore_index_is_explicit(
    metric_type: type[F1Score] | type[JaccardIndex],
):
    metric = metric_type(ignore_index=0)
    metric.attach_context(make_context(0))
    metric.update(
        create_segmentation_message(np.array([[0, 0]], dtype=np.uint8)),
        {"/segmentation": np.array([[[0, 1]]])},
    )
    assert metric.compute()[metric_type.__name__] == pytest.approx(1.0)


@pytest.mark.parametrize("metric_type", [F1Score, JaccardIndex])
def test_include_background_requires_migration(
    metric_type: type[F1Score] | type[JaccardIndex],
):
    with pytest.raises(ValueError, match="uses ignore_index"):
        metric_type(include_background=False)


@pytest.mark.parametrize("background_class", [False, True])
@pytest.mark.parametrize("include_background", [False, True])
@pytest.mark.parametrize("metric_type", [MIoU, DiceCoefficient])
def test_parser_background_predictions_count_as_false_negatives(
    background_class: bool,
    include_background: bool,
    metric_type: type[MIoU] | type[DiceCoefficient],
):
    # Both spatial axes exceed the class count for the upstream layout heuristic.
    logits = np.array(
        [[[2, 0, 2, 0]] * 3, [[0, 2, 0, 2]] * 3], dtype=np.float32
    )
    predictions = SegmentationParser(background_class=background_class).parse(
        ONNXEngineOutput({"mask": logits})
    )
    original_mask = predictions.getCvMask().copy()
    assert original_mask[0].tolist() == [
        255 if background_class else 0,
        1,
        255 if background_class else 0,
        1,
    ]
    metric = metric_type(num_classes=2, include_background=include_background)
    metric.attach_context(make_context(0))
    metric.update(predictions, {"/segmentation": np.ones((3, 4), dtype=int)})
    if metric_type is MIoU:
        expected = 0.25 if include_background else 0.5
        key = "mIoU"
    else:
        expected = 0.5 if include_background else 2 / 3
        key = "Dice Score"
    assert metric.compute()[key] == pytest.approx(expected)
    np.testing.assert_array_equal(predictions.getCvMask(), original_mask)


@pytest.mark.parametrize("background_class", [False, True])
@pytest.mark.parametrize("ignore_index", [None, 0])
@pytest.mark.parametrize("metric_type", [F1Score, JaccardIndex])
def test_parser_background_is_normalized_before_ignore_index(
    background_class: bool,
    ignore_index: int | None,
    metric_type: type[F1Score] | type[JaccardIndex],
):
    logits = np.array(
        [[[2, 0, 2, 0]] * 3, [[0, 2, 0, 2]] * 3], dtype=np.float32
    )
    predictions = SegmentationParser(background_class=background_class).parse(
        ONNXEngineOutput({"mask": logits})
    )
    metric = metric_type(num_classes=2, ignore_index=ignore_index)
    metric.attach_context(make_context(0))
    metric.update(predictions, {"/segmentation": np.ones((3, 4), dtype=int)})
    expected = 0.5
    if metric_type is JaccardIndex and ignore_index is None:
        expected = 0.25
    assert metric.compute()[metric_type.__name__] == pytest.approx(expected)


@pytest.mark.parametrize("metric_type", [F1Score, JaccardIndex])
def test_single_channel_parser_keeps_binary_encoding(
    metric_type: type[F1Score] | type[JaccardIndex],
):
    logits = np.array([[[2, -2, 2, -2]] * 3], dtype=np.float32)
    predictions = SegmentationParser().parse(
        ONNXEngineOutput({"mask": logits})
    )
    metric = metric_type()
    metric.attach_context(make_context(None))
    metric.update(
        predictions, {"/segmentation": np.ones((1, 3, 4), dtype=int)}
    )
    expected = 2 / 3 if metric_type is F1Score else 0.5
    assert metric.compute()[metric_type.__name__] == pytest.approx(expected)


@pytest.mark.parametrize("include_background", [False, True])
@pytest.mark.parametrize("metric_type", [MIoU, DiceCoefficient])
def test_foreground_false_positives_on_background_are_scored(
    include_background: bool,
    metric_type: type[MIoU] | type[DiceCoefficient],
):
    metric = metric_type(num_classes=2, include_background=include_background)
    metric.attach_context(make_context(0))
    predictions = create_segmentation_message(np.ones((3, 4), dtype=np.uint8))
    targets = {"/segmentation": np.array([[0, 0, 1, 1]] * 3)}
    metric.update(predictions, targets)
    if metric_type is MIoU:
        expected = 0.25 if include_background else 0.5
        key = "mIoU"
    else:
        expected = 0.5 if include_background else 2 / 3
        key = "Dice Score"
    assert metric.compute()[key] == pytest.approx(expected)
    metric.reset()
    metric.update(predictions, targets)
    assert metric.compute()[key] == pytest.approx(expected)


@pytest.mark.parametrize("include_background", [False, True])
@pytest.mark.parametrize(
    ("metric_type", "input_format"),
    [
        (MIoU, "index"),
        (MIoU, "one-hot"),
        (MIoU, "mixed"),
        (DiceCoefficient, "index"),
        (DiceCoefficient, "one-hot"),
    ],
)
def test_segmentation_matches_training_torchmetrics_inputs(
    include_background: bool,
    input_format: str,
    metric_type: type[MIoU] | type[DiceCoefficient],
):
    metric = metric_type(
        num_classes=2,
        include_background=include_background,
        input_format=input_format,
    )
    reference_type = MeanIoU if metric_type is MIoU else DiceScore
    reference_params = {} if metric_type is MIoU else {"average": "micro"}
    reference = reference_type(
        num_classes=2,
        include_background=include_background,
        input_format="index",
        **reference_params,
    )
    metric.attach_context(make_context(0))
    # Different foreground counts per row expose accidental row-wise averaging.
    target = np.array([[1, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]])
    pred = np.array([[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    for prediction in (pred, target):
        metric.update(
            create_segmentation_message(prediction.astype(np.uint8)),
            {"/segmentation": target},
        )
        reference.update(
            torch.from_numpy(prediction).unsqueeze(0),
            torch.from_numpy(target).unsqueeze(0),
        )
    key = "mIoU" if metric_type is MIoU else "Dice Score"
    assert metric.compute()[key] == pytest.approx(float(reference.compute()))


@pytest.mark.parametrize("metric_type", [MIoU, DiceCoefficient])
def test_one_hot_targets_preserve_unassigned_and_overlapping_pixels(
    metric_type: type[MIoU] | type[DiceCoefficient],
):
    metric = metric_type(num_classes=2, input_format="one-hot")
    reference_type = MeanIoU if metric_type is MIoU else DiceScore
    reference_params = {} if metric_type is MIoU else {"average": "micro"}
    reference = reference_type(
        num_classes=2, input_format="one-hot", **reference_params
    )
    metric.attach_context(make_context(0))
    target = np.array([[[0, 1, 0]], [[0, 1, 1]]])
    pred = np.array([[0, 1, 1]])
    metric.update(
        create_segmentation_message(pred.astype(np.uint8)),
        {"/segmentation": target},
    )
    reference.update(
        torch.nn.functional.one_hot(torch.from_numpy(pred), 2)
        .movedim(-1, 0)
        .unsqueeze(0),
        torch.from_numpy(target).bool().unsqueeze(0),
    )
    key = "mIoU" if metric_type is MIoU else "Dice Score"
    assert metric.compute()[key] == pytest.approx(float(reference.compute()))


@pytest.mark.parametrize("class_index_map", [None, {0: 0, 1: 1}])
def test_native_unassigned_maps_to_background(
    class_index_map: dict[int, int] | None,
):
    prepared = prepare_segmentation_metric_inputs(
        create_segmentation_message(np.array([[255, 1]], dtype=np.uint8)),
        {"/segmentation": np.array([[1, 1]])},
        target_bg=0,
        class_index_map=class_index_map,
    )
    np.testing.assert_array_equal(prepared.pred_mask, [[0, 1]])


def test_unassigned_background_is_not_remapped_twice():
    # Dataset IDs: cat=0, dog=1, background=2. Model IDs: background=0, cat=1, dog=2.
    prepared = prepare_segmentation_metric_inputs(
        create_segmentation_message(np.array([[255, 1, 2]], dtype=np.uint8)),
        {"/segmentation": np.array([[0, 0, 1]])},
        target_bg=2,
        class_index_map={0: 1, 1: 2, 2: 0},
    )
    np.testing.assert_array_equal(prepared.pred_mask, [[2, 0, 1]])


def test_unmapped_classes_are_not_treated_as_background():
    with pytest.raises(ValueError, match=r"no dataset class mapping: \[2\]"):
        prepare_segmentation_metric_inputs(
            create_segmentation_message(np.array([[255, 2]], dtype=np.uint8)),
            {"/segmentation": np.array([[1, 1]])},
            target_bg=0,
            class_index_map={0: 0, 1: 1},
        )


def test_unassigned_without_dataset_background_is_rejected():
    with pytest.raises(ValueError, match="dataset has no background class"):
        prepare_segmentation_metric_inputs(
            create_segmentation_message(np.array([[255, 1]], dtype=np.uint8)),
            {"/segmentation": np.array([[1, 1]])},
        )


@pytest.mark.parametrize("metric_type", [MIoU, DiceCoefficient])
def test_per_class_names_exclude_background(
    metric_type: type[MIoU] | type[DiceCoefficient],
):
    params = {"per_class": True} if metric_type is MIoU else {"average": None}
    metric = metric_type(num_classes=3, include_background=False, **params)
    classes = {0: "background", 1: "cat", 2: "dog"}
    context = EvalContext(
        model_spec=make_context(0).model_spec,
        class_map=classes,
        target_class_map=classes,
        class_index_map={0: 0, 1: 1, 2: 2},
        category_ids=(0, 1, 2),
        target_background_index=0,
    )
    metric.attach_context(context)
    metric.update(
        create_segmentation_message(
            np.array([[255, 1, 2]] * 2, dtype=np.uint8)
        ),
        {"/segmentation": np.array([[1, 1, 2]] * 2)},
    )
    if metric_type is MIoU:
        assert metric.compute() == pytest.approx(
            {"mIoU": 0.75, "mIoU (cat)": 0.5, "mIoU (dog)": 1.0}
        )
    else:
        assert metric.compute() == pytest.approx(
            {"Dice Score_cat": 2 / 3, "Dice Score_dog": 1.0}
        )


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
