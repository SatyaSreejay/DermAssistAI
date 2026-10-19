"""Phase 3 tests: model, transforms, dataset, metrics, class weights, checkpoints,
and a tiny CPU training + evaluation run on synthetic images.

No pretrained weights are downloaded (pretrained=False), and nothing here
produces a real result: the images are random noise.
"""

import json

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from scripts import evaluate, make_splits, train_baseline, validate_data  # noqa: E402
from src.data.dataset import MissingImagesError, SkinLesionDataset, load_split_frame  # noqa: E402
from src.data.image_index import build_image_index  # noqa: E402
from src.data.metadata import CLASS_NAMES  # noqa: E402
from src.evaluation.metrics import compute_metrics, plot_confusion_matrix  # noqa: E402
from src.models.resnet import build_resnet18, count_parameters, set_trainable, train_mode  # noqa: E402
from src.preprocessing.transforms import build_eval_transform, build_train_transform  # noqa: E402
from src.training.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from src.training.imbalance import class_weights  # noqa: E402

SMALL_INPUT = {"resize": 72, "image_size": 64, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


# --- model ---------------------------------------------------------------------------


def test_resnet18_outputs_seven_logits():
    model = build_resnet18(7, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(3, 3, 64, 64))
    assert out.shape == (3, 7)


def test_head_stage_trains_only_the_classifier():
    model = build_resnet18(7, pretrained=False, dropout=0.2)
    set_trainable(model, "head")
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith("fc.") for n in trainable)
    assert count_parameters(model)["trainable"] == 512 * 7 + 7


def test_layer4_stage_unfreezes_layer4_and_head_only():
    model = build_resnet18(7, pretrained=False)
    set_trainable(model, "layer4")
    trainable = {n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"layer4", "fc"}


def test_frozen_batchnorm_stays_in_eval_mode():
    model = build_resnet18(7, pretrained=False)
    set_trainable(model, "head")
    train_mode(model)
    assert not model.bn1.training and not model.layer4[0].bn1.training
    set_trainable(model, "layer4")
    train_mode(model)
    assert model.layer4[0].bn1.training and not model.bn1.training


def test_unknown_stage_raises():
    with pytest.raises(ValueError):
        set_trainable(build_resnet18(7, pretrained=False), "everything")


# --- transforms and dataset -----------------------------------------------------------


def test_transforms_produce_normalised_tensors():
    from PIL import Image

    image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (45, 60, 3), dtype=np.uint8))
    ev = build_eval_transform(SMALL_INPUT)
    a, b = ev(image), ev(image)
    assert a.shape == (3, 64, 64) and torch.equal(a, b)  # evaluation is deterministic
    tr = build_train_transform(SMALL_INPUT, {"rotation_degrees": 10, "color_jitter": 0.1})
    assert tr(image).shape == (3, 64, 64)


@pytest.fixture
def splits_csv(config_file, tmp_path):
    validate_data.main(["--config", str(config_file)])
    make_splits.main(["--config", str(config_file)])
    return tmp_path / "processed" / "splits.csv"


def test_dataset_returns_tensor_and_label(splits_csv, synthetic):
    frame = load_split_frame(splits_csv, "val")
    ds = SkinLesionDataset(frame, build_image_index([synthetic.image_dir]), build_eval_transform(SMALL_INPUT))
    x, y = ds[0]
    assert x.shape == (3, 64, 64) and 0 <= y < len(CLASS_NAMES)
    assert len(ds) == len(frame)


def test_dataset_refuses_missing_images(splits_csv, synthetic):
    frame = load_split_frame(splits_csv, "val")
    next(synthetic.image_dir.rglob(f"{frame['image_id'].iloc[0]}.jpg")).unlink()
    with pytest.raises(MissingImagesError, match=frame["image_id"].iloc[0]):
        SkinLesionDataset(frame, build_image_index([synthetic.image_dir]), build_eval_transform(SMALL_INPUT))


def test_split_frame_subset_and_errors(splits_csv):
    full = load_split_frame(splits_csv, "train", "train_100pct")
    small = load_split_frame(splits_csv, "train", "train_10pct")
    assert len(small) < len(full)
    with pytest.raises(ValueError, match="Available"):
        load_split_frame(splits_csv, "train", "train_33pct")
    with pytest.raises(ValueError):
        load_split_frame(splits_csv, "val", "train_10pct")


# --- metrics and class weights --------------------------------------------------------


def test_metrics_perfect_and_known_values():
    y = np.array([0, 0, 1, 2])
    perfect = np.eye(3)[y]
    m = compute_metrics(y, perfect, ["a", "b", "c"])
    assert m["accuracy"] == m["macro_f1"] == m["roc_auc_macro_ovr"] == 1.0

    # predict class 0 for everything: accuracy 0.5, but macro recall only 1/3
    always0 = np.tile([0.8, 0.1, 0.1], (4, 1))
    m = compute_metrics(y, always0, ["a", "b", "c"])
    assert m["accuracy"] == 0.5
    assert m["macro_recall"] == pytest.approx(1 / 3)
    assert m["per_class"]["b"]["recall"] == 0.0
    assert m["confusion_matrix"] == [[2, 0, 0], [1, 0, 0], [1, 0, 0]]


def test_roc_auc_undefined_when_class_absent():
    y = np.array([0, 1, 0, 1])
    probs = np.array([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.6, 0.3, 0.1], [0.3, 0.6, 0.1]])
    m = compute_metrics(y, probs, ["a", "b", "c"])
    assert m["roc_auc_macro_ovr"] is None and "c" in m["roc_auc_note"]
    json.dumps(m)


def test_metrics_shape_check():
    with pytest.raises(ValueError):
        compute_metrics([0, 1], np.ones((2, 3)) / 3, ["a", "b"])


def test_confusion_matrix_plot(tmp_path):
    path = plot_confusion_matrix([[3, 1], [0, 2]], ["a", "b"], tmp_path / "cm.png")
    assert path.stat().st_size > 0


def test_class_weights():
    labels = [0, 0, 0, 1]
    w = class_weights(labels, 2, "balanced")
    assert w.tolist() == pytest.approx([4 / (2 * 3), 4 / (2 * 1)])
    assert class_weights(labels, 2, "sqrt").tolist() == pytest.approx(np.sqrt(w).tolist())
    assert class_weights(labels, 2, "none") is None
    with pytest.raises(ValueError, match="no training examples"):
        class_weights([0, 0], 2, "balanced")


# --- checkpoint ------------------------------------------------------------------------


def test_checkpoint_round_trip(tmp_path):
    model = build_resnet18(7, pretrained=False, dropout=0.2).eval()
    meta = {"class_names": list(CLASS_NAMES), "model": {"name": "resnet18", "dropout": 0.2}, "input": SMALL_INPUT}
    path = save_checkpoint(tmp_path / "m.pt", model, meta)
    loaded, loaded_meta = load_checkpoint(path)
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x))
    assert loaded_meta["class_names"] == list(CLASS_NAMES)
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model, meta)
    with pytest.raises(ValueError):
        save_checkpoint(tmp_path / "bad.pt", model, {"model": {}})


# --- end to end: tiny training run and evaluation --------------------------------------


def _baseline_config(tmp_path, synthetic, splits_csv, stages):
    cfg = yaml.safe_load((train_baseline.PROJECT_ROOT / "configs" / "baseline.yaml").read_text())
    cfg["data"] |= {"splits_csv": str(splits_csv), "image_dirs": [str(synthetic.image_dir)]}
    cfg["input"] = SMALL_INPUT
    cfg["training"] |= {"batch_size": 8, "num_workers": 0, "stages": stages, "early_stopping_patience": 1}
    cfg["output"]["artifacts_dir"] = str(tmp_path / "artifacts")
    path = tmp_path / "baseline.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_check_only_does_not_train(tmp_path, synthetic, splits_csv):
    cfg = _baseline_config(tmp_path, synthetic, splits_csv, [
        {"name": "head", "trainable": "head", "epochs": 1, "lr": 1e-3}])
    assert train_baseline.main(["--config", str(cfg), "--check-only", "--no-pretrained", "--device", "cpu"]) == 0
    assert not (tmp_path / "artifacts").exists()


def test_train_then_evaluate_on_cpu(tmp_path, synthetic, splits_csv):
    stages = [
        {"name": "head", "trainable": "head", "epochs": 2, "lr": 1e-3, "weight_decay": 1e-4},
        {"name": "finetune_layer4", "trainable": "layer4", "epochs": 1, "lr": 1e-4},
    ]
    cfg = _baseline_config(tmp_path, synthetic, splits_csv, stages)
    args = ["--config", str(cfg), "--no-pretrained", "--device", "cpu", "--run-name", "tiny"]
    assert train_baseline.main(args) == 0
    run = tmp_path / "artifacts" / "tiny"
    for name in ("best.pt", "history.csv", "run_config.yaml", "train_summary.json", "val_metrics.json",
                 "val_confusion_matrix.png"):
        assert (run / name).is_file(), name
    summary = json.loads((run / "train_summary.json").read_text())
    assert summary["epochs_run"] >= 2 and summary["class_weighting"] == "balanced"
    _, meta = load_checkpoint(run / "best.pt")
    assert meta["class_names"] == list(CLASS_NAMES) and meta["input"] == SMALL_INPUT
    assert meta["splits_csv_sha256"]

    with pytest.raises(FileExistsError):  # a finished run is never silently replaced
        train_baseline.main(args)

    assert evaluate.main(["--checkpoint", str(run / "best.pt"), "--split", "val", "--splits", str(splits_csv),
                          "--image-dir", str(synthetic.image_dir), "--device", "cpu", "--num-workers", "0"]) == 0
    metrics = json.loads((run / "eval_val" / "metrics.json").read_text())
    assert metrics["split"] == "val" and metrics["n_samples"] > 0
    assert len(metrics["confusion_matrix"]) == len(CLASS_NAMES)
    assert metrics["inference"]["ms_per_image"] > 0
    preds = (run / "eval_val" / "predictions.csv").read_text().splitlines()
    assert preds[0].startswith("image_id,true,pred,p_akiec")
