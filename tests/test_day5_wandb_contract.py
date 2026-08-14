import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _source_tree():
    return ast.parse((REPO_ROOT / "src" / "train.py").read_text(encoding="utf-8"))


def _train_function():
    return next(node for node in _source_tree().body if isinstance(node, ast.FunctionDef) and node.name == "train")


def _call_names(node, name):
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "wandb"
        and call.func.attr == name
    ]


def test_gitignore_protects_environment_files():
    rules = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in rules
    assert ".env.*" in rules
    assert "!.env.example" in rules


def test_env_example_contains_only_blank_secret_placeholders():
    env_example = REPO_ROOT / ".env.example"
    assert env_example.exists()
    assert env_example.read_text(encoding="utf-8").splitlines() == [
        "WANDB_API_KEY=",
        "HF_TOKEN=",
    ]


def test_wandb_is_declared_as_a_dependency():
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "wandb" in {line.strip() for line in requirements.splitlines()}


def test_train_has_verified_wandb_configuration_and_metric_logging():
    source = _source_tree()
    imports_wandb = any(
        isinstance(node, ast.Import)
        and any(alias.name == "wandb" for alias in node.names)
        for node in source.body
    )
    assert imports_wandb

    train = _train_function()
    init_calls = _call_names(train, "init")
    assert len(init_calls) == 1
    init = init_calls[0]
    project = next(keyword.value for keyword in init.keywords if keyword.arg == "project")
    assert isinstance(project, ast.Constant)
    assert project.value == "xai-medical-imaging"

    config = next(keyword.value for keyword in init.keywords if keyword.arg == "config")
    assert isinstance(config, ast.Dict)
    config_keys = {key.value for key in config.keys if isinstance(key, ast.Constant)}
    assert config_keys == {
        "model",
        "lr_head",
        "lr_finetune",
        "batch_size",
        "epochs",
        "warmup_epochs",
        "weight_decay",
        "num_classes",
        "dataset",
        "mean_test_auc",
    }

    logs = _call_names(train, "log")
    assert len(logs) == 2
    for log in logs:
        payload = log.args[0]
        assert isinstance(payload, ast.Dict)
        keys = {key.value for key in payload.keys if isinstance(key, ast.Constant)}
        assert keys == {"epoch", "phase", "train_loss", "train_auc", "val_loss", "val_auc", "lr"}

    assert len(_call_names(train, "finish")) == 1
