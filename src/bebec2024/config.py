import importlib
import inspect
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import toml

import bebec2024
from bebec2024.pipeline import CharacterizationPipeline

default_session_dir = Path(bebec2024.__file__).parent / "sessions"


@dataclass
class ConfigModel:
    cls_name: str
    args: dict

    def create_instance(self):
        if not hasattr(self, "_instance"):
            cls_name = getattr(
                importlib.import_module("bebec2024.models"), self.cls_name
            )
            self._instance = cls_name(**self.args)
        return self._instance


@dataclass
class ConfigPipeline:
    cls_name: str
    args: dict

    def create_instance(self):
        if not hasattr(self, "_instance"):
            cls_name = getattr(
                importlib.import_module("bebec2024.pipeline"), self.cls_name
            )
            self._instance = cls_name(**self.args)
        return self._instance


@dataclass
class ConfigSplit:
    split: str
    size: int
    f: float = None
    he_min: float = None
    he_max: float = None
    num: int = 0
    batchsize: int = 1
    shuffle: bool = False
    shuffle_buffer_size: int = 1
    cache: bool = False
    prefetch_size: int = None
    seed: int = 1
    progress_bar: bool = False
    start_idx: int = 0
    num: int = 0
    features: list[str] = None


@dataclass
class ConfigDataset:
    cls_name: str
    args: dict
    config_cls_name: str = None
    config_args: dict = None
    pipeline: ConfigPipeline = None
    training: ConfigSplit = None
    validation: ConfigSplit = None
    test: ConfigSplit = None

    def create_instance(self):
        if not hasattr(self, "_instance"):
            dataset_cls = getattr(
                importlib.import_module("bebec2024.datasets"), self.cls_name
            )
            if self.config_cls_name is None:
                self._instance = dataset_cls(**self.args)
            else:
                config_cls = getattr(
                    importlib.import_module("bebec2024.datasets"), self.config_cls_name
                )
                config = config_cls(**self.config_args)
                self._instance = dataset_cls(config=config, **self.args)
        return self._instance

    def get_training_pipeline(self):
        pipeline = self.pipeline.create_instance()
        dataset = self.create_instance()
        return pipeline(dataset, **asdict(self.training))

    def get_training_generator(self, **kwargs):
        dataset = self.create_instance()
        dataset_kwargs = {
            "split": self.training.split,
            "size": self.training.size,
            "he_min": self.training.he_min,
            "he_max": self.training.he_max,
            "progress_bar": self.training.progress_bar,
            "start_idx": self.training.start_idx,
            "num": self.training.num,
            "features": self.training.features,
        }
        dataset_kwargs = CharacterizationPipeline.filter_kwargs(
            dataset, **dataset_kwargs
        )
        dataset_kwargs.update(kwargs)
        return dataset.generate(**dataset_kwargs)

    def get_validation_pipeline(self):
        pipeline = self.pipeline.create_instance()
        dataset = self.create_instance()
        return pipeline(dataset, **asdict(self.validation))

    def get_validation_generator(self, **kwargs):
        dataset = self.create_instance()
        dataset_kwargs = {
            "split": self.validation.split,
            "size": self.validation.size,
            "he_min": self.validation.he_min,
            "he_max": self.validation.he_max,
            "progress_bar": self.validation.progress_bar,
            "start_idx": self.validation.start_idx,
            "num": self.validation.num,
            "features": self.validation.features,
        }
        dataset_kwargs = CharacterizationPipeline.filter_kwargs(
            dataset, **dataset_kwargs
        )
        dataset_kwargs.update(kwargs)
        return dataset.generate(**dataset_kwargs)

    def get_test_pipeline(self):
        pipeline = self.pipeline.create_instance()
        dataset = self.create_instance()
        return pipeline(dataset, **asdict(self.test))

    def get_test_generator(self, **kwargs):
        dataset = self.create_instance()
        dataset_kwargs = {
            "split": self.test.split,
            "size": self.test.size,
            "he_min": self.test.he_min,
            "he_max": self.test.he_max,
            "progress_bar": self.test.progress_bar,
            "start_idx": self.test.start_idx,
            "num": self.test.num,
            "features": self.test.features,
        }
        dataset_kwargs = CharacterizationPipeline.filter_kwargs(
            dataset, **dataset_kwargs
        )
        dataset_kwargs.update(kwargs)
        return dataset.generate(**dataset_kwargs)


@dataclass
class ConfigBase:
    datasets: list[ConfigDataset]
    model: ConfigModel
    session_dir: str = str(default_session_dir.resolve())
    log_dir_name: str = "tensorboard"
    ckpt_dir_name: str = "ckpt"
    sessionname: str = None
    # initialized from init
    log_dir: Path = field(init=False)
    ckpt_dir: Path = field(init=False)
    learning_rate: float = 1
    weight_decay: float = 1e-5
    use_lr_scheduler: bool = False
    use_wd_scheduler: bool = False
    epochs: int = 250
    steps_per_epoch: int = 2000
    ckpt: bool = True
    tensorboard: bool = True

    def __post_init__(self):
        self.set_sessionname()
        self.set_log_dir()
        self.set_ckpt_dir()

    def compare_signature(self):
        default_params = inspect.signature(self.__class__).parameters
        return {
            k: getattr(self, k)
            for k, v in default_params.items()
            if getattr(self, k) != v.default
        }

    def set_sessionname(self):
        diff_dict = self.compare_signature()
        if self.sessionname is None:
            kwstr = "_".join(
                [
                    f"{k}{v}"
                    for k, v in diff_dict.items()
                    if k
                    not in ["pipeline", "training", "datasets", "model", "session_dir"]
                ]
            )
            sessionname = f"{self.model.cls_name}_{kwstr}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
            self.sessionname = sessionname

    def set_log_dir(self):
        self.log_dir = str(
            Path(
                self.session_dir,
                self.__class__.__name__,
                self.sessionname,
                self.log_dir_name,
            ).resolve()
        )

    def set_ckpt_dir(self):
        self.ckpt_dir = str(
            Path(
                self.session_dir,
                self.__class__.__name__,
                self.sessionname,
                self.ckpt_dir_name,
            ).resolve()
        )

    def to_json(self, file=None):
        if file is None:
            file = self._handle_filepath("config.json")
        with open(file, "w") as f:
            json.dump(asdict(self), f, indent=4, sort_keys=True)

    def to_toml(self, file=None):
        if file is None:
            file = self._handle_filepath("config.toml")
        with open(file, "w") as f:
            toml.dump(asdict(self), f)

    def _handle_filepath(self, filename):
        log_dir = Path(self.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        return (log_dir.parent / filename).resolve()

    @classmethod
    def from_toml(cls, file):
        file = Path(file).resolve()
        assert file.exists(), f"file missing: {file}"
        data = toml.load(file)
        if data.get("datasets"):
            datasets = []
            for v in data["datasets"]:
                if v.get("pipeline"):
                    v["pipeline"] = ConfigPipeline(**v["pipeline"])
                if v.get("training"):
                    v["training"] = ConfigSplit(**v["training"])
                if v.get("validation"):
                    v["validation"] = ConfigSplit(**v["validation"])
                if v.get("test"):
                    v["test"] = ConfigSplit(**v["test"])
                datasets.append(ConfigDataset(**v))
            data["datasets"] = datasets
        if data.get("model"):
            data["model"] = ConfigModel(**data["model"])
        # pop log_dir and ckpt_dir
        if data.get("log_dir"):
            data.pop("log_dir")
        if data.get("ckpt_dir"):
            data.pop("ckpt_dir")
        return cls(**data)

    def set_attributes_from_toml(self, file):
        file = Path(file).resolve()
        assert file.exists(), f"file missing: {file}"
        data = toml.load(file)
        for k, v in data.items():
            setattr(self, k, v)
        return data
