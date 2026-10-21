"""
utils.py

Declaration of arguments dataclasses and helper
functions for loading configurations

Written by Will Solow, 2025
"""

import os
import pickle
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf
from torch import nn

from mm_transfer.assets import CROP_NAMES, REGIONS, SITES, STATIONS
from mm_transfer.models import DMC


@dataclass
class PConfig:
    """Path to configuration files"""

    config_fpath: str | None = None
    """Model to use"""
    model: str | None = None
    """Model parameters to use"""
    model_parameters: str | None = None
    """Latitude of location"""
    latitude: int | None = None
    """Longitude of location"""
    longitude: int | None = None
    """Start date"""
    start_date: str | None = None
    """Output Variables"""
    output_vars: list | None = None
    """Input variables"""
    input_vars: list | None = None


@dataclass
class DConfig:
    """Model Type"""

    type: str | None = None
    """Model architecture"""
    arch: str | None = None
    """Embedding operation"""
    embed_op: str | None = None
    """Hidden dim of Model"""
    hidden_dim: int | None = None
    """Width of each learned embedding. None keeps the model input width."""
    embed_size: int | None = None
    """How verbose of a string to use. Literal["base", "verb", "bare", "mini", "test"]]"""
    verbose_level: str | None = "verb"
    """Prediction path: direct, text, or adapt."""
    transfer_type: str | None = None
    """Historical seasons sampled when adapting."""
    num_transfer_years: int | None = None
    """Training batch size. Launch uses one season at a time."""
    batch_size: int | None = 1
    """Loss used by classification models."""
    loss_func: str | None = ""


@dataclass
class DataConfig:
    """Path to real data"""

    data_fpath: str | None = None
    """Data type (cold-hardiness, phenology, wheat)"""
    dtype: str | None = None
    """Region of data"""
    region: str | None = None
    """Weather station for data"""
    station: str | None = None
    """Site of data"""
    site: str | None = None
    """Cultivar"""
    cultivar: str | None = None
    """If using synthetic data"""
    synth_data: str | None = None
    """Size of the test set"""
    test_set_size: int | None = 2
    """Tuples held out of training."""
    withold: Any | None = None
    """How many primary cultivars to keep uncapped."""
    prim_cultivars: int | None = None
    """Maximum seasons kept per group."""
    data_cap: int | None = None


@dataclass
class Args:
    """Model configuration"""

    PConfig: object = PConfig
    """Deep Model Args"""
    DConfig: object = DConfig
    """Data Args"""
    DataConfig: object = DataConfig

    """Path to save model"""
    log_path: str | None = None
    """Run name for experiment"""
    run_name: str | None = None
    """Seed"""
    seed: int | None = None
    """Parameters to predict"""
    params: list | None = None
    """Ranges for each parameter to predict"""
    params_range: list | None = None
    """Scale applied when casting predicted parameters."""
    param_scale: Any | None = None


def _load_pickle(handle):
    """Load a dataframe pickle saved by a newer pandas than the one installed here."""
    from pandas._libs.arrays import NDArrayBacked
    from pandas.core.arrays.string_ import StringArray, StringDtype

    original_init = StringDtype.__init__
    original_state = StringArray.__setstate__ if "__setstate__" in StringArray.__dict__ else None

    def _init(self, storage=None, na_value=pd.NA):
        original_init(self, storage)

    def _setstate(self, state):
        if isinstance(state, tuple) and len(state) == 2:
            state = (state[0], state[1], {})
        return NDArrayBacked.__setstate__(self, state)

    StringDtype.__init__ = _init
    StringArray.__setstate__ = _setstate
    try:
        return pickle.load(handle)
    finally:
        StringDtype.__init__ = original_init
        if original_state is None:
            del StringArray.__setstate__
        else:
            StringArray.__setstate__ = original_state


def _repair_config_text(text: str) -> str:
    """Put fields back under their section when the saved file lost its indent."""
    lines = []
    for line in text.splitlines():
        if line.startswith("embed_op:") or line.startswith("test_set_size:"):
            line = f"  {line}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def _normalize_loaded_config(config) -> None:
    """Match saved key names to the structured config."""
    data = config.get("DataConfig")
    if data is not None and "Region" in data:
        if "region" not in data:
            data.region = data.Region
        del data["Region"]
    if "test_set_size" in config and data is not None:
        if data.get("test_set_size") is None:
            data.test_set_size = config.test_set_size
        del config["test_set_size"]


def _load_saved_config(path: str):
    text = _repair_config_text(Path(path).read_text())
    config = OmegaConf.create(yaml.safe_load(text))
    _normalize_loaded_config(config)
    return OmegaConf.merge(Args, config)


def load_config_data(args: Namespace) -> tuple[DictConfig, list[pd.DataFrame]]:
    """
    Load the configuration and data and assumes
    that the configuration file is in _train_configs
    """

    config = _load_saved_config(f"_train_configs/{args.config}.yaml")
    config.seed = int(args.seed)

    if hasattr(args, "cultivar"):
        config.DataConfig.cultivar = args.cultivar if args.cultivar is not None else config.DataConfig.cultivar
    if hasattr(args, "region"):
        config.DataConfig.region = args.region if args.region is not None else config.DataConfig.region
    if hasattr(args, "station"):
        config.DataConfig.station = args.station if args.station is not None else config.DataConfig.station
    if hasattr(args, "site"):
        config.DataConfig.site = args.site if args.site is not None else config.DataConfig.site
    if hasattr(args, "withold"):
        config.DataConfig.withold = (
            OmegaConf.create(yaml.safe_load(args.withold)) if args.withold is not None else config.DataConfig.withold
        )
    if hasattr(args, "transfer_type"):
        config.DConfig.transfer_type = (
            args.transfer_type if args.transfer_type is not None else config.DConfig.transfer_type
        )

    return config, load_data_from_config(config, args)


def load_config_data_fpath(args: Namespace, includes: str = None) -> tuple[DictConfig, list[pd.DataFrame], str]:
    """
    Loads configuration file and data and returns
    filepath to load model
    """

    fpath = f"{os.getcwd()}/{args.config}"

    config = _load_saved_config(f"{fpath}/config.yaml")

    if hasattr(args, "transfer_type"):
        config.DConfig.transfer_type = (
            args.transfer_type if args.transfer_type is not None else config.DConfig.transfer_type
        )
    if hasattr(args, "verbose_level"):
        config.DConfig.verbose_level = (
            args.verbose_level if args.verbose_level is not None else config.DConfig.verbose_level
        )
    return config, load_data_from_config(config, args, includes=includes), fpath


def load_data_from_config(config: DictConfig, args: Namespace, includes: str = None) -> list[pd.DataFrame]:
    """
    Loads data from a OmegaConf configuration file

    Args:
        config: Configuration loaded
        args: Args
        includes: sting that must be included in filename
    """

    def find_pickle_files(root_dir: str, prefix: str = "", contains: str = "", exclude: str = None):
        """
        Recursively find all pickle files in root_dir and subdirectories
        that match a given prefix and contain a specified substring.

        Args:
            root_dir (str): The directory to search in.
            prefix (str): The filename prefix to match (default: "").
            contains (str): A substring that must appear in the filename (default: "").

        Returns:
            List[str]: List of full paths to matching pickle files.
        """
        pickle_paths = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if exclude is not None:
                    if (
                        filename.endswith(".pkl")
                        and filename.startswith(prefix)
                        and contains in filename
                        and exclude not in filename
                    ):
                        full_path = os.path.join(dirpath, filename)
                        pickle_paths.append(full_path)
                else:
                    if filename.endswith(".pkl") and filename.startswith(prefix) and contains in filename:
                        full_path = os.path.join(dirpath, filename)
                        pickle_paths.append(full_path)
        return pickle_paths

    dtype = config.DataConfig.dtype.rsplit("_", 1)[0]
    if hasattr(args, "all"):
        if args.all:
            config.DataConfig.cultivar = "All"
            config.DataConfig.region = "All"
            config.DataConfig.station = "All"
            config.DataConfig.site = "All"
            config.DataConfig.withold = None

    PREFIX = f"{config.DataConfig.data_fpath}{dtype}/"
    if config.DataConfig.synth_data is not None:
        paths = find_pickle_files(f"{PREFIX}{config.DataConfig.synth_data}/", prefix="synth_")
    else:
        if config.DataConfig.region == "All":
            paths = find_pickle_files(
                f"{PREFIX}",
                contains="" if config.DataConfig.cultivar == "All" else config.DataConfig.cultivar,
                exclude="synth",
            )
        else:
            if config.DataConfig.station == "All":
                paths = find_pickle_files(
                    f"{PREFIX}{config.DataConfig.region}/",
                    contains="" if config.DataConfig.cultivar == "All" else config.DataConfig.cultivar,
                    exclude="synth",
                )
            else:
                if config.DataConfig.site == "All":
                    paths = find_pickle_files(
                        f"{PREFIX}{config.DataConfig.region}/{config.DataConfig.station}/",
                        contains="" if config.DataConfig.cultivar == "All" else config.DataConfig.cultivar,
                        exclude="synth",
                    )
                else:
                    paths = find_pickle_files(
                        f"{PREFIX}{config.DataConfig.region}/{config.DataConfig.station}/{config.DataConfig.site}/",
                        contains="" if config.DataConfig.cultivar == "All" else config.DataConfig.cultivar,
                        exclude="synth",
                    )
    data = []
    for p in paths:
        # if includes is not None and includes not in p:
        if "nyus" in p:
            continue
        cultivar = p.split(f"{dtype}_")[-1].replace(".pkl", "").removesuffix("_nyus")
        if cultivar not in CROP_NAMES[config.DataConfig.dtype]:
            continue
        region = p.split("/")[2]
        station = p.split("/")[3]
        site = p.split("/")[4]
        if region not in REGIONS:
            continue
        if station not in STATIONS:
            continue
        if site not in SITES:
            continue
        with open(p, "rb") as f:
            cult_data = _load_pickle(f)
        for cult in cult_data:
            if not isinstance(cult, pd.DataFrame):
                cult = pd.DataFrame(cult)
            cult["CULTIVAR"] = np.where(CROP_NAMES[config.DataConfig.dtype] == cultivar)[0][0]
            if config.DataConfig.synth_data is None:
                cult["REGION"] = np.where(REGIONS == region)[0][0]
                cult["STATION"] = np.where(STATIONS == station)[0][0]
                cult["SITE"] = np.where(SITES == site)[0][0]
            data.append(cult)
    for d in data:
        if "DATE" in d.columns:
            d.rename(columns={"DATE": "DAY"}, inplace=True)

    return data


def load_model_from_config(config: DictConfig, data: list[pd.DataFrame]) -> nn.Module:
    """
    Load the model to train using the configuration and pass
    it args and data
    """
    if config.DConfig.type == "FoundationDeep":
        calibrator = DMC.DeepFoundationRNN(config, data)
    elif config.DConfig.type == "FoundationParam":
        calibrator = DMC.ParamFoundationRNN(config, data)
    else:
        raise NotImplementedError(f"Unrecognized RNN model type `{config.DConfig.type}`")

    return calibrator


def load_dfs(path: str) -> list[pd.DataFrame]:
    """
    Load dataframes from a filepath
    """
    df = pd.read_csv(path)

    df["DAY"] = pd.to_datetime(df["DAY"])

    inds = np.argwhere((df.DAY.dt.month == 1) & (df.DAY.dt.day == 1)).flatten()
    inds = np.concatenate((inds, [-1]))
    df_list = []
    for i in range(1, len(inds)):
        if inds[i] == -1:
            df_list.append(df.loc[inds[i - 1] :])
        else:
            df_list.append(df.loc[inds[i - 1] : inds[i] - 1])
    return df_list


def find_config_yaml_dirs(start_dir="."):
    config_dirs = []
    for root, dirs, files in os.walk(start_dir):
        if "config.yaml" in files:
            relative_path = os.path.relpath(root, start_dir)
            config_dirs.append(relative_path)
    return config_dirs


def load_named_pickles(
    folder_paths: list[str],
    target_name: str,
    include: str = None,
    exclude_multi: bool = False,
):
    """
    Load all pickle files matching a given name in all subdirectories.
    """
    results = {}

    for root in folder_paths:
        root = Path(f"./_runs/{root}")
        for pkl_file in root.rglob(target_name):
            try:
                # Get relative subdirectory name
                subdir = "/".join(pkl_file.parent.parts[-6:])
                if include is not None and include not in subdir:
                    continue
                if "All" in subdir and exclude_multi:
                    continue
                with open(pkl_file, "rb") as f:
                    results[subdir] = pickle.load(f)
            except Exception:
                pass

    return results
