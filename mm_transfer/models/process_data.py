"""
process_data.py

Contains functions to process phenology and cold hardiness data
for model training

Written by Will Solow, 2025
"""

import datetime
import os
import pickle as pkl

import numpy as np
import pandas as pd
import torch
from model_engine.inputs.input_providers import (
    MultiTensorWeatherDataProvider,
    WeatherDataProvider,
)
from model_engine.util import EPS
from omegaconf import DictConfig, ListConfig
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from mm_transfer.assets import CROP_NAMES, REGIONS, SITES, STATIONS


def withold_checker(model: nn.Module, wd: dict, c: int, si: int, s: int, r: int) -> bool:
    """Check whether or not to include data in training"""
    wd_keys = wd.keys()

    for key in wd_keys:
        assert key in ["Region", "Station", "Site", "Cultivar"], f"Key `{key}` not allowed in keys."
    for val in wd.values():
        assert isinstance(val, ListConfig), f"Value `{val}` is not a ListConfig."

    if "Region" in wd_keys:
        if "Station" in wd_keys:
            if "Site" in wd_keys:
                if "Cultivar" in wd_keys:
                    if (
                        CROP_NAMES[model.config.DataConfig.dtype][c] in wd["Cultivar"]
                        and SITES[si] in wd["Site"]
                        and STATIONS[s] in wd["Station"]
                        and REGIONS[r] in wd["Region"]
                    ):
                        return True
                else:
                    if SITES[si] in wd["Site"] and STATIONS[s] in wd["Station"] and REGIONS[r] in wd["Region"]:
                        return True
            else:
                if "Cultivar" in wd_keys:
                    if (
                        CROP_NAMES[model.config.DataConfig.dtype][c] in wd["Cultivar"]
                        and STATIONS[s] in wd["Station"]
                        and REGIONS[r] in wd["Region"]
                    ):
                        return True
                else:
                    if STATIONS[s] in wd["Station"] and REGIONS[r] in wd["Region"]:
                        return True
        else:
            if "Site" in wd_keys:
                if "Cultivar" in wd_keys:
                    if (
                        CROP_NAMES[model.config.DataConfig.dtype][c] in wd["Cultivar"]
                        and SITES[si] in wd["Site"]
                        and REGIONS[r] in wd["Region"]
                    ):
                        return True
                else:
                    if SITES[si] in wd["Site"] and REGIONS[r] in wd["Region"]:
                        return True
            else:
                if "Cultivar" in wd_keys:
                    if CROP_NAMES[model.config.DataConfig.dtype][c] in wd["Cultivar"] and REGIONS[r] in wd["Region"]:
                        return True
                else:
                    if REGIONS[r] in wd["Region"]:
                        return True
    else:
        if "Station" in wd_keys:
            if "Site" in wd_keys:
                if "Cultivar" in wd_keys:
                    if (
                        CROP_NAMES[model.config.DataConfig.dtype][c] in wd["Cultivar"]
                        and SITES[si] in wd["Site"]
                        and STATIONS[s] in wd["Station"]
                    ):
                        return True
                else:
                    if SITES[si] in wd["Site"] and STATIONS[s] in wd["Station"]:
                        return True
            else:
                if "Cultivar" in wd_keys:
                    if CROP_NAMES[model.config.DataConfig.dtype][c] in wd["Cultivar"] and STATIONS[s] in wd["Station"]:
                        return True
                else:
                    if STATIONS[s] in wd["Station"]:
                        return True
        else:
            if "Site" in wd_keys:
                if "Cultivar" in wd_keys:
                    if CROP_NAMES[model.config.DataConfig.dtype][c] in wd["Cultivar"] and SITES[si] in wd["Site"]:
                        return True
                else:
                    if SITES[si] in wd["Site"]:
                        return True
            else:
                if "Cultivar" in wd_keys:
                    if CROP_NAMES[model.config.DataConfig.dtype][c] in wd["Cultivar"]:
                        return True
                else:
                    return False
    return False


def process_data(model: nn.Module, data: list[pd.DataFrame]) -> None:
    """Process all of the initial data"""

    if len(data) < 2:
        raise Exception(f"Data size: {len(data)}. Insufficient data for building training set.")
    model.output_vars = model.config.PConfig.output_vars
    model.input_vars = model.config.PConfig.input_vars

    model.params = model.config.params
    model.params_range = torch.tensor(np.array(model.config.params_range, dtype=np.float32)).to(model.device)

    # Get normalized (weather) data
    normalized_input_data, model.drange = embed_and_normalize_zscore([d.loc[:, model.input_vars] for d in data])

    normalized_input_data = pad_sequence(normalized_input_data, batch_first=True, padding_value=0).to(model.device)
    model.drange = model.drange.to(torch.float32).to(model.device)

    # Get input data for use with model to avoid unnormalizing
    if "CULTIVAR" in data[0].columns:
        extra_feats = ["CULTIVAR"]
        extra_feats = extra_feats + ["DAY"] if "DAY" not in model.input_vars else extra_feats
        model.input_data = make_tensor_inputs(model.config, [d.loc[:, model.input_vars + extra_feats] for d in data])
    else:
        extra_feats = ["DAY"] if "DAY" not in model.input_vars else []
        model.input_data = make_tensor_inputs(model.config, [d.loc[:, model.input_vars + extra_feats] for d in data])
    # Get validation data
    output_data, output_range = embed_output([d.loc[:, model.output_vars] for d in data])
    output_data = pad_sequence(output_data, batch_first=True, padding_value=model.target_mask).to(model.device)
    model.output_range = output_range.to(torch.float32).to(model.device)

    # Get the dates
    dates = [d.loc[:, "DAY"].to_numpy().astype("datetime64[D]") for d in data]
    max_len = max(len(arr) for arr in dates)
    # Pad each array to the maximum length
    dates = [np.pad(arr, (0, max_len - len(arr)), mode="maximum") for arr in dates]
    model.max_dlen = max_len
    x = model.config.DataConfig.test_set_size  # Number of years for testing set

    # Shuffle to get train and test splits for data
    train_inds = np.empty(shape=(0,))
    test_inds = np.empty(shape=(0,))
    cultivar_data = np.array([d.loc[0, "CULTIVAR"] for d in data]) if "CULTIVAR" in data[0].columns else None
    region_data = np.array([d.loc[0, "REGION"] for d in data]) if "REGION" in data[0].columns else None
    station_data = np.array([d.loc[0, "STATION"] for d in data]) if "STATION" in data[0].columns else None
    site_data = np.array([d.loc[0, "SITE"] for d in data]) if "SITE" in data[0].columns else None

    c_counts = []
    for c in range(len(CROP_NAMES[model.config.DataConfig.dtype])):
        c_counts.append(len(np.argwhere(c == cultivar_data).flatten()))

    c_inds = (
        np.argsort(c_counts)[::-1][: model.config.DataConfig.prim_cultivars :]
        if model.config.DataConfig.prim_cultivars
        else np.array([])
    )
    if model.config.DataConfig.synth_data is None:
        for r in range(len(REGIONS)):
            for s in range(len(STATIONS)):
                for si in range(len(SITES)):
                    for c in range(len(CROP_NAMES[model.config.DataConfig.dtype])):
                        cultivar_inds = np.argwhere(
                            (c == cultivar_data) & (r == region_data) & (s == station_data) & (si == site_data)
                        ).flatten()
                        if len(cultivar_inds) < 2:
                            continue
                        if model.config.DataConfig.withold is not None:
                            if withold_checker(model, model.config.DataConfig.withold, c, si, s, r):
                                print(
                                    f"Witholding: {REGIONS[r]}, {STATIONS[s]}, {SITES[si]}, {CROP_NAMES[model.config.DataConfig.dtype][c]}"
                                )
                                continue

                        np.random.shuffle(cultivar_inds)
                        test_inds = np.concatenate((test_inds, cultivar_inds[:x])).astype(np.int32)

                        if (
                            model.config.DataConfig.data_cap is None
                            or c in c_inds
                            or len(cultivar_inds[x:]) <= model.config.DataConfig.data_cap
                        ):
                            train_inds = np.concatenate((train_inds, cultivar_inds[x:][:])).astype(np.int32)
                        else:
                            train_inds = np.concatenate(
                                (train_inds, cultivar_inds[x:][: model.config.DataConfig.data_cap])
                            ).astype(np.int32)
    else:
        for c in range(len(CROP_NAMES[model.config.DataConfig.dtype])):
            cultivar_inds = np.argwhere(c == cultivar_data).flatten()
            if len(cultivar_inds) < 2:
                continue
            np.random.shuffle(cultivar_inds)
            test_inds = np.concatenate((test_inds, cultivar_inds[:x])).astype(np.int32)

            if (
                model.config.DataConfig.data_cap is None
                or c in c_inds
                or len(cultivar_inds[x:]) <= model.config.DataConfig.data_cap
            ):
                train_inds = np.concatenate((train_inds, cultivar_inds[x:][:])).astype(np.int32)
            else:
                train_inds = np.concatenate((train_inds, cultivar_inds[x:][: model.config.DataConfig.data_cap])).astype(
                    np.int32
                )

    np.random.shuffle(train_inds)
    np.random.shuffle(test_inds)
    model.data = {
        "train": torch.stack([normalized_input_data[i] for i in train_inds]).to(torch.float32),
        "test": (
            torch.stack([normalized_input_data[i] for i in test_inds]).to(torch.float32)
            if len(test_inds) > 0
            else torch.tensor([])
        ),
    }
    model.val = {
        "train": torch.stack([output_data[i] for i in train_inds]).to(torch.float32),
        "test": (
            torch.stack([output_data[i] for i in test_inds]).to(torch.float32)
            if len(test_inds) > 0
            else torch.tensor([])
        ),
    }

    model.dates = {
        "train": np.array([dates[i] for i in train_inds]),
        "test": (np.array([dates[i] for i in test_inds]) if len(test_inds) > 0 else np.array([])),
    }

    cultivar_data = (
        torch.tensor(cultivar_data).to(torch.float32).to(model.device).unsqueeze(1)
        if cultivar_data is not None
        else None
    )
    region_data = (
        torch.tensor(region_data).to(torch.float32).to(model.device).unsqueeze(1) if region_data is not None else None
    )
    station_data = (
        torch.tensor(station_data).to(torch.float32).to(model.device).unsqueeze(1) if station_data is not None else None
    )
    site_data = (
        torch.tensor(site_data).to(torch.float32).to(model.device).unsqueeze(1) if site_data is not None else None
    )

    model.num_cultivars = len(torch.unique(cultivar_data)) if cultivar_data is not None else 1
    model.num_regions = len(torch.unique(region_data)) if region_data is not None else 1
    model.num_stations = len(torch.unique(station_data)) if station_data is not None else 1
    model.num_sites = len(torch.unique(site_data)) if site_data is not None else 1

    model.cultivars = {
        "train": torch.stack([cultivar_data[i] for i in train_inds]).to(torch.float32),
        "test": torch.stack([cultivar_data[i] for i in test_inds]).to(torch.float32),
    }
    model.regions = (
        {
            "train": torch.stack([region_data[i] for i in train_inds]).to(torch.float32),
            "test": torch.stack([region_data[i] for i in test_inds]).to(torch.float32),
        }
        if region_data is not None
        else None
    )
    model.stations = (
        {
            "train": torch.stack([station_data[i] for i in train_inds]).to(torch.float32),
            "test": torch.stack([station_data[i] for i in test_inds]).to(torch.float32),
        }
        if station_data is not None
        else None
    )
    model.sites = (
        {
            "train": torch.stack([site_data[i] for i in train_inds]).to(torch.float32),
            "test": torch.stack([site_data[i] for i in test_inds]).to(torch.float32),
        }
        if site_data is not None
        else None
    )

    model.all_inds = {
        "train": torch.cat(
            (model.regions["train"], model.stations["train"], model.sites["train"], model.cultivars["train"]), dim=1
        ),
        "test": torch.cat(
            (model.regions["test"], model.stations["test"], model.sites["test"], model.cultivars["test"]), dim=1
        ),
    }

    model.all_ids = {
        "train": torch.unique(
            torch.cat(
                (model.regions["train"], model.stations["train"], model.sites["train"], model.cultivars["train"]), dim=1
            ),
            dim=0,
        ),
        "test": torch.unique(
            torch.cat(
                (model.regions["test"], model.stations["test"], model.sites["test"], model.cultivars["test"]), dim=1
            ),
            dim=0,
        ),
    }

    if len(model.data["test"]) < 1:
        raise Exception("Insuffient per-cultivar data to build test set")


def process_error(model: nn.Module, fpath: str) -> None:
    """
    Load and process error data
    """
    try:
        fpath = f"{os.getcwd()}/{fpath}"
        with open(f"{fpath}/error.pkl", "rb") as f:
            err = pkl.load(f)
        f.close()
    except:
        raise Exception(f"Unable to open path: {fpath}/error.pkl")

    model.err_val = {
        "train": torch.tensor(err["train"]).to(model.device),
        "test": torch.tensor(err["test"]).to(model.device),
    }


def make_tensor_inputs(config: DictConfig, dfs: list[pd.DataFrame]) -> WeatherDataProvider:
    """
    Make input providers based on the given data frames
    Converts data frames to tensor table
    """

    wp = MultiTensorWeatherDataProvider(pd.concat(dfs, ignore_index=True))

    return wp


def embed_and_normalize_zscore(
    data: list[pd.DataFrame],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Embed and normalize all data using z-score normalization
    Handle if "DAY" is present in the first entry
    """
    tens = []
    stacked_data = (
        np.vstack([d.to_numpy()[:, 1:] for d in data]).astype(np.float32)
        if "DAY" in data[0].columns
        else np.vstack([d.to_numpy() for d in data]).astype(np.float32)
    )
    data_mean = np.nanmean(stacked_data, axis=0).astype(np.float32)
    data_std = np.std(stacked_data, axis=0).astype(np.float32)

    if "DAY" in data[0].columns:
        data_mean = np.concatenate(([0, 0], data_mean)).astype(np.float32)
        data_std = np.concatenate(([1 / np.sqrt(2), 1 / np.sqrt(2)], data_std)).astype(np.float32)

    for d in data:
        d = d.to_numpy()
        if "DAY" in data[0].columns:
            dt = np.reshape([date_to_cyclic(d[i, 0]) for i in range(len(d[:, 0]))], (-1, 2))
            d = np.concatenate((dt, d[:, 1:]), axis=1).astype(np.float32)
        # Z-score normalization
        d = (d - data_mean) / (data_std + EPS)
        tens.append(torch.tensor(d.astype(np.float32), dtype=torch.float32))

    return tens, torch.tensor(np.stack((data_mean, data_std), axis=-1))


def embed_output(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns output data and mean, std to normalize if needed
    """
    tens = []
    stacked_data = np.vstack([d.to_numpy() for d in data]).astype(np.float32)
    data_mean = np.nanmean(stacked_data, axis=0).astype(np.float32)
    data_std = np.nanstd(stacked_data, axis=0).astype(np.float32)  # This used to be std

    for d in data:
        d = d.to_numpy()
        tens.append(torch.tensor(d, dtype=torch.float32))

    return tens, torch.tensor(np.stack((data_mean, data_std), axis=-1))


def date_to_cyclic(date_str: str | datetime.date) -> list[np.ndarray]:
    """
    Convert datetime to cyclic embedding
    """
    if isinstance(date_str, str):
        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    elif isinstance(date_str, datetime.date):
        date_obj = date_str
    else:
        msg = "Invalid type to convert to date"
        raise Exception(msg)
    day_of_year = date_obj.timetuple().tm_yday
    year_sin = np.sin(2 * np.pi * day_of_year / 365)
    year_cos = np.cos(2 * np.pi * day_of_year / 365)

    return [year_sin, year_cos]


def int_to_cyclic(i: int) -> list[np.ndarray]:
    """
    Convert int to day of year
    """
    year_sin = np.sin(2 * np.pi * i / 365)
    year_cos = np.cos(2 * np.pi * i / 365)

    return [year_sin, year_cos]


def normalize(data: torch.Tensor | np.ndarray, drange: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """
    Normalize data given a range
    """
    return (data - drange[:, 0]) / (drange[:, 1] + EPS)


def unnormalize(data: torch.Tensor | np.ndarray, drange: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """
    Unnormalize data given a range
    """
    return data * (drange[:, 1] + EPS) + drange[:, 0]
