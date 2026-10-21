"""
DMC_Transfer.py

Contains classes for running Param RNN models. We assume that
the model predicts parameters for a physical model.

Written by Will Solow, 2025
"""

import numpy as np
import pandas as pd
import torch
from model_engine.engine import get_engine
from model_engine.util import per_task_param_loader
from omegaconf import DictConfig
from torch import nn

from mm_transfer.assets import CROP_NAMES, REGIONS, SITES, STATIONS
from mm_transfer.models.base import BaseModel
from mm_transfer.models.DMC_Base import FoundationEmbeddingGRU, FoundationParamEmbeddingGRU
from mm_transfer.models.process_data import EPS, date_to_cyclic, make_tensor_inputs


def _name_index(name: str | None, table: np.ndarray, frame: pd.DataFrame, column: str) -> int:
    """Use an explicit target name, or the index already stored on the season."""
    if name is None:
        return int(frame[column].iloc[0])
    matches = np.where(table == name)[0]
    if len(matches) == 0:
        raise ValueError(f"Unknown {column.lower()} `{name}`.")
    return int(matches[0])


def _normalize_like_process_data(model: nn.Module, frame: pd.DataFrame) -> torch.Tensor:
    """Normalize one season with the training statistics from process_data."""
    table = frame.loc[:, list(model.input_vars)]
    raw = table.to_numpy()
    if "DAY" in table.columns:
        if table.columns[0] != "DAY":
            raise ValueError("DAY must be the first input so normalization matches process_data.")
        cyclic = np.asarray(
            [date_to_cyclic(pd.Timestamp(raw[i, 0]).to_pydatetime().date()) for i in range(len(raw))],
            dtype=np.float32,
        )
        values = np.concatenate((cyclic, raw[:, 1:].astype(np.float32)), axis=1)
    else:
        values = raw.astype(np.float32)
    stats = model.drange.detach().cpu().numpy()
    if values.shape[1] != stats.shape[0]:
        raise ValueError(
            f"Selected season has {values.shape[1]} inputs, but the trained normalization has {stats.shape[0]}."
        )
    values = (values - stats[:, 0]) / (stats[:, 1] + EPS)
    return torch.tensor(np.ascontiguousarray(values), dtype=torch.float32, device=model.device)


def _install_selected_weather(model: nn.Module, frame: pd.DataFrame, cultivar_index: int) -> None:
    """Point the physical model at this season, labeled with the target cultivar."""
    engine = getattr(model, "model", None)
    if engine is None or not hasattr(engine, "inputdataprovider"):
        return
    stamped = frame.copy()
    stamped["CULTIVAR"] = cultivar_index
    columns = list(model.input_vars)
    extra = ["CULTIVAR"] if "CULTIVAR" not in columns else []
    if "DAY" not in columns:
        extra.append("DAY")
    model.input_data = make_tensor_inputs(model.config, [stamped.loc[:, columns + extra]])
    engine.inputdataprovider = model.input_data


def _history_embed(model: nn.Module, history: list[tuple[str, int]], batch_size: int) -> torch.Tensor:
    """Condition on the selected stored seasons instead of a random training sample."""
    if not history:
        raise ValueError("Select at least one historical year.")
    weather = torch.stack([model.data[split][index] for split, index in history])
    observed = torch.stack([model.val[split][index] for split, index in history])
    grouping = torch.zeros(len(history), dtype=torch.long, device=model.device)
    return model.nn.compute_embed(weather, ch=observed, map_idx=grouping, bsize=batch_size)


def _text_kwargs(text: dict | None) -> dict[str, str | None]:
    """Descriptions to encode, or nothing when this run uses another embedding."""
    if text is None:
        return {}
    return {
        "region_text": text.get("region"),
        "station_text": text.get("station"),
        "site_text": text.get("site"),
        "cultivar_text": text.get("cultivar"),
    }


def _forward_with_data(
    model: nn.Module,
    data: pd.DataFrame,
    region: str | None,
    station: str | None,
    site: str | None,
    cultivar: str | None,
    text: dict | None = None,
    history: list[tuple[str, int]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frame = data.copy()
    if "DAY" not in frame.columns and "DATE" in frame.columns:
        frame = frame.rename(columns={"DATE": "DAY"})
    weather = _normalize_like_process_data(model, frame)
    dates = pd.to_datetime(frame["DAY"]).to_numpy().astype("datetime64[D]")
    crops = CROP_NAMES[model.config.DataConfig.dtype]
    cultivar_index = _name_index(cultivar, crops, frame, "CULTIVAR")
    region_index = _name_index(region, REGIONS, frame, "REGION")
    station_index = _name_index(station, STATIONS, frame, "STATION")
    site_index = _name_index(site, SITES, frame, "SITE")
    _install_selected_weather(model, frame, cultivar_index)

    def column(index: int) -> torch.Tensor:
        return torch.tensor([[index]], dtype=torch.float32, device=model.device)

    model.eval()
    with torch.no_grad():
        return model.forward(
            data=weather,
            dates=dates,
            cultivars=column(cultivar_index),
            regions=column(region_index),
            stations=column(station_index),
            sites=column(site_index),
            history=history,
            text=text,
        )


class BaseRNN(BaseModel):
    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:

        super().__init__(config, data)

        self.nn = BaseRNN.make_rnn(self, config).to(self.device)
        (
            self.nn.rnn.flatten_parameters()
            if hasattr(self.nn, "rnn") and hasattr(self.nn.rnn, "flatten_parameters")
            else None
        )

    @staticmethod
    def make_rnn(model: nn.Module, config: DictConfig) -> nn.Module:
        """Make the RNN"""

        # Create RNN Model
        if config.DConfig.arch == "FoundationGRU":
            nn = FoundationEmbeddingGRU(config, model)
        elif config.DConfig.arch == "FoundationParamGRU":
            nn = FoundationParamEmbeddingGRU(config, model)
        else:
            raise Exception(f"Unrecognized Model Architecture `{config.DConfig.arch}`")

        return nn

    def forward_with_data(
        self,
        data: pd.DataFrame,
        region: str | None = None,
        station: str | None = None,
        site: str | None = None,
        cultivar: str | None = None,
        text: dict | None = None,
        history: list[tuple[str, int]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one season the same way process_data prepares a training year."""
        return _forward_with_data(self, data, region, station, site, cultivar, text, history)


class DeepFoundationRNN(BaseRNN):
    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:
        super().__init__(config, data)

        from sentence_transformers import SentenceTransformer

        self.text_model = SentenceTransformer("intfloat/e5-large-v2", device=self.device)

    def forward_with_data(
        self,
        data: pd.DataFrame,
        region: str | None = None,
        station: str | None = None,
        site: str | None = None,
        cultivar: str | None = None,
        text: dict | None = None,
        history: list[tuple[str, int]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize one selected season and run it through the deep model."""
        return super().forward_with_data(data, region, station, site, cultivar, text, history)

    def forward(
        self,
        data: torch.Tensor = None,
        dates: np.ndarray = None,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        history: list | None = None,
        text: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Predict with direct embeddings, unless text descriptions or history are provided."""
        data, dates, cultivars, b_size, _dlen = self.handle_data(data, dates, cultivars)
        self.nn.zero_grad()
        hn = self.nn.get_init_state(batch_size=b_size)

        if history is not None:
            embed = _history_embed(self, history, b_size)
            output, _hn = self.nn.forward_embed(data, embed=embed, hn=hn)
            return output

        map_idx = torch.zeros_like(cultivars, device=self.device)
        if text is not None:
            map_idx = torch.ones_like(cultivars, device=self.device)
        output, _hn = self.nn(
            data,
            cultivars=cultivars,
            regions=regions,
            stations=stations,
            sites=sites,
            hn=hn,
            text_embedder=self.text_model,
            map_idx=map_idx,
            **_text_kwargs(text),
        )
        return output


class ParamFoundationRNN(BaseRNN):
    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:
        super().__init__(config, data)

        from sentence_transformers import SentenceTransformer

        self.text_model = SentenceTransformer("intfloat/e5-large-v2", device=self.device)

        self.model = get_engine(self.config)(
            num_models=self.batch_size,
            config=config.PConfig,
            inputprovider=self.input_data,
            device=self.device,
        )

        self.task_params = per_task_param_loader(CROP_NAMES, config, self.params).to(self.device)

    def forward_with_data(
        self,
        data: pd.DataFrame,
        region: str | None = None,
        station: str | None = None,
        site: str | None = None,
        cultivar: str | None = None,
        text: dict | None = None,
        history: list[tuple[str, int]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize one selected season and run it through the parameter model."""
        return super().forward_with_data(data, region, station, site, cultivar, text, history)

    def forward(
        self,
        data: torch.Tensor = None,
        dates: np.ndarray = None,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        history: list | None = None,
        text: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Step the physical model with direct embeddings, text, or selected history."""
        data, dates, cultivars, b_size, dlen = self.handle_data(data, dates, cultivars)
        output_tens, param_tens, _model_output = self.setup_storage(b_size, dlen)

        batch_params = (
            self.task_params[cultivars.to(torch.long).squeeze()] if cultivars is not None else self.task_params[0]
        )
        self.model.set_model_params(batch_params, self.params)
        output = self.model.reset(b_size)

        self.nn.zero_grad()
        hn = self.nn.get_init_state(batch_size=b_size)
        embed = None
        reg_embed = stat_embed = site_embed = cult_embed = None
        if history is not None:
            embed = _history_embed(self, history, b_size)
        else:
            map_idx = torch.zeros_like(cultivars, device=self.device)
            if text is not None:
                map_idx = torch.ones_like(cultivars, device=self.device)
            reg_embed, stat_embed, site_embed, cult_embed = self.nn.get_embeds(
                cultivars,
                regions,
                stations,
                sites,
                self.text_model,
                map_idx,
                **_text_kwargs(text),
            )

        for i in range(dlen):
            step_input = torch.cat((output.view(output.shape[0], -1).detach(), data[:, i]), dim=-1)
            if history is not None:
                step_params, hn = self.nn.forward_embed(step_input, embed=embed, hn=hn)
            else:
                step_params, hn = self.nn.forward_with_embed(
                    step_input,
                    hn=hn,
                    reg_embed=reg_embed,
                    stat_embed=stat_embed,
                    site_embed=site_embed,
                    cult_embed=cult_embed,
                )
            step_params = self.param_cast(step_params.squeeze(), prev_params=batch_params)
            param_tens[:, i] = step_params
            self.model.set_model_params(step_params, self.params)
            output = self.model.run(dates=dates[:, i], cultivars=cultivars)
            output_tens[:, i] = output

        return output_tens
