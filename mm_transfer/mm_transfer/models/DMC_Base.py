"""Base class for Param RNN

Written by Will Solow, 2025"""

import torch
from omegaconf import DictConfig
from torch import nn

from mm_transfer.assets import (
    CROP_NAMES,
    CULT_STR,
    CULT_STR_BARE,
    CULT_STR_MINI,
    CULT_STR_TEST,
    CULT_STR_VERB,
    REGION_STR,
    REGION_STR_BARE,
    REGION_STR_MINI,
    REGION_STR_TEST,
    REGION_STR_VERB,
    REGIONS,
    SITE_STR,
    SITE_STR_BARE,
    SITE_STR_MINI,
    SITE_STR_TEST,
    SITE_STR_VERB,
    SITES,
    STATION_STR,
    STATION_STR_BARE,
    STATION_STR_MINI,
    STATION_STR_TEST,
    STATION_STR_VERB,
    STATIONS,
)


def set_embedding_op(model: nn.Module, embed_op: str, embed_size: int | None):
    """
    Set the embedding operation to be used
    in MultiTask Embedding Models
    """
    if embed_op == "concat":

        def concat(embed, input):
            return torch.concatenate((embed, input), dim=-1)

        model.embed_op = concat
        return 2 * model.input_dim if embed_size is None else model.input_dim + embed_size
    elif embed_op == "concat_reg":

        def concat(embed1, embed2, embed3, embed4, input):
            return torch.concatenate((embed1, embed2, embed3, embed4, input), dim=-1)

        model.embed_op = concat
        return 5 * model.input_dim if embed_size is None else model.input_dim + 4 * embed_size
    elif embed_op == "concat_stat":

        def concat(embed1, embed2, embed3, input):
            return torch.concatenate((embed1, embed2, embed3, input), dim=-1)

        model.embed_op = concat
        return 4 * model.input_dim if embed_size is None else model.input_dim + 3 * embed_size
    elif embed_op == "concat_site":

        def concat(embed1, embed2, input):
            return torch.concatenate((embed1, embed2, input), dim=-1)

        model.embed_op = concat
        return 3 * model.input_dim if embed_size is None else model.input_dim + 2 * embed_size
    elif embed_op == "add":

        def add(embed, input):
            return embed + input

        model.embed_op = add
        return model.input_dim
    elif embed_op == "mult":

        def mult(embed, input):
            return embed * input

        model.embed_op = mult
        return model.input_dim


def _phrases(table, mapping, indices: torch.Tensor, override: str | None, count: int):
    """Use an edited description when one was provided, otherwise the asset-table text."""
    if override is not None:
        return [override] * count
    return table[mapping[indices.flatten().to(torch.int)].cpu().numpy()]


class BaseModule(nn.Module):
    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super().__init__()
        self.config = c
        self.hidden_dim = c.DConfig.hidden_dim
        self.dim2 = int(self.hidden_dim / 2)
        self.input_dim = model.get_input_dim(c)
        self.output_dim = model.get_output_dim(c)
        self.embed_dim = set_embedding_op(self, c.DConfig.embed_op, c.DConfig.embed_size)

        self.embed_op_str = c.DConfig.embed_op
        self.synth_data = c.DataConfig.synth_data

        cult_orig = torch.unique(torch.concatenate(list(model.cultivars.values()), axis=0)).to(torch.int)
        self.cult_mapping = torch.zeros((int(cult_orig.max()) + 1,)).to(torch.int).to(model.device)
        self.cult_mapping[cult_orig] = torch.arange(len(cult_orig)).to(torch.int).to(model.device)

        if self.synth_data is None:
            reg_orig = torch.unique(torch.concatenate(list(model.regions.values()), axis=0)).to(torch.int)
            self.reg_mapping = torch.zeros((int(reg_orig.max()) + 1,)).to(torch.int).to(model.device)
            self.reg_mapping[reg_orig] = torch.arange(len(reg_orig)).to(torch.int).to(model.device)

            stat_orig = torch.unique(torch.concatenate(list(model.stations.values()), axis=0)).to(torch.int)
            self.stat_mapping = torch.zeros((int(stat_orig.max()) + 1,)).to(torch.int).to(model.device)
            self.stat_mapping[stat_orig] = torch.arange(len(stat_orig)).to(torch.int).to(model.device)

            site_orig = torch.unique(torch.concatenate(list(model.sites.values()), axis=0)).to(torch.int)
            self.site_mapping = torch.zeros((int(site_orig.max()) + 1,)).to(torch.int).to(model.device)
            self.site_mapping[site_orig] = torch.arange(len(site_orig)).to(torch.int).to(model.device)

    def get_init_state(self, batch_size):
        return self.h0.repeat(1, batch_size, 1)


class FoundationEmbeddingGRU(BaseModule):
    """
    Multi Task GRU
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super().__init__(c, model)

        self.weather_output_dim = 128

        self.cult_embedder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.input_dim if c.DConfig.embed_size is None else c.DConfig.embed_size),
        )
        self.site_embedder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.input_dim if c.DConfig.embed_size is None else c.DConfig.embed_size),
        )

        self.stat_embedder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.input_dim if c.DConfig.embed_size is None else c.DConfig.embed_size),
        )

        self.cult_embedding_layer = nn.Embedding(
            len(CROP_NAMES[c.DataConfig.dtype]),
            self.input_dim if c.DConfig.embed_size is None else c.DConfig.embed_size,
        )
        self.reg_embedding_layer = nn.Embedding(
            len(REGIONS), self.input_dim if c.DConfig.embed_size is None else c.DConfig.embed_size
        )
        self.stat_embedding_layer = nn.Embedding(
            len(STATIONS), self.input_dim if c.DConfig.embed_size is None else c.DConfig.embed_size
        )
        self.site_embedding_layer = nn.Embedding(
            len(SITES), self.input_dim if c.DConfig.embed_size is None else c.DConfig.embed_size
        )

        self.fc1 = nn.Linear(self.embed_dim, self.dim2)
        self.fc2 = nn.Linear(self.dim2, self.hidden_dim)
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.fc3 = nn.Linear(self.hidden_dim, self.dim2)
        self.hidden_to_output = nn.Linear(self.dim2, self.output_dim)

        self.transfer_weather_rnn = nn.GRU(
            self.input_dim if "Deep" in c.DConfig.type else (self.input_dim - len(c.PConfig.output_vars)),
            self.weather_output_dim,
            batch_first=True,
        )
        self.transfer_ch_rnn = nn.GRU(
            self.output_dim if "Deep" in c.DConfig.type else len(c.PConfig.output_vars),
            self.weather_output_dim,
            batch_first=True,
        )
        self.transfer_pool_attn = nn.Linear(self.weather_output_dim, 1)
        self.transfer_embed_attn = nn.Linear(self.weather_output_dim, self.embed_dim - self.input_dim)

        self.h0 = nn.Parameter(torch.zeros(1, self.hidden_dim))

    def get_embeds(
        self,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        text_embedder: nn.Module = None,
        map_idx: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        direct_mask = map_idx == 0
        text_mask = map_idx == 1
        transfer_mask = map_idx == 2

        reg_embed = torch.zeros(
            size=(
                cultivars.shape[0],
                self.input_dim if self.config.DConfig.embed_size is None else self.config.DConfig.embed_size,
            ),
            device=cultivars.device,
        )
        stat_embed = torch.zeros(
            size=(
                cultivars.shape[0],
                self.input_dim if self.config.DConfig.embed_size is None else self.config.DConfig.embed_size,
            ),
            device=cultivars.device,
        )
        site_embed = torch.zeros(
            size=(
                cultivars.shape[0],
                self.input_dim if self.config.DConfig.embed_size is None else self.config.DConfig.embed_size,
            ),
            device=cultivars.device,
        )
        cult_embed = torch.zeros(
            size=(
                cultivars.shape[0],
                self.input_dim if self.config.DConfig.embed_size is None else self.config.DConfig.embed_size,
            ),
            device=cultivars.device,
        )

        if text_mask.sum() != 0:
            reg_embed_text, stat_embed_text, site_embed_text, cult_embed_text = self.forward_embed_text(
                cultivars[text_mask],
                regions[text_mask],
                stations[text_mask],
                sites[text_mask],
                text_embedder=text_embedder,
                **kwargs,
            )
            reg_embed[text_mask.flatten()] = reg_embed_text
            stat_embed[text_mask.flatten()] = stat_embed_text
            site_embed[text_mask.flatten()] = site_embed_text
            cult_embed[text_mask.flatten()] = cult_embed_text

        if direct_mask.sum() != 0:
            reg_embed_direct, stat_embed_direct, site_embed_direct, cult_embed_direct = self.forward_embed_direct(
                cultivars[direct_mask], regions[direct_mask], stations[direct_mask], sites[direct_mask]
            )

            reg_embed[direct_mask.flatten()] = reg_embed_direct
            stat_embed[direct_mask.flatten()] = stat_embed_direct
            site_embed[direct_mask.flatten()] = site_embed_direct
            cult_embed[direct_mask.flatten()] = cult_embed_direct

        return reg_embed, stat_embed, site_embed, cult_embed

    def forward_with_embed(
        self,
        input: torch.Tensor = None,
        hn: torch.Tensor = None,
        reg_embed: torch.Tensor = None,
        stat_embed: torch.Tensor = None,
        site_embed: torch.Tensor = None,
        cult_embed: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if input.ndim == 2:
            input = input.unsqueeze(1)

        cult_embed = cult_embed.unsqueeze(1).repeat(1, input.shape[1], 1)
        site_embed = site_embed.unsqueeze(1).repeat(1, input.shape[1], 1)
        stat_embed = stat_embed.unsqueeze(1).repeat(1, input.shape[1], 1)
        reg_embed = reg_embed.unsqueeze(1).repeat(1, input.shape[1], 1)

        if self.embed_op_str == "concat_reg":
            gru_input = self.embed_op(cult_embed, site_embed, stat_embed, reg_embed, input)
        elif self.embed_op_str == "concat_stat":
            gru_input = self.embed_op(cult_embed, site_embed, stat_embed, input)
        elif self.embed_op_str == "concat_site":
            gru_input = self.embed_op(cult_embed, site_embed, input)
        elif self.embed_op_str == "concat":
            gru_input = self.embed_op(cult_embed, input)

        x = self.fc1(gru_input).relu()
        x = self.fc2(x).relu()
        gru_out, hn = self.rnn(x, hn)

        gru_out = self.fc3(gru_out).relu()
        output = self.hidden_to_output(gru_out)

        return output, hn

    def forward(
        self,
        input: torch.Tensor = None,
        hn: torch.Tensor = None,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        text_embedder: nn.Module = None,
        map_idx: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if input.ndim == 2:
            input = input.unsqueeze(1)

        direct_mask = map_idx == 0
        text_mask = map_idx == 1

        reg_embed = torch.zeros(
            size=(
                input.shape[0],
                self.input_dim if self.config.DConfig.embed_size is None else self.config.DConfig.embed_size,
            ),
            device=input.device,
        )
        stat_embed = torch.zeros(
            size=(
                input.shape[0],
                self.input_dim if self.config.DConfig.embed_size is None else self.config.DConfig.embed_size,
            ),
            device=input.device,
        )
        site_embed = torch.zeros(
            size=(
                input.shape[0],
                self.input_dim if self.config.DConfig.embed_size is None else self.config.DConfig.embed_size,
            ),
            device=input.device,
        )
        cult_embed = torch.zeros(
            size=(
                input.shape[0],
                self.input_dim if self.config.DConfig.embed_size is None else self.config.DConfig.embed_size,
            ),
            device=input.device,
        )

        if text_mask.sum() != 0:
            reg_embed_text, stat_embed_text, site_embed_text, cult_embed_text = self.forward_embed_text(
                cultivars[text_mask],
                regions[text_mask],
                stations[text_mask],
                sites[text_mask],
                text_embedder=text_embedder,
                **kwargs,
            )
            reg_embed[text_mask.flatten()] = reg_embed_text
            stat_embed[text_mask.flatten()] = stat_embed_text
            site_embed[text_mask.flatten()] = site_embed_text
            cult_embed[text_mask.flatten()] = cult_embed_text

        if direct_mask.sum() != 0:
            reg_embed_direct, stat_embed_direct, site_embed_direct, cult_embed_direct = self.forward_embed_direct(
                cultivars[direct_mask], regions[direct_mask], stations[direct_mask], sites[direct_mask]
            )

            reg_embed[direct_mask.flatten()] = reg_embed_direct
            stat_embed[direct_mask.flatten()] = stat_embed_direct
            site_embed[direct_mask.flatten()] = site_embed_direct
            cult_embed[direct_mask.flatten()] = cult_embed_direct

        cult_embed = cult_embed.unsqueeze(1).repeat(1, input.shape[1], 1)
        site_embed = site_embed.unsqueeze(1).repeat(1, input.shape[1], 1)
        stat_embed = stat_embed.unsqueeze(1).repeat(1, input.shape[1], 1)
        reg_embed = reg_embed.unsqueeze(1).repeat(1, input.shape[1], 1)

        if self.embed_op_str == "concat_reg":
            gru_input = self.embed_op(cult_embed, site_embed, stat_embed, reg_embed, input)
        elif self.embed_op_str == "concat_stat":
            gru_input = self.embed_op(cult_embed, site_embed, stat_embed, input)
        elif self.embed_op_str == "concat_site":
            gru_input = self.embed_op(cult_embed, site_embed, input)
        elif self.embed_op_str == "concat":
            gru_input = self.embed_op(cult_embed, input)

        x = self.fc1(gru_input).relu()
        x = self.fc2(x).relu()
        gru_out, hn = self.rnn(x, hn)

        gru_out = self.fc3(gru_out).relu()
        output = self.hidden_to_output(gru_out)

        return output, hn

    def forward_embed_text(
        self,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        text_embedder: nn.Module = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        with torch.no_grad():
            if self.config.DConfig.verbose_level == "verb":
                cult_str = CULT_STR_VERB
                site_str = SITE_STR_VERB
                stat_str = STATION_STR_VERB
                reg_str = REGION_STR_VERB
            elif self.config.DConfig.verbose_level == "mini":
                cult_str = CULT_STR_MINI
                site_str = SITE_STR_MINI
                stat_str = STATION_STR_MINI
                reg_str = REGION_STR_MINI
            elif self.config.DConfig.verbose_level == "bare":
                cult_str = CULT_STR_BARE
                site_str = SITE_STR_BARE
                stat_str = STATION_STR_BARE
                reg_str = REGION_STR_BARE
            elif self.config.DConfig.verbose_level == "base":
                cult_str = CULT_STR
                site_str = SITE_STR
                stat_str = STATION_STR
                reg_str = REGION_STR
            elif self.config.DConfig.verbose_level == "test":
                cult_str = CULT_STR_TEST
                site_str = SITE_STR_TEST
                stat_str = STATION_STR_TEST
                reg_str = REGION_STR_TEST
            cult_embed_text = text_embedder.encode(
                _phrases(cult_str, self.cult_mapping, cultivars, kwargs.get("cultivar_text"), int(cultivars.shape[0])),
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=cultivars.device,
            ).clone()
            site_embed_text = text_embedder.encode(
                _phrases(site_str, self.site_mapping, sites, kwargs.get("site_text"), int(sites.shape[0])),
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=cultivars.device,
            ).clone()
            stat_embed_text = text_embedder.encode(
                _phrases(stat_str, self.stat_mapping, stations, kwargs.get("station_text"), int(stations.shape[0])),
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=cultivars.device,
            ).clone()
            reg_embed_text = text_embedder.encode(
                _phrases(reg_str, self.reg_mapping, regions, kwargs.get("region_text"), int(regions.shape[0])),
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=cultivars.device,
            ).clone()
        cult_embed = self.cult_embedder(site_embed_text)
        site_embed = self.site_embedder(cult_embed_text)
        stat_embed = self.stat_embedder(stat_embed_text)
        reg_embed = self.stat_embedder(reg_embed_text)

        return reg_embed, stat_embed, site_embed, cult_embed

    def forward_embed_direct(
        self,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cult_embed = self.cult_embedding_layer(self.cult_mapping[cultivars.flatten().to(torch.int)])
        reg_embed = (
            self.reg_embedding_layer(self.reg_mapping[regions.flatten().to(torch.int)])
            if self.synth_data is None
            else None
        )
        stat_embed = (
            self.stat_embedding_layer(self.stat_mapping[stations.flatten().to(torch.int)])
            if self.synth_data is None
            else None
        )
        site_embed = (
            self.site_embedding_layer(self.site_mapping[sites.flatten().to(torch.int)])
            if self.synth_data is None
            else None
        )

        return reg_embed, stat_embed, site_embed, cult_embed

    def forward_embed_adapt(
        self,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        calibrator=None,
        **kwargs,
    ):

        transfer_inds = torch.arange(cultivars.shape[0])

        b_all_inds = torch.cat((regions, stations, sites, cultivars), dim=1)[transfer_inds]
        all_inds = calibrator.all_inds["train"]
        adapt_inds = []
        map_idx = []
        for i, ind in enumerate(b_all_inds):
            ids = torch.argwhere((ind == all_inds).all(axis=1)).flatten()
            idx = ids[
                torch.randperm(ids.shape[0])[: min(ids.shape[0] + 1, calibrator.config.DConfig.num_transfer_years + 1)]
            ]
            adapt_inds.append(idx)
            map_idx.append(torch.ones_like(idx, device=idx.device) * i)
        adapt_inds = torch.cat(adapt_inds)
        map_idx = torch.cat(map_idx)
        embed = self.compute_embed(
            calibrator.data["train"][adapt_inds],
            ch=calibrator.val["train"][adapt_inds],
            map_idx=map_idx,
            bsize=(transfer_inds).sum().item(),
        )

    def forward_embed(
        self,
        input: torch.Tensor = None,
        hn: torch.Tensor = None,
        embed: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if input.ndim == 2:
            input = input.unsqueeze(1)

        embed = embed.unsqueeze(1).expand(-1, input.shape[1], -1)

        gru_input = torch.cat((embed, input), dim=-1)

        x = self.fc1(gru_input).relu()
        x = self.fc2(x).relu()
        gru_out, hn = self.rnn(x, hn)

        gru_out = self.fc3(gru_out).relu()
        output = self.hidden_to_output(gru_out)

        return output, hn

    def compute_embed(
        self,
        weather_input: torch.Tensor = None,
        ch: torch.Tensor = None,
        map_idx: torch.Tensor = None,
        bsize: int = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if weather_input.ndim == 2:
            weather_input = weather_input.unsqueeze(1)
        if ch.ndim == 2:
            ch = ch.unsqueeze(1)
        w_encode, _ = self.transfer_weather_rnn(weather_input)  # (T,d)

        B, T, d = ch.shape
        # Create mask for CH
        ch_mask = ~torch.isnan(ch)
        ch_clean = torch.where(ch_mask, ch, torch.zeros_like(ch))

        ch_lengths = ch_mask.sum(dim=(1, 2))
        max_len = ch_lengths.max().item()
        if max_len == 0:
            return torch.zeros(B, self.embed_dim, device=ch.device, requires_grad=True)
        ch_arr = torch.zeros(size=(B, max_len), device=ch.device, requires_grad=True)

        ch_arr_list = []
        # Go through batch and pad data, loop avoids breaking gradient flow via masking
        for i in range(B):
            mask_i = ch_mask[i].any(dim=-1)
            valid_data = ch_clean[i][mask_i]
            valid_len = valid_data.shape[0]
            if valid_len < max_len:
                pad = torch.zeros((max_len - valid_len, d), device=ch.device, dtype=ch.dtype)
                valid_data = torch.cat([valid_data, pad], dim=0)
            ch_arr_list.append(valid_data)
        ch_arr = torch.stack(ch_arr_list)

        ch_embed, _ = self.transfer_ch_rnn(ch_arr)  # (ch_max_len, d), padded with zeros

        attn_mask = torch.arange(max_len, device=ch.device).unsqueeze(0) < ch_lengths.unsqueeze(1)
        ch_encode = torch.where(~attn_mask.unsqueeze(-1), torch.zeros(ch_embed.shape[-1], device=ch.device), ch_embed)

        attn_scores = torch.bmm(w_encode, ch_encode.transpose(1, 2))
        attn_scores = attn_scores.masked_fill(
            ~attn_mask.unsqueeze(1),  # (B, 1, h_max_len)
            float("-inf"),
        )
        attn_weights = torch.softmax(attn_scores, dim=2)  # (B, T, ch_max_len)
        ch_aligned = torch.bmm(attn_weights, ch_encode)  # (B, T, d)

        attn_weights = self.transfer_pool_attn(ch_aligned).softmax(dim=1)  # (B, T, 1)
        attn_pooled = (ch_aligned * attn_weights).sum(dim=1)  # (B, d)
        embed_all = self.transfer_embed_attn(attn_pooled)  # (B,embed_dim)

        embed = torch.zeros((bsize, embed_all.shape[-1]), device=ch.device)
        embed = embed.scatter_reduce(
            0, map_idx.unsqueeze(1).expand(-1, embed_all.shape[-1]), embed_all, reduce="mean", include_self=False
        )
        return embed


class FoundationParamEmbeddingGRU(BaseModule):
    """
    Multi Task GRU
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super().__init__(c, model)

        self.weather_output_dim = 128

        self.cult_embedder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.input_dim if c.DConfig.embed_size is None else c.DConfig.embed_size),
        )
        self.site_embedder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.input_dim if c.DConfig.embed_size is None else c.DConfig.embed_size),
        )

        self.stat_embedder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.input_dim if c.DConfig.embed_size is None else c.DConfig.embed_size),
        )

        self.cult_embedding_layer = nn.Embedding(
            len(CROP_NAMES[c.DataConfig.dtype]),
            self.input_dim if c.DConfig.embed_size is None else c.DConfig.embed_size,
        )
        self.reg_embedding_layer = nn.Embedding(
            len(REGIONS), self.input_dim if c.DConfig.embed_size is None else c.DConfig.embed_size
        )
        self.stat_embedding_layer = nn.Embedding(
            len(STATIONS), self.input_dim if c.DConfig.embed_size is None else c.DConfig.embed_size
        )
        self.site_embedding_layer = nn.Embedding(
            len(SITES), self.input_dim if c.DConfig.embed_size is None else c.DConfig.embed_size
        )

        self.fc1 = nn.Linear(self.embed_dim, self.dim2)
        self.fc2 = nn.Linear(self.dim2, self.hidden_dim)
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.fc3 = nn.Linear(self.hidden_dim, self.dim2)
        self.hidden_to_output = nn.Linear(self.dim2, self.output_dim)

        self.transfer_weather_rnn = nn.GRU(
            self.input_dim if "Deep" in c.DConfig.type else (self.input_dim - len(c.PConfig.output_vars)),
            self.weather_output_dim,
            batch_first=True,
        )
        self.transfer_ch_rnn = nn.GRU(
            self.output_dim if "Deep" in c.DConfig.type else len(c.PConfig.output_vars),
            self.weather_output_dim,
            batch_first=True,
        )
        self.transfer_pool_attn = nn.Linear(self.weather_output_dim, 1)
        self.transfer_embed_attn = nn.Linear(self.weather_output_dim, self.embed_dim - self.input_dim)

        self.h0 = nn.Parameter(torch.zeros(1, self.hidden_dim))

    def get_embeds(
        self,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        text_embedder: nn.Module = None,
        map_idx: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        direct_mask = map_idx == 0
        text_mask = map_idx == 1
        transfer_mask = map_idx == 2

        reg_embed = torch.zeros(
            size=(
                cultivars.shape[0],
                self.input_dim if self.config.DConfig.embed_size is None else self.config.DConfig.embed_size,
            ),
            device=cultivars.device,
        )
        stat_embed = torch.zeros(
            size=(
                cultivars.shape[0],
                self.input_dim if self.config.DConfig.embed_size is None else self.config.DConfig.embed_size,
            ),
            device=cultivars.device,
        )
        site_embed = torch.zeros(
            size=(
                cultivars.shape[0],
                self.input_dim if self.config.DConfig.embed_size is None else self.config.DConfig.embed_size,
            ),
            device=cultivars.device,
        )
        cult_embed = torch.zeros(
            size=(
                cultivars.shape[0],
                self.input_dim if self.config.DConfig.embed_size is None else self.config.DConfig.embed_size,
            ),
            device=cultivars.device,
        )

        if text_mask.sum() != 0:
            reg_embed_text, stat_embed_text, site_embed_text, cult_embed_text = self.forward_embed_text(
                cultivars[text_mask],
                regions[text_mask],
                stations[text_mask],
                sites[text_mask],
                text_embedder=text_embedder,
                **kwargs,
            )
            reg_embed[text_mask.flatten()] = reg_embed_text
            stat_embed[text_mask.flatten()] = stat_embed_text
            site_embed[text_mask.flatten()] = site_embed_text
            cult_embed[text_mask.flatten()] = cult_embed_text

        if direct_mask.sum() != 0:
            reg_embed_direct, stat_embed_direct, site_embed_direct, cult_embed_direct = self.forward_embed_direct(
                cultivars[direct_mask], regions[direct_mask], stations[direct_mask], sites[direct_mask]
            )

            reg_embed[direct_mask.flatten()] = reg_embed_direct
            stat_embed[direct_mask.flatten()] = stat_embed_direct
            site_embed[direct_mask.flatten()] = site_embed_direct
            cult_embed[direct_mask.flatten()] = cult_embed_direct

        return reg_embed, stat_embed, site_embed, cult_embed

    def forward_with_embed(
        self,
        input: torch.Tensor = None,
        hn: torch.Tensor = None,
        reg_embed: torch.Tensor = None,
        stat_embed: torch.Tensor = None,
        site_embed: torch.Tensor = None,
        cult_embed: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if input.ndim == 2:
            input = input.unsqueeze(1)

        cult_embed = cult_embed.unsqueeze(1).repeat(1, input.shape[1], 1)
        site_embed = site_embed.unsqueeze(1).repeat(1, input.shape[1], 1)
        stat_embed = stat_embed.unsqueeze(1).repeat(1, input.shape[1], 1)
        reg_embed = reg_embed.unsqueeze(1).repeat(1, input.shape[1], 1)

        if self.embed_op_str == "concat_reg":
            gru_input = self.embed_op(cult_embed, site_embed, stat_embed, reg_embed, input)
        elif self.embed_op_str == "concat_stat":
            gru_input = self.embed_op(cult_embed, site_embed, stat_embed, input)
        elif self.embed_op_str == "concat_site":
            gru_input = self.embed_op(cult_embed, site_embed, input)
        elif self.embed_op_str == "concat":
            gru_input = self.embed_op(cult_embed, input)

        x = self.fc1(gru_input)
        x = self.fc2(x)
        gru_out, hn = self.rnn(x, hn)

        gru_out = self.fc3(gru_out.relu()).relu()
        output = self.hidden_to_output(gru_out)

        return output, hn

    def forward(
        self,
        input: torch.Tensor = None,
        hn: torch.Tensor = None,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        text_embedder: nn.Module = None,
        map_idx: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if input.ndim == 2:
            input = input.unsqueeze(1)

        direct_mask = map_idx == 0
        text_mask = map_idx == 1
        transfer_mask = map_idx == 2

        reg_embed = torch.zeros(size=(input.shape[0], self.input_dim), device=input.device)
        stat_embed = torch.zeros(size=(input.shape[0], self.input_dim), device=input.device)
        site_embed = torch.zeros(size=(input.shape[0], self.input_dim), device=input.device)
        cult_embed = torch.zeros(size=(input.shape[0], self.input_dim), device=input.device)

        if text_mask.sum() != 0:
            reg_embed_text, stat_embed_text, site_embed_text, cult_embed_text = self.forward_embed_text(
                cultivars[text_mask],
                regions[text_mask],
                stations[text_mask],
                sites[text_mask],
                text_embedder=text_embedder,
                **kwargs,
            )
            reg_embed[text_mask.flatten()] = reg_embed_text
            stat_embed[text_mask.flatten()] = stat_embed_text
            site_embed[text_mask.flatten()] = site_embed_text
            cult_embed[text_mask.flatten()] = cult_embed_text

        if direct_mask.sum() != 0:
            reg_embed_direct, stat_embed_direct, site_embed_direct, cult_embed_direct = self.forward_embed_direct(
                cultivars[direct_mask], regions[direct_mask], stations[direct_mask], sites[direct_mask]
            )

            reg_embed[direct_mask.flatten()] = reg_embed_direct
            stat_embed[direct_mask.flatten()] = stat_embed_direct
            site_embed[direct_mask.flatten()] = site_embed_direct
            cult_embed[direct_mask.flatten()] = cult_embed_direct

        cult_embed = cult_embed.unsqueeze(1).repeat(1, input.shape[1], 1)
        site_embed = site_embed.unsqueeze(1).repeat(1, input.shape[1], 1)
        stat_embed = stat_embed.unsqueeze(1).repeat(1, input.shape[1], 1)
        reg_embed = reg_embed.unsqueeze(1).repeat(1, input.shape[1], 1)

        if self.embed_op_str == "concat_reg":
            gru_input = self.embed_op(cult_embed, site_embed, stat_embed, reg_embed, input)
        elif self.embed_op_str == "concat_stat":
            gru_input = self.embed_op(cult_embed, site_embed, stat_embed, input)
        elif self.embed_op_str == "concat_site":
            gru_input = self.embed_op(cult_embed, site_embed, input)
        elif self.embed_op_str == "concat":
            gru_input = self.embed_op(cult_embed, input)

        x = self.fc1(gru_input)
        x = self.fc2(x)
        gru_out, hn = self.rnn(x, hn)

        gru_out = self.fc3(gru_out.relu()).relu()
        output = self.hidden_to_output(gru_out)

        return output, hn

    def forward_embed_text(
        self,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        text_embedder: nn.Module = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        with torch.no_grad():
            if self.config.DConfig.verbose_level == "verb":
                cult_str = CULT_STR_VERB
                site_str = SITE_STR_VERB
                stat_str = STATION_STR_VERB
                reg_str = REGION_STR_VERB
            elif self.config.DConfig.verbose_level == "mini":
                cult_str = CULT_STR_MINI
                site_str = SITE_STR_MINI
                stat_str = STATION_STR_MINI
                reg_str = REGION_STR_MINI
            elif self.config.DConfig.verbose_level == "bare":
                cult_str = CULT_STR_BARE
                site_str = SITE_STR_BARE
                stat_str = STATION_STR_BARE
                reg_str = REGION_STR_BARE
            elif self.config.DConfig.verbose_level == "base":
                cult_str = CULT_STR
                site_str = SITE_STR
                stat_str = STATION_STR
                reg_str = REGION_STR
            elif self.config.DConfig.verbose_level == "test":
                cult_str = CULT_STR_TEST
                site_str = SITE_STR_TEST
                stat_str = STATION_STR_TEST
                reg_str = REGION_STR_TEST
            cult_embed_text = text_embedder.encode(
                _phrases(cult_str, self.cult_mapping, cultivars, kwargs.get("cultivar_text"), int(cultivars.shape[0])),
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=cultivars.device,
            ).clone()
            site_embed_text = text_embedder.encode(
                _phrases(site_str, self.site_mapping, sites, kwargs.get("site_text"), int(sites.shape[0])),
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=cultivars.device,
            ).clone()
            stat_embed_text = text_embedder.encode(
                _phrases(stat_str, self.stat_mapping, stations, kwargs.get("station_text"), int(stations.shape[0])),
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=cultivars.device,
            ).clone()
            reg_embed_text = text_embedder.encode(
                _phrases(reg_str, self.reg_mapping, regions, kwargs.get("region_text"), int(regions.shape[0])),
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=cultivars.device,
            ).clone()
        cult_embed = self.cult_embedder(site_embed_text)
        site_embed = self.site_embedder(cult_embed_text)
        stat_embed = self.stat_embedder(stat_embed_text)
        reg_embed = self.stat_embedder(reg_embed_text)

        return reg_embed, stat_embed, site_embed, cult_embed

    def forward_embed_direct(
        self,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cult_embed = self.cult_embedding_layer(self.cult_mapping[cultivars.flatten().to(torch.int)])
        reg_embed = (
            self.reg_embedding_layer(self.reg_mapping[regions.flatten().to(torch.int)])
            if self.synth_data is None
            else None
        )
        stat_embed = (
            self.stat_embedding_layer(self.stat_mapping[stations.flatten().to(torch.int)])
            if self.synth_data is None
            else None
        )
        site_embed = (
            self.site_embedding_layer(self.site_mapping[sites.flatten().to(torch.int)])
            if self.synth_data is None
            else None
        )

        return reg_embed, stat_embed, site_embed, cult_embed

    def forward_embed_adapt(
        self,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        calibrator=None,
        **kwargs,
    ):

        transfer_inds = torch.arange(cultivars.shape[0])

        b_all_inds = torch.cat((regions, stations, sites, cultivars), dim=1)[transfer_inds]
        all_inds = calibrator.all_inds["train"]
        adapt_inds = []
        map_idx = []
        for i, ind in enumerate(b_all_inds):
            ids = torch.argwhere((ind == all_inds).all(axis=1)).flatten()
            idx = ids[
                torch.randperm(ids.shape[0])[: min(ids.shape[0] + 1, calibrator.config.DConfig.num_transfer_years + 1)]
            ]
            adapt_inds.append(idx)
            map_idx.append(torch.ones_like(idx, device=idx.device) * i)
        adapt_inds = torch.cat(adapt_inds)
        map_idx = torch.cat(map_idx)
        embed = self.compute_embed(
            calibrator.data["train"][adapt_inds],
            ch=calibrator.val["train"][adapt_inds],
            map_idx=map_idx,
            bsize=(transfer_inds).sum().item(),
        )

    def forward_embed(
        self,
        input: torch.Tensor = None,
        hn: torch.Tensor = None,
        embed: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if input.ndim == 2:
            input = input.unsqueeze(1)

        embed = embed.unsqueeze(1).expand(-1, input.shape[1], -1)

        gru_input = torch.cat((embed, input), dim=-1)

        x = self.fc1(gru_input)
        x = self.fc2(x)
        gru_out, hn = self.rnn(x, hn)

        gru_out = self.fc3(gru_out.relu()).relu()
        output = self.hidden_to_output(gru_out)

        return output, hn

    def compute_embed(
        self,
        weather_input: torch.Tensor = None,
        ch: torch.Tensor = None,
        map_idx: torch.Tensor = None,
        bsize: int = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if weather_input.ndim == 2:
            weather_input = weather_input.unsqueeze(1)
        if ch.ndim == 2:
            ch = ch.unsqueeze(1)
        w_encode, _ = self.transfer_weather_rnn(weather_input)  # (T,d)

        B, T, d = ch.shape
        # Create mask for CH
        ch_mask = ~torch.isnan(ch)
        ch_clean = torch.where(ch_mask, ch, torch.zeros_like(ch))

        ch_lengths = ch_mask.sum(dim=(1, 2))
        max_len = ch_lengths.max().item()
        if max_len == 0:
            return torch.zeros(B, self.embed_dim, device=ch.device, requires_grad=True)
        ch_arr = torch.zeros(size=(B, max_len), device=ch.device, requires_grad=True)

        ch_arr_list = []
        # Go through batch and pad data, loop avoids breaking gradient flow via masking
        for i in range(B):
            mask_i = ch_mask[i].any(dim=-1)
            valid_data = ch_clean[i][mask_i]
            valid_len = valid_data.shape[0]
            if valid_len < max_len:
                pad = torch.zeros((max_len - valid_len, d), device=ch.device, dtype=ch.dtype)
                valid_data = torch.cat([valid_data, pad], dim=0)
            ch_arr_list.append(valid_data)
        ch_arr = torch.stack(ch_arr_list)

        ch_embed, _ = self.transfer_ch_rnn(ch_arr)  # (ch_max_len, d), padded with zeros

        attn_mask = torch.arange(max_len, device=ch.device).unsqueeze(0) < ch_lengths.unsqueeze(1)
        ch_encode = torch.where(~attn_mask.unsqueeze(-1), torch.zeros(ch_embed.shape[-1], device=ch.device), ch_embed)

        attn_scores = torch.bmm(w_encode, ch_encode.transpose(1, 2))
        attn_scores = attn_scores.masked_fill(
            ~attn_mask.unsqueeze(1),  # (B, 1, h_max_len)
            float("-inf"),
        )
        attn_weights = torch.softmax(attn_scores, dim=2)  # (B, T, ch_max_len)
        ch_aligned = torch.bmm(attn_weights, ch_encode)  # (B, T, d)

        attn_weights = self.transfer_pool_attn(ch_aligned).softmax(dim=1)  # (B, T, 1)
        attn_pooled = (ch_aligned * attn_weights).sum(dim=1)  # (B, d)
        embed_all = self.transfer_embed_attn(attn_pooled)  # (B,embed_dim)

        embed = torch.zeros((bsize, embed_all.shape[-1]), device=ch.device)
        embed = embed.scatter_reduce(
            0, map_idx.unsqueeze(1).expand(-1, embed_all.shape[-1]), embed_all, reduce="mean", include_self=False
        )
        return embed
