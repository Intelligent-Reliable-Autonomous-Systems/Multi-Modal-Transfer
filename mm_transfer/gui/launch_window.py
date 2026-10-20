"""Launch window: choose a weather season and the target to predict for."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import ttk
from typing import TYPE_CHECKING

from mm_transfer.gui.util import (
    RoundedButton,
    Theme,
    card,
    lte50_series,
    packed_scroll,
    prefer_local_model_engine,
    section_label,
    temperature_series,
)

if TYPE_CHECKING:
    from mm_transfer.gui.browser import ModelBrowser
    from mm_transfer.gui.prediction import PredictionDisplay


PREDICTION_MODES = (
    ("direct", "Predict directly"),
    ("text", "Predict from text"),
    ("adapt", "Predict from historical data"),
)

MODE_NOTES = {
    "direct": "Direct prediction uses the learned embeddings for the region, station, site, and cultivar selected above.",
    "text": "These descriptions start from the verbose text for the selection above. Edits are kept until you choose another mode.",
    "adapt": "Check the seasons to condition on. Every available year starts selected, and the selection is kept until you choose another mode.",
}

TEXT_FIELDS = (
    ("region", "Region"),
    ("station", "Station"),
    ("site", "Site"),
    ("cultivar", "Cultivar"),
)


def verbose_text(kind: str, name: str) -> str:
    """Verbose asset description for one region, station, site, or cultivar."""
    import numpy as np

    from mm_transfer.assets import (
        CULT_STR_VERB,
        REGION_STR_VERB,
        REGIONS,
        SITE_STR_VERB,
        SITES,
        STATION_STR_VERB,
        STATIONS,
    )

    tables = {
        "region": (REGIONS, REGION_STR_VERB),
        "station": (STATIONS, STATION_STR_VERB),
        "site": (SITES, SITE_STR_VERB),
    }
    if kind == "cultivar":
        readable = name.replace("_", " ")
        prefix = f"query: {readable}"
        for text in CULT_STR_VERB:
            body = str(text)
            if body == prefix or body.startswith(prefix + " ") or body.startswith(prefix + ","):
                return body
        return prefix
    names, strings = tables[kind]
    matches = np.where(names == name)[0]
    if len(matches) and int(matches[0]) < len(strings):
        return str(strings[int(matches[0])])
    return f"query: {name.replace('_', ' ')}"


def _season_dates(frame):
    import pandas as pd

    column = "DAY" if "DAY" in frame.columns else "DATE"
    return pd.to_datetime(frame[column]).to_numpy().astype("datetime64[D]")


def _row_matches_dates(row_dates, season_dates) -> bool:
    import numpy as np

    count = len(season_dates)
    row = np.asarray(row_dates, dtype="datetime64[D]")
    if count == 0 or count > len(row):
        return False
    if not np.array_equal(row[:count], season_dates):
        return False
    if count == len(row):
        return True
    return bool(np.all(row[count:] == row[count - 1]))


def match_history_rows(model, region: str, station: str, site: str, cultivar: str, frames_by_year: dict) -> dict:
    """Map each year label to a (split, row) in the model's stored seasons."""
    import numpy as np

    from mm_transfer.assets import CROP_NAMES, REGIONS, SITES, STATIONS

    crops = CROP_NAMES[model.config.DataConfig.dtype]

    def position(table, name: str) -> int:
        matches = np.where(table == name)[0]
        if len(matches) == 0:
            raise ValueError(f"Unknown name `{name}`.")
        return int(matches[0])

    wanted = (
        position(REGIONS, region),
        position(STATIONS, station),
        position(SITES, site),
        position(crops, cultivar),
    )
    located: dict[str, tuple[str, int]] = {}
    used: set[tuple[str, int]] = set()
    for year, frame in frames_by_year.items():
        season_dates = _season_dates(frame)
        for split in ("train", "test"):
            identifiers = model.all_inds[split]
            if identifiers is None or identifiers.numel() == 0:
                continue
            ids = np.rint(identifiers.detach().cpu().numpy())
            candidates = np.flatnonzero(
                (ids[:, 0] == wanted[0])
                & (ids[:, 1] == wanted[1])
                & (ids[:, 2] == wanted[2])
                & (ids[:, 3] == wanted[3])
            )
            for index in candidates:
                key = (split, int(index))
                if key in used:
                    continue
                if _row_matches_dates(model.dates[split][index], season_dates):
                    located[year] = key
                    used.add(key)
                    break
            if year in located:
                break
    return located


def _config_argument(model_dir: Path) -> str:
    """load_config_data_fpath joins this path onto the working directory."""
    try:
        return str(model_dir.resolve().relative_to(Path.cwd()))
    except ValueError as exc:
        raise ValueError(f"The model folder must sit inside the working directory ({Path.cwd()}).") from exc


def index_seasons(frames: list, config) -> dict[str, dict]:
    """Group seasons as region, station, site, cultivar, then year.

    The loaded list repeats those labels on every row and across seasons.
    Each unique place and year is kept once.
    """
    import pandas as pd

    from mm_transfer.assets import CROP_NAMES, REGIONS, SITES, STATIONS

    crops = CROP_NAMES[config.DataConfig.dtype]
    tree: dict[str, dict] = {}
    for frame in frames:
        missing_day = "DAY" not in frame.columns and "DATE" not in frame.columns
        if "REGION" not in frame.columns or missing_day:
            continue
        season = frame if "DAY" in frame.columns else frame.rename(columns={"DATE": "DAY"})
        days = pd.to_datetime(season["DAY"])
        if days.empty:
            continue
        region = str(REGIONS[int(season["REGION"].iloc[0])])
        station = str(STATIONS[int(season["STATION"].iloc[0])])
        site = str(SITES[int(season["SITE"].iloc[0])])
        cultivar = str(crops[int(season["CULTIVAR"].iloc[0])])
        start, end = days.min(), days.max()
        label = str(start.year) if start.year == end.year else f"{start.year}-{end.year}"
        span = f"{start.date().isoformat()} to {end.date().isoformat()}"
        bucket = tree.setdefault(region, {}).setdefault(station, {}).setdefault(site, {}).setdefault(cultivar, {})
        if label in bucket:
            previous = pd.to_datetime(bucket[label]["DAY"])
            previous_span = f"{previous.min().date().isoformat()} to {previous.max().date().isoformat()}"
            if previous_span == span:
                continue
            label = span
        if label in bucket:
            continue
        bucket[label] = season
    return tree


def aggregate_site_years(site: dict) -> dict:
    """Union of seasons at one site across every cultivar.

    When several cultivars share a year label, keep the longer season.
    """
    years: dict = {}
    for cultivar_years in site.values():
        if not isinstance(cultivar_years, dict):
            continue
        for label, frame in cultivar_years.items():
            current = years.get(label)
            if current is None or len(frame) > len(current):
                years[label] = frame
    return years


def _ordered(names: list[str], table) -> list[str]:
    rank = {str(name): index for index, name in enumerate(table)}
    return sorted(names, key=lambda name: rank.get(name, 10**6))


def load_launched_model(model_dir: Path, checkpoint_name: str):
    """Load config, observations, and weights for one saved run."""
    from argparse import Namespace

    prefer_local_model_engine()
    from mm_transfer.utils import load_config_data_fpath, load_model_from_config

    config, data, fpath = load_config_data_fpath(Namespace(config=_config_argument(model_dir)))
    config.DConfig.batch_size = 1
    model = load_model_from_config(config, data)
    model.load_model(fpath, checkpoint_name)
    model.eval()
    model.batch_size = 1
    return model, index_seasons(data, config)


class LaunchWindow(tk.Toplevel):
    """Choose a source season and the location and cultivar to predict for."""

    def __init__(self, master: ModelBrowser, on_back, *, autoload: bool = True) -> None:
        super().__init__(master)
        self.title("Launch Model")
        self.configure(bg=Theme.bg)
        self.geometry(master.geometry())
        self.minsize(1100, 720)
        self.model_dir = master.selected_model_dir
        self.checkpoint = master.selected_checkpoint
        self.on_back = on_back
        self.model = None
        self.seasons: dict = {}
        self.source_frame = None
        self.target_region = ""
        self.target_station = ""
        self.target_site = ""
        self.target_cultivar = ""
        self.prediction_mode = "direct"
        self._text_edits = {kind: {} for kind, _caption in TEXT_FIELDS}
        self._text_names = {kind: "" for kind, _caption in TEXT_FIELDS}
        self._text_boxes: dict = {}
        self._history_edits: dict = {}
        self._history_key = None
        self._history_vars: dict = {}
        self._cancel = threading.Event()
        self._messages: queue.Queue = queue.Queue()
        self._predict_messages: queue.Queue = queue.Queue()
        self._predicting = False
        self._prediction_view: PredictionDisplay | None = None
        self.protocol("WM_DELETE_WINDOW", self._quit)
        self.bind("<Escape>", lambda _event: self._request_close())

        bar = tk.Frame(self, bg=Theme.bg)
        bar.pack(side="bottom", fill="x", padx=24, pady=24)
        RoundedButton(
            bar,
            "Select Different Model",
            self._request_close,
            font=master.font_ui_bold,
            fill=Theme.card,
            fg=Theme.text,
            hover=Theme.hover,
            disabled_fill=Theme.card,
            disabled_fg=Theme.muted,
            bg=Theme.bg,
            outline=Theme.border,
        ).pack(side="left")
        self.predict_button = RoundedButton(
            bar,
            "Make Prediction",
            self._make_prediction,
            font=master.font_ui_bold,
            fill=Theme.accent_strong,
            fg=Theme.white,
            hover=Theme.accent_hover,
            disabled_fill=Theme.disabled,
            disabled_fg=Theme.disabled_text,
            bg=Theme.bg,
        )
        self.predict_button.pack(side="right")
        self.predict_button.set_enabled(False)
        self.predict_status = tk.Label(
            bar,
            text="",
            bg=Theme.bg,
            fg=Theme.muted,
            font=master.font_small,
            anchor="e",
            justify="right",
            wraplength=480,
        )
        self.predict_status.pack(side="right", padx=(0, 16))

        self.body = tk.Frame(self, bg=Theme.bg)
        self.body.pack(fill="both", expand=True, padx=22, pady=(18, 0))
        self.status = tk.Label(
            self.body,
            text="Loading model and data…",
            bg=Theme.bg,
            fg=Theme.muted,
            font=master.font_ui,
            justify="center",
            wraplength=720,
        )
        self.status.place(relx=0.5, rely=0.45, anchor="center")
        if autoload:
            self.after(50, self._start_load)

    def _request_close(self) -> None:
        self._cancel.set()
        self.on_back()

    def _quit(self) -> None:
        """Close button ends the program. The footer button still returns to model selection."""
        self._cancel.set()
        self.protocol("WM_DELETE_WINDOW", lambda: None)
        master = self.master
        if master is not None and master.winfo_exists():
            master._quit()
            return
        self.destroy()

    def _start_load(self) -> None:
        threading.Thread(target=self._load_worker, daemon=True).start()
        self.after(100, self._poll_load)

    def _load_worker(self) -> None:
        try:
            if self._cancel.is_set():
                return
            self._messages.put(("status", "Loading model and data..."))
            model, seasons = load_launched_model(self.model_dir, self.checkpoint.name)
            if self._cancel.is_set():
                return
            self._messages.put(("ready", (model, seasons)))
        except Exception:
            self._messages.put(("error", traceback.format_exc()))

    def _poll_load(self) -> None:
        if not self.winfo_exists() or self._cancel.is_set():
            return
        try:
            kind, payload = self._messages.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_load)
            return
        if kind == "status":
            self.status.configure(text=payload)
            self.after(100, self._poll_load)
            return
        if kind == "error":
            lines = [line.strip() for line in payload.splitlines() if line.strip()]
            while lines and (
                lines[-1].startswith("object_type=")
                or lines[-1].startswith("full_key:")
                or lines[-1].startswith("reference_type=")
            ):
                lines.pop()
            detail = lines[-1] if lines else "Unknown error"
            if ": " in detail and (detail.startswith("omegaconf.") or detail.startswith("yaml.")):
                detail = detail.split(": ", 1)[1]
            self.status.configure(text="The model could not be loaded.\n\n" + detail, fg="#f0b27a")
            return
        model, seasons = payload
        self._show_workspace(model, seasons)

    def _show_workspace(self, model, seasons: dict) -> None:
        self.model = model
        self.seasons = seasons
        self.model.config.DConfig.transfer_type = self.prediction_mode
        self.status.destroy()
        self._style_inputs()
        self.body.grid_columnconfigure(0, weight=2, minsize=250)
        self.body.grid_columnconfigure(1, weight=3, minsize=340)
        self.body.grid_columnconfigure(2, weight=3, minsize=320)
        self.body.grid_rowconfigure(0, weight=1)

        self._build_model_column().grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        self._target_panel = self._build_target_column()
        self._target_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 14))
        self._build_source_column().grid(row=0, column=2, sticky="nsew")
        self._fill_targets()
        self._fill_sources()
        self.after_idle(self._hold_target_size)

    def _hold_target_size(self) -> None:
        """Keep the Predict For column at the size it has for direct prediction."""
        if getattr(self, "_target_held", False) or not self.winfo_exists():
            return
        if self.prediction_mode != "direct":
            self.after(50, self._hold_target_size)
            return
        panel = self._target_panel
        self.update_idletasks()
        if not self.winfo_viewable() or panel.winfo_width() <= 1 or panel.winfo_height() <= 1:
            self.after(50, self._hold_target_size)
            return
        size = (panel.winfo_reqwidth(), panel.winfo_reqheight())
        if getattr(self, "_target_seen", None) != size:
            self._target_seen = size
            self.after(50, self._hold_target_size)
            return
        self._target_held = True
        panel.pack_propagate(False)
        panel.configure(width=size[0], height=size[1])

    def _style_inputs(self) -> None:
        style = ttk.Style(self)
        style.configure(
            "Launch.TCombobox",
            fieldbackground=Theme.inset,
            background=Theme.card,
            foreground=Theme.text,
            arrowcolor=Theme.text,
            bordercolor=Theme.border,
            lightcolor=Theme.border,
            darkcolor=Theme.border,
            padding=6,
            font=self.master.font_ui,
        )
        style.map(
            "Launch.TCombobox",
            fieldbackground=[("readonly", Theme.inset)],
            foreground=[("readonly", Theme.text)],
            selectbackground=[("readonly", Theme.inset)],
            selectforeground=[("readonly", Theme.text)],
        )
        self.option_add("*TCombobox*Listbox.background", Theme.inset)
        self.option_add("*TCombobox*Listbox.foreground", Theme.text)
        self.option_add("*TCombobox*Listbox.selectBackground", Theme.selected)
        self.option_add("*TCombobox*Listbox.selectForeground", Theme.white)
        self.option_add("*TCombobox*Listbox.font", self.master.font_ui)

    def _build_model_column(self) -> tk.Frame:
        panel = card(self.body)
        section_label(panel, "Loaded model", self.master.font_section)
        config = self.model.config
        details = (
            ("Run", self.model_dir.name),
            ("Checkpoint", self.checkpoint.name),
            ("Type", str(config.DConfig.type)),
            ("Architecture", str(config.DConfig.arch)),
            ("Data", str(config.DataConfig.dtype)),
        )
        for caption, value in details:
            tk.Label(panel, text=caption, bg=Theme.card, fg=Theme.faint, font=self.master.font_small, anchor="w").pack(
                fill="x", padx=16
            )
            tk.Label(
                panel,
                text=value,
                bg=Theme.card,
                fg=Theme.text,
                font=self.master.font_ui,
                anchor="w",
                justify="left",
                wraplength=240,
            ).pack(fill="x", padx=16, pady=(0, 10))
        tk.Label(
            panel,
            text="Ready for a source season and a prediction target.",
            bg=Theme.card,
            fg=Theme.muted,
            font=self.master.font_small,
            anchor="w",
            justify="left",
            wraplength=240,
        ).pack(fill="x", padx=16, pady=(4, 16))
        return panel

    def _build_target_column(self) -> tk.Frame:
        panel = card(self.body)
        section_label(panel, "Predict For", self.master.font_section)
        self.target_region_box = self._selector(panel, "Region", self._on_target_region)
        self.target_station_box = self._selector(panel, "Station", self._on_target_station)
        self.target_site_box = self._selector(panel, "Site", self._on_target_site)
        self.target_cultivar_box = self._selector(panel, "Cultivar", self._on_target_cultivar)
        section_label(panel, "Prediction Mode", self.master.font_section, pady=(18, 8))
        self._mode_rows = {}
        for value, label in PREDICTION_MODES:
            self._mode_rows[value] = self._mode_row(panel, value, label)
        self._paint_modes()
        self.mode_panel = tk.Frame(panel, bg=Theme.card)
        self.mode_panel.pack(fill="both", expand=True, padx=16, pady=(12, 16))
        return panel

    def _build_source_column(self) -> tk.Frame:
        panel = card(self.body)
        section_label(panel, "Source data", self.master.font_section)
        tk.Label(
            panel,
            text="Choose the season to run. Years combine every cultivar at the selected region, station, and site.",
            bg=Theme.card,
            fg=Theme.muted,
            font=self.master.font_small,
            anchor="w",
            justify="left",
            wraplength=280,
        ).pack(fill="x", padx=16, pady=(0, 8))
        self.source_region_box = self._selector(panel, "Region", self._on_source_region)
        self.source_station_box = self._selector(panel, "Station", self._on_source_station)
        self.source_site_box = self._selector(panel, "Site", self._on_source_site)
        self.source_year_box = self._selector(panel, "Year", self._on_source_year)
        self.source_summary = tk.Label(
            panel,
            text="",
            bg=Theme.card,
            fg=Theme.text,
            font=self.master.font_ui,
            anchor="w",
            justify="left",
            wraplength=280,
        )
        self.source_summary.pack(fill="x", padx=16, pady=(12, 16))
        return panel

    def _selector(self, parent: tk.Frame, caption: str, on_change) -> ttk.Combobox:
        tk.Label(parent, text=caption, bg=Theme.card, fg=Theme.faint, font=self.master.font_small, anchor="w").pack(
            fill="x", padx=16, pady=(6, 2)
        )
        shell = tk.Frame(parent, bg=Theme.border)
        shell.pack(fill="x", padx=16)
        box = ttk.Combobox(shell, state="readonly", style="Launch.TCombobox", font=self.master.font_ui)
        box.pack(fill="x", padx=1, pady=1, ipady=4)
        box.bind("<<ComboboxSelected>>", lambda _event: on_change())
        return box

    def _mode_row(self, parent: tk.Frame, value: str, label: str) -> dict:
        row = tk.Frame(parent, bg=Theme.inset, highlightthickness=1, highlightbackground=Theme.border, cursor="hand2")
        row.pack(fill="x", padx=16, pady=4)
        name = tk.Label(row, text=label, bg=Theme.inset, fg=Theme.text, font=self.master.font_ui, anchor="w")
        name.pack(side="left", fill="x", expand=True, padx=12, pady=10)
        mark = tk.Label(row, text="", bg=Theme.inset, fg=Theme.accent, font=self.master.font_small)
        mark.pack(side="right", padx=12)
        record = {"row": row, "name": name, "mark": mark}

        def choose(_event=None, mode=value) -> None:
            self._set_mode(mode)

        for widget in (row, name, mark):
            widget.bind("<Button-1>", choose)
        return record

    def _set_mode(self, mode: str) -> None:
        if mode != self.prediction_mode:
            self._reset_mode_memory()
        self.prediction_mode = mode
        if self.model is not None:
            self.model.config.DConfig.transfer_type = mode
        self._paint_modes()
        self._render_mode_panel()

    def _reset_mode_memory(self) -> None:
        self._text_edits = {kind: {} for kind, _caption in TEXT_FIELDS}
        self._text_names = {kind: "" for kind, _caption in TEXT_FIELDS}
        self._text_boxes = {}
        self._history_edits = {}
        self._history_key = None
        self._history_vars = {}

    def _asset_table(self, kind: str):
        from mm_transfer.assets import CROP_NAMES, REGIONS, SITES, STATIONS

        return {
            "region": REGIONS,
            "station": STATIONS,
            "site": SITES,
            "cultivar": CROP_NAMES[self.model.config.DataConfig.dtype],
        }[kind]

    def _asset_names(self, kind: str) -> list[str]:
        return [str(name) for name in self._asset_table(kind)]

    def _capture_mode_state(self) -> None:
        for kind, box in self._text_boxes.items():
            name = self._text_names.get(kind, "")
            if name and box.winfo_exists():
                self._text_edits[kind][name] = box.get("1.0", "end-1c")
        if self._history_key is not None and self._history_vars:
            self._history_edits[self._history_key] = {year for year, var in self._history_vars.items() if var.get()}

    def _store_text(self, kind: str) -> None:
        box = self._text_boxes.get(kind)
        name = self._text_names.get(kind, "")
        if box is not None and name and box.winfo_exists():
            self._text_edits[kind][name] = box.get("1.0", "end-1c")

    def _current_text(self, kind: str) -> str:
        box = self._text_boxes.get(kind)
        if box is not None and box.winfo_exists():
            return box.get("1.0", "end-1c")
        name = getattr(self, f"target_{kind}")
        return self._text_edits[kind].get(name, verbose_text(kind, name))

    def _tuple_years(self) -> dict:
        return (
            self.seasons.get(self.target_region, {})
            .get(self.target_station, {})
            .get(self.target_site, {})
            .get(self.target_cultivar, {})
        )

    def _history_tuple(self) -> tuple[str, str, str, str]:
        return (self.target_region, self.target_station, self.target_site, self.target_cultivar)

    def _render_mode_panel(self) -> None:
        if not hasattr(self, "mode_panel"):
            return
        self._text_boxes = {}
        self._history_vars = {}
        for child in self.mode_panel.winfo_children():
            child.destroy()
        if self.prediction_mode == "text":
            self._render_text_panel()
        elif self.prediction_mode == "adapt":
            self._render_history_panel()
        else:
            self._render_direct_panel()

    def _render_direct_panel(self) -> None:
        note = tk.Frame(self.mode_panel, bg=Theme.inset, highlightthickness=1, highlightbackground=Theme.border)
        note.pack(fill="both", expand=True)
        label = tk.Label(
            note,
            text=MODE_NOTES["direct"],
            bg=Theme.inset,
            fg=Theme.muted,
            font=self.master.font_ui,
            anchor="nw",
            justify="left",
            wraplength=320,
        )
        label.pack(fill="both", expand=True, padx=12, pady=12)
        note.bind("<Configure>", lambda event: label.configure(wraplength=max(160, event.width - 24)))

    def _render_text_panel(self) -> None:
        tk.Label(
            self.mode_panel,
            text=MODE_NOTES["text"],
            bg=Theme.card,
            fg=Theme.muted,
            font=self.master.font_small,
            anchor="w",
            justify="left",
            wraplength=300,
        ).pack(fill="x")
        inner, bind_wheel = packed_scroll(self.mode_panel)
        for kind, caption in TEXT_FIELDS:
            name = getattr(self, f"target_{kind}")
            self._text_names[kind] = name
            tk.Label(inner, text=caption, bg=Theme.card, fg=Theme.faint, font=self.master.font_small, anchor="w").pack(
                fill="x", pady=(8, 2)
            )
            box = tk.Text(
                inner,
                height=4,
                width=1,
                wrap="word",
                bg=Theme.inset,
                fg=Theme.text,
                insertbackground=Theme.accent,
                relief="flat",
                highlightthickness=1,
                highlightbackground=Theme.border,
                highlightcolor=Theme.accent_strong,
                font=self.master.font_small,
                padx=8,
                pady=6,
                undo=True,
            )
            box.pack(fill="x")
            box.insert("1.0", self._text_edits[kind].get(name, verbose_text(kind, name)))
            box.bind("<KeyRelease>", lambda _event, field=kind: self._store_text(field))
            self._text_boxes[kind] = box
        bind_wheel(inner)

    def _render_history_panel(self) -> None:
        import pandas as pd

        tk.Label(
            self.mode_panel,
            text=MODE_NOTES["adapt"],
            bg=Theme.card,
            fg=Theme.muted,
            font=self.master.font_small,
            anchor="w",
            justify="left",
            wraplength=300,
        ).pack(fill="x")
        years = self._tuple_years()
        self._history_key = self._history_tuple()
        if not years:
            tk.Label(
                self.mode_panel,
                text="No historical seasons are available for this region, station, site, and cultivar.",
                bg=Theme.card,
                fg=Theme.muted,
                font=self.master.font_ui,
                anchor="nw",
                justify="left",
                wraplength=300,
            ).pack(fill="both", expand=True, pady=(10, 0))
            return
        saved = self._history_edits.get(self._history_key)
        inner, bind_wheel = packed_scroll(self.mode_panel)
        for year in sorted(years, key=lambda label: (label[:4], label)):
            frame = years[year]
            days = pd.to_datetime(frame["DAY"] if "DAY" in frame.columns else frame["DATE"])
            span = f"{days.min().date().isoformat()} to {days.max().date().isoformat()}"
            checked = year in saved if saved is not None else True
            var = tk.BooleanVar(value=checked)
            self._history_vars[year] = var
            row = tk.Frame(inner, bg=Theme.inset)
            row.pack(fill="x", pady=3)
            tk.Checkbutton(
                row,
                text=f"{year}    {span}",
                variable=var,
                command=self._capture_mode_state,
                bg=Theme.inset,
                fg=Theme.text,
                activebackground=Theme.hover,
                activeforeground=Theme.text,
                selectcolor=Theme.accent_strong,
                font=self.master.font_ui,
                anchor="w",
                highlightthickness=0,
                bd=0,
                wraplength=280,
                justify="left",
            ).pack(fill="x", padx=8, pady=6)
        bind_wheel(inner)

    def _checked_history(self) -> tuple[list[tuple[str, int]], list[str]]:
        self._capture_mode_state()
        years = self._tuple_years()
        if not years:
            raise ValueError("No historical seasons are available for this selection.")
        saved = self._history_edits.get(self._history_tuple())
        chosen = [
            year for year in sorted(years, key=lambda label: (label[:4], label)) if saved is None or year in saved
        ]
        if not chosen:
            raise ValueError("Select at least one historical year.")
        located = match_history_rows(
            self.model,
            self.target_region,
            self.target_station,
            self.target_site,
            self.target_cultivar,
            {year: years[year] for year in chosen},
        )
        missing = [year for year in chosen if year not in located]
        if missing:
            raise ValueError("These years are not in the model's stored seasons: " + ", ".join(missing))
        return [located[year] for year in chosen], chosen

    def _paint_modes(self) -> None:
        for value, record in self._mode_rows.items():
            selected = value == self.prediction_mode
            bg = Theme.selected if selected else Theme.inset
            record["row"].configure(bg=bg, highlightbackground=Theme.accent_strong if selected else Theme.border)
            record["name"].configure(bg=bg, fg=Theme.white if selected else Theme.text)
            record["mark"].configure(bg=bg, text="Selected" if selected else "")

    def _set_options(self, box: ttk.Combobox, options: list[str], preferred: str | None = None) -> str:
        box["values"] = options
        if preferred in options:
            choice = preferred
        else:
            choice = options[0] if options else ""
        box.set(choice)
        return choice

    def _asset_order(self, names: list[str], kind: str) -> list[str]:
        return _ordered(names, self._asset_table(kind))

    def _fill_targets(self) -> None:
        self.target_region = self._set_options(self.target_region_box, self._asset_names("region"), self.target_region)
        self.target_station = self._set_options(
            self.target_station_box, self._asset_names("station"), self.target_station
        )
        self.target_site = self._set_options(self.target_site_box, self._asset_names("site"), self.target_site)
        self.target_cultivar = self._set_options(
            self.target_cultivar_box, self._asset_names("cultivar"), self.target_cultivar
        )
        self._render_mode_panel()

    def _on_target_region(self) -> None:
        self._remember_target("region", self.target_region_box)

    def _on_target_station(self) -> None:
        self._remember_target("station", self.target_station_box)

    def _on_target_site(self) -> None:
        self._remember_target("site", self.target_site_box)

    def _on_target_cultivar(self) -> None:
        self._remember_target("cultivar", self.target_cultivar_box)

    def _remember_target(self, kind: str, box: ttk.Combobox) -> None:
        self._capture_mode_state()
        setattr(self, f"target_{kind}", box.get())
        if self.prediction_mode != "direct":
            self._render_mode_panel()

    def _fill_sources(self) -> None:
        regions = self._asset_order(list(self.seasons), "region")
        self._set_options(self.source_region_box, regions)
        self._on_source_region()

    def _on_source_region(self) -> None:
        stations = self._asset_order(list(self.seasons.get(self.source_region_box.get(), {})), "station")
        self._set_options(self.source_station_box, stations)
        self._on_source_station()

    def _on_source_station(self) -> None:
        region = self.seasons.get(self.source_region_box.get(), {})
        sites = self._asset_order(list(region.get(self.source_station_box.get(), {})), "site")
        self._set_options(self.source_site_box, sites)
        self._on_source_site()

    def _on_source_site(self) -> None:
        years = self._source_years()
        labels = sorted(years, key=lambda label: (label[:4], label))
        self._set_options(self.source_year_box, labels)
        self._apply_source()

    def _on_source_year(self) -> None:
        self._apply_source()

    def _source_years(self) -> dict:
        site = (
            self.seasons.get(self.source_region_box.get(), {})
            .get(self.source_station_box.get(), {})
            .get(self.source_site_box.get(), {})
        )
        return aggregate_site_years(site)

    def _apply_source(self) -> None:
        import pandas as pd

        frame = self._source_years().get(self.source_year_box.get())
        self.source_frame = frame
        if frame is None:
            self.source_summary.configure(text="No season is available for this selection.")
            self.predict_button.set_enabled(False)
            return
        days = pd.to_datetime(frame["DAY"])
        self.source_summary.configure(
            text=(f"{len(frame)} days\n{days.min().date().isoformat()} to {days.max().date().isoformat()}")
        )
        self.predict_button.set_enabled(not self._predicting)

    def _make_prediction(self) -> None:
        if self._predicting or self.model is None or self.source_frame is None:
            return
        if not all((self.target_region, self.target_station, self.target_site, self.target_cultivar)):
            self.predict_status.configure(
                text="Choose a region, station, site, and cultivar to predict for.",
                fg=Theme.number,
            )
            return
        text = None
        history = None
        history_labels = None
        if self.prediction_mode == "text":
            self._capture_mode_state()
            text = {kind: self._current_text(kind) for kind, _caption in TEXT_FIELDS}
        elif self.prediction_mode == "adapt":
            try:
                history, history_labels = self._checked_history()
            except ValueError as exc:
                self.predict_status.configure(text=str(exc), fg=Theme.number)
                return
        request = {
            "frame": self.source_frame.copy(),
            "target_region": self.target_region,
            "target_station": self.target_station,
            "target_site": self.target_site,
            "target_cultivar": self.target_cultivar,
            "source_region": self.source_region_box.get(),
            "source_station": self.source_station_box.get(),
            "source_site": self.source_site_box.get(),
            "year": self.source_year_box.get(),
            "mode": self.prediction_mode,
            "mode_label": dict(PREDICTION_MODES)[self.prediction_mode],
            "model_run": self.model_dir.name,
            "checkpoint": self.checkpoint.name,
            "model_type": str(self.model.config.DConfig.type),
            "architecture": str(self.model.config.DConfig.arch),
            "text": text,
            "history": history,
            "history_labels": history_labels,
        }
        self._predicting = True
        self.predict_button.set_enabled(False)
        self.predict_status.configure(text="Running Model...", fg=Theme.muted)
        threading.Thread(target=self._predict_worker, args=(request,), daemon=True).start()
        self.after(100, self._poll_prediction)

    def _predict_worker(self, request: dict) -> None:
        try:
            if self._cancel.is_set():
                self._predict_messages.put(("cancelled", None))
                return
            self.model.config.DConfig.transfer_type = request["mode"]
            output = self.model.forward_with_data(
                request["frame"],
                region=request["target_region"],
                station=request["target_station"],
                site=request["target_site"],
                cultivar=request["target_cultivar"],
                text=request.get("text"),
                history=request.get("history"),
            )
            names = [str(name) for name in self.model.config.PConfig.output_vars]
            lte50 = lte50_series(output, names)
            days, minimum, average, maximum = temperature_series(request["frame"])
            count = min(len(days), len(lte50), len(minimum), len(average), len(maximum))
            if count == 0:
                raise ValueError("The selected season has no days to plot.")
            request["days"] = days.iloc[:count]
            request["minimum"] = minimum[:count]
            request["average"] = average[:count]
            request["maximum"] = maximum[:count]
            request["lte50"] = lte50[:count]
            self._predict_messages.put(("ok", request))
        except Exception:
            self._predict_messages.put(("error", traceback.format_exc()))

    def _poll_prediction(self) -> None:
        if not self.winfo_exists():
            return
        try:
            kind, payload = self._predict_messages.get_nowait()
        except queue.Empty:
            if self._predicting:
                self.after(100, self._poll_prediction)
            return
        self._predicting = False
        if self.source_frame is not None:
            self.predict_button.set_enabled(True)
        if kind == "cancelled":
            self.predict_status.configure(text="")
            return
        if kind == "error":
            detail = payload.strip().splitlines()[-1]
            self.predict_status.configure(text=detail, fg=Theme.number)
            return
        self.predict_status.configure(text="", fg=Theme.muted)
        self._open_prediction(payload)

    def _open_prediction(self, payload: dict) -> None:
        from mm_transfer.gui.prediction import PredictionDisplay

        existing = self._prediction_view
        if existing is not None:
            try:
                if existing.winfo_exists():
                    existing.destroy()
            except tk.TclError:
                pass
        try:
            self._prediction_view = PredictionDisplay(self, payload)
        except Exception:
            self.predict_status.configure(text=traceback.format_exc().splitlines()[-1], fg=Theme.number)
            return
        self._prediction_view.update_idletasks()
        self._prediction_view.deiconify()
        self._prediction_view.lift()
        self._prediction_view.focus_force()
