"""Shared theme, widgets, and helpers for the model browser GUI."""

from __future__ import annotations

import sys
import tkinter as tk
import tkinter.font as tkfont
from collections.abc import Callable
from pathlib import Path
from tkinter import ttk


class Theme:
    """Dark palette shared by every surface, including the config viewer."""

    bg = "#0e1218"
    sidebar = "#12171f"
    card = "#171d27"
    inset = "#121820"
    hover = "#222b3b"
    selected = "#1d3356"
    border = "#2c3648"
    text = "#e8eef7"
    muted = "#93a0b5"
    faint = "#667386"
    accent = "#9dc0ff"
    accent_strong = "#4d7dff"
    accent_hover = "#6d96ff"
    white = "#ffffff"
    disabled = "#243044"
    disabled_text = "#6d7890"
    key = "#9dc4ff"
    string = "#b7e39a"
    number = "#f0b27a"
    keyword = "#d7b4ff"
    comment = "#667386"
    punct = "#7f8ba0"


def repo_root() -> Path:
    """Repository root, two levels above this package."""
    return Path(__file__).resolve().parents[2]


def resolve_model_folder(raw: str) -> Path:
    """Resolve a model folder from the working directory, then the repository root."""
    requested = Path(raw).expanduser()
    if requested.is_dir():
        return requested.resolve()
    if not requested.is_absolute():
        beside_repo = repo_root() / requested
        if beside_repo.is_dir():
            return beside_repo.resolve()
    return requested.resolve()


def prefer_local_model_engine() -> None:
    """The repository directory named model_engine hides the importable package inside it."""
    package_root = repo_root() / "model_engine"
    if (package_root / "model_engine").is_dir():
        location = str(package_root)
        if location in sys.path:
            sys.path.remove(location)
        sys.path.insert(0, location)


def first_family(families: list[str], candidates: tuple[str, ...]) -> str:
    lookup = {name.lower(): name for name in families}
    for name in candidates:
        match = lookup.get(name.lower())
        if match is not None:
            return match
    return "helvetica"


def _round_rect(canvas: tk.Canvas, x1: int, y1: int, x2: int, y2: int, radius: int, **kwargs) -> int:
    radius = int(min(radius, max(1, (x2 - x1) / 2), max(1, (y2 - y1) / 2)))
    points = [
        x1 + radius,
        y1,
        x2 - radius,
        y1,
        x2,
        y1,
        x2,
        y1 + radius,
        x2,
        y2 - radius,
        x2,
        y2,
        x2 - radius,
        y2,
        x1 + radius,
        y2,
        x1,
        y2,
        x1,
        y2 - radius,
        x1,
        y1 + radius,
        x1,
        y1,
    ]
    return canvas.create_polygon(points, smooth=True, splinesteps=24, **kwargs)


class RoundedButton(tk.Canvas):
    """Flat button with rounded corners drawn to match the surrounding surface."""

    def __init__(
        self,
        parent: tk.Misc,
        text: str,
        command,
        *,
        font: tkfont.Font,
        fill: str,
        fg: str,
        hover: str,
        disabled_fill: str,
        disabled_fg: str,
        bg: str,
        outline: str | None = None,
        radius: int = 8,
        pad_x: int = 18,
        pad_y: int = 10,
    ) -> None:
        width = font.measure(text) + pad_x * 2
        height = font.metrics("linespace") + pad_y * 2
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=bg,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        self.command = command
        self.font = font
        self._text = text
        self.fill = fill
        self.fg = fg
        self.hover_fill = hover
        self.disabled_fill = disabled_fill
        self.disabled_fg = disabled_fg
        self.outline = outline
        self.radius = radius
        self.enabled = True
        self._draw(fill, fg)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        self.configure(cursor="hand2" if enabled else "arrow")
        if enabled:
            self._draw(self.fill, self.fg)
        else:
            self._draw(self.disabled_fill, self.disabled_fg)

    def _draw(self, fill: str, fg: str) -> None:
        self.delete("all")
        width = int(self.cget("width"))
        height = int(self.cget("height"))
        if self.outline:
            _round_rect(self, 1, 1, width - 1, height - 1, self.radius, fill=self.outline, outline=self.outline)
            _round_rect(
                self,
                2,
                2,
                width - 2,
                height - 2,
                max(1, self.radius - 1),
                fill=fill,
                outline=fill,
            )
        else:
            _round_rect(self, 1, 1, width - 1, height - 1, self.radius, fill=fill, outline=fill)
        self.create_text(width / 2, height / 2, text=self._text, fill=fg, font=self.font)

    def _on_enter(self, _event: tk.Event) -> None:
        if self.enabled:
            self._draw(self.hover_fill, self.fg)

    def _on_leave(self, _event: tk.Event) -> None:
        if self.enabled:
            self._draw(self.fill, self.fg)
        else:
            self._draw(self.disabled_fill, self.disabled_fg)

    def _on_click(self, _event: tk.Event) -> None:
        if self.enabled and self.command is not None:
            self.command()


def card(parent: tk.Misc) -> tk.Frame:
    """Bordered surface used by the launch and prediction windows."""
    return tk.Frame(parent, bg=Theme.card, highlightthickness=1, highlightbackground=Theme.border)


def section_label(parent: tk.Misc, text: str, font, *, pady: tuple[int, int] = (16, 8)) -> tk.Label:
    label = tk.Label(parent, text=text, bg=Theme.card, fg=Theme.text, font=font, anchor="w")
    label.pack(fill="x", padx=16, pady=pady)
    return label


def scrollable(parent: tk.Misc, *, bg: str = Theme.card) -> tuple[tk.Frame, Callable[[tk.Misc], None]]:
    """Fill ``parent`` with a vertical canvas and return the inner frame plus a wheel binder.

    The scrollbar stays hidden while the contents fit. ``tk.Text`` children are left alone
    so they can scroll their own contents.
    """
    parent.grid_rowconfigure(0, weight=1)
    parent.grid_columnconfigure(0, weight=1)
    canvas = tk.Canvas(parent, bg=bg, highlightthickness=0, bd=0)
    scroll = ttk.Scrollbar(parent, orient="vertical", style="Card.Vertical.TScrollbar", command=canvas.yview)
    canvas.configure(yscrollcommand=scroll.set)
    canvas.grid(row=0, column=0, sticky="nsew")
    inner = tk.Frame(canvas, bg=bg)
    window = canvas.create_window((0, 0), window=inner, anchor="nw")

    def sync(_event: tk.Event | None = None) -> None:
        canvas.update_idletasks()
        bbox = canvas.bbox("all")
        canvas.configure(scrollregion=bbox or (0, 0, 0, 0))
        height = 0 if bbox is None else bbox[3] - bbox[1]
        if height <= canvas.winfo_height() + 1:
            scroll.grid_remove()
        else:
            scroll.grid(row=0, column=1, sticky="ns")

    def resize(event: tk.Event) -> None:
        canvas.itemconfigure(window, width=event.width)
        sync()

    inner.bind("<Configure>", sync)
    canvas.bind("<Configure>", resize)
    parent.after(50, sync)

    def wheel(event: tk.Event) -> str:
        if scroll.winfo_ismapped():
            number = getattr(event, "num", None)
            delta = getattr(event, "delta", 0)
            step = -3 if number == 4 or delta > 0 else 3
            canvas.yview_scroll(step, "units")
        return "break"

    def bind_wheel(widget: tk.Misc) -> None:
        if isinstance(widget, tk.Text):
            return
        widget.bind("<Button-4>", wheel)
        widget.bind("<Button-5>", wheel)
        widget.bind("<MouseWheel>", wheel)
        for child in widget.winfo_children():
            bind_wheel(child)

    return inner, bind_wheel


def packed_scroll(parent: tk.Misc, *, bg: str = Theme.card, pady: tuple[int, int] = (8, 0)):
    """Pack a scrollable area into ``parent`` and return its inner frame and wheel binder."""
    host = tk.Frame(parent, bg=bg)
    host.pack(fill="both", expand=True, pady=pady)
    return scrollable(host, bg=bg)


def lte50_series(output, output_names: list[str]):
    """Reduce a model forward result to one LTE50 value per day."""
    import numpy as np

    values = output[0] if isinstance(output, (tuple, list)) else output
    series = values.detach().float().cpu().numpy()
    if series.ndim == 3:
        series = series[0]
    if series.ndim == 2:
        index = output_names.index("LTE50") if "LTE50" in output_names else 0
        series = series[:, index]
    return np.atleast_1d(series).reshape(-1)


def temperature_series(frame):
    """Daily minimum, average, and maximum temperature, with the season dates."""
    import pandas as pd

    if "DAY" not in frame.columns:
        raise ValueError("The selected season has no DAY column.")
    missing = [name for name in ("TMIN", "TEMP", "TMAX") if name not in frame.columns]
    if missing:
        raise ValueError("Season is missing temperature columns: " + ", ".join(missing))
    days = pd.to_datetime(frame["DAY"])
    minimum = frame["TMIN"].to_numpy(dtype=float)
    average = frame["TEMP"].to_numpy(dtype=float)
    maximum = frame["TMAX"].to_numpy(dtype=float)
    return days, minimum, average, maximum


def draw_season_plot(ax, days, minimum, average, maximum, lte50) -> None:
    """Plot one season of temperature and predicted cold hardiness, in Celsius."""
    import matplotlib.dates as mdates
    import pandas as pd

    dates = pd.to_datetime(pd.Series(days)).to_numpy(dtype="datetime64[ns]")
    ax.set_facecolor(Theme.inset)
    ax.fill_between(dates, minimum, maximum, color="#4d7dff", alpha=0.16, linewidth=0)
    ax.plot(dates, minimum, color="#7eb6ff", linewidth=1.3, label="Minimum temperature")
    ax.plot(dates, average, color="#f0b27a", linewidth=1.6, label="Average temperature")
    ax.plot(dates, maximum, color="#ff8f7a", linewidth=1.3, label="Maximum temperature")
    ax.plot(dates, lte50, color="#d7b4ff", linewidth=2.2, label="LTE50", zorder=4)
    ax.axhline(0, color=Theme.border, linewidth=0.8, zorder=0)
    ax.set_title("Cold hardiness through the growing season", color=Theme.text, pad=12, fontsize=13)
    ax.set_ylabel("°C", color=Theme.muted)
    ax.set_xlabel("")
    ax.grid(True, axis="y", color=Theme.border, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.margins(x=0.01, y=0.08)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(colors=Theme.muted, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(Theme.border)
    legend = ax.legend(loc="upper left", frameon=True, facecolor=Theme.card, edgecolor=Theme.border)
    for text in legend.get_texts():
        text.set_color(Theme.text)
    ax.figure.autofmt_xdate(rotation=30, ha="right")
    ax.figure.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.18)
