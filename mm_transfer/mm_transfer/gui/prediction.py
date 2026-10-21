"""Plot window for one cold-hardiness prediction."""

from __future__ import annotations

import tkinter as tk
from typing import TYPE_CHECKING

from mm_transfer.gui.util import Theme, card, display_name, draw_season_plot, scrollable, section_label

if TYPE_CHECKING:
    from mm_transfer.gui.launch_window import LaunchWindow


class PredictionDisplay(tk.Toplevel):
    """Plot predicted LTE50 against the season's temperatures, with the run context beside it."""

    def __init__(self, launch: LaunchWindow, payload: dict) -> None:
        super().__init__(launch)
        self.title("LTE50 Prediction")
        self.configure(bg=Theme.bg)
        self.geometry(launch.geometry())
        self.minsize(1040, 680)
        self.bind("<Escape>", lambda _event: self.destroy())
        browser = launch.master

        body = tk.Frame(self, bg=Theme.bg)
        body.pack(fill="both", expand=True, padx=22, pady=18)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=0, minsize=300)
        body.grid_rowconfigure(0, weight=1)

        plot_card = card(body)
        plot_card.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        plot_card.grid_rowconfigure(0, weight=1)
        plot_card.grid_columnconfigure(0, weight=1)
        self._build_plot(plot_card, payload)

        side_card = card(body)
        side_card.grid(row=0, column=1, sticky="nsew")
        side, bind_wheel = scrollable(side_card)
        section_label(side, "Model", browser.font_section, pady=(16, 6))
        self._fact(side, browser, "Run", payload["model_run"])
        self._fact(side, browser, "Checkpoint", payload["checkpoint"])
        self._fact(side, browser, "Type", payload["model_type"])
        self._fact(side, browser, "Architecture", payload["architecture"])
        self._fact(side, browser, "Mode", payload["mode_label"])
        section_label(side, "Prediction Target", browser.font_section, pady=(16, 6))
        self._fact(side, browser, "Region", display_name(payload["target_region"]))
        self._fact(side, browser, "Station", display_name(payload["target_station"]))
        self._fact(side, browser, "Site", display_name(payload["target_site"]))
        self._fact(side, browser, "Cultivar", display_name(payload["target_cultivar"]))
        if payload.get("history_labels"):
            self._fact(side, browser, "Years", ", ".join(payload["history_labels"]))
        section_label(side, "Weather Source", browser.font_section, pady=(16, 6))
        self._fact(side, browser, "Region", display_name(payload["source_region"]))
        self._fact(side, browser, "Station", display_name(payload["source_station"]))
        self._fact(side, browser, "Year", payload["year"])
        days = payload["days"]
        span = f"{days.min().date().isoformat()} to {days.max().date().isoformat()}"
        self._fact(side, browser, "Season", f"{len(days)} days\n{span}")
        if payload.get("rmse") is not None:
            section_label(side, "Seasonal LTE50 RMSE", browser.font_section, pady=(16, 6))
            self._fact(side, browser, "RMSE", f"{payload['rmse']:.2f} °C")
        tk.Label(
            side,
            text="LTE50 is the predicted lethal temperature at which 50% of dormant buds are damaged by frost.",
            bg=Theme.card,
            fg=Theme.muted,
            font=browser.font_small,
            anchor="w",
            justify="left",
            wraplength=260,
        ).pack(fill="x", padx=16, pady=(8, 16))
        bind_wheel(side)

    def _build_plot(self, parent: tk.Frame, payload: dict) -> None:
        import matplotlib

        matplotlib.use("TkAgg", force=False)
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(8.4, 5.2), dpi=110, facecolor=Theme.card)
        ax = figure.add_subplot(111)
        draw_season_plot(
            ax,
            payload["days"],
            payload["minimum"],
            payload["average"],
            payload["maximum"],
            payload["lte50"],
            measured=payload.get("measured"),
        )
        canvas = FigureCanvasTkAgg(figure, master=parent)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.configure(bg=Theme.card, highlightthickness=0)
        widget.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

    def _fact(self, parent: tk.Frame, browser, caption: str, value: str) -> None:
        row = tk.Frame(parent, bg=Theme.card)
        row.pack(fill="x", padx=16, pady=2)
        tk.Label(
            row,
            text=caption,
            bg=Theme.card,
            fg=Theme.faint,
            font=browser.font_small,
            anchor="nw",
            width=14,
        ).pack(side="left", anchor="n")
        tk.Label(
            row,
            text=value,
            bg=Theme.card,
            fg=Theme.text,
            font=browser.font_ui,
            anchor="w",
            justify="left",
            wraplength=168,
        ).pack(side="left", fill="x", expand=True)
