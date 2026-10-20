"""Folder browser for trained model runs."""

from __future__ import annotations

import re
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import ttk
from typing import TYPE_CHECKING

from mm_transfer.gui.util import RoundedButton, Theme, first_family

if TYPE_CHECKING:
    from mm_transfer.gui.launch_window import LaunchWindow


_READONLY_NAV = {
    "Left",
    "Right",
    "Up",
    "Down",
    "Home",
    "End",
    "Prior",
    "Next",
    "Shift_L",
    "Shift_R",
    "Control_L",
    "Control_R",
    "Alt_L",
    "Alt_R",
    "Caps_Lock",
    "Num_Lock",
    "Tab",
}


@dataclass
class Node:
    path: Path
    is_model: bool
    children: list[Node] = field(default_factory=list)
    model_count: int = 0

    @property
    def name(self) -> str:
        return self.path.name


_QUOTED = re.compile(r"('([^'\\]|\\.)*'|\"([^\"\\]|\\.)*\")")
_NUMBER = re.compile(r"(?<![\w.])[+-]?(?:\d+\.\d+|\d+)(?:[eE][+-]?\d+)?(?![\w.])")
_KEYWORD = re.compile(r"(?<![\w])(?:null|Null|NULL|true|false|True|False|~)(?![\w])")


def _comment_index(line: str) -> int | None:
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return index
    return None


def _inside(position: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in spans)


class ModelBrowser(tk.Tk):
    def __init__(self, model_root: Path) -> None:
        super().__init__()
        self.model_root = model_root
        self.selected_model_dir: Path | None = None
        self.selected_checkpoint: Path | None = None
        self._launch_win: LaunchWindow | None = None
        self._roots: list[Node] = []
        self._parent: dict[Path, Path | None] = {}
        self._nodes: dict[Path, Node] = {}
        self._expanded: set[Path] = set()
        self._selected_path: Path | None = None
        self._hover_path: Path | None = None
        self._visible: list[Node] = []
        self._rows: dict[Path, dict] = {}
        self._pt_rows: dict[Path, dict] = {}
        self._tree_width = 0
        self._resize_job: str | None = None
        self._rebuilding = False
        self._path_wrap = 0

        self.title("Model Selection")
        self.configure(bg=Theme.bg)
        self.minsize(1040, 700)
        self.tk_setPalette(
            background=Theme.bg,
            foreground=Theme.text,
            activeBackground=Theme.hover,
            activeForeground=Theme.text,
            selectColor=Theme.accent_strong,
            highlightBackground=Theme.border,
            highlightColor=Theme.accent_strong,
        )
        self._init_fonts()
        self._init_styles()
        self._row_height = max(36, self.font_ui.metrics("linespace") + 16)

        if model_root.is_dir():
            self._roots = self._scan(model_root, parent=None)
            self._expanded = {node.path for node in self._roots}
        self._run_count = sum(node.model_count for node in self._roots)

        self._build_layout()
        self._center(1280, 820)
        self.protocol("WM_DELETE_WINDOW", self._quit)
        self.tree_canvas.bind("<Configure>", self._on_tree_configure)
        self.after(80, self.tree_canvas.focus_set)

    def _init_fonts(self) -> None:
        families = list(tkfont.families(self))
        ui = first_family(families, ("texgyreheros", "nimbus sans l", "latin modern sans", "helvetica"))
        mono = first_family(families, ("nimbus mono l", "texgyrecursor", "latin modern typewriter", "courier"))
        self.font_ui = tkfont.Font(self, family=ui, size=12)
        self.font_ui_bold = tkfont.Font(self, family=ui, size=12, weight="bold")
        self.font_title = tkfont.Font(self, family=ui, size=20, weight="bold")
        self.font_section = tkfont.Font(self, family=ui, size=13, weight="bold")
        self.font_small = tkfont.Font(self, family=ui, size=10)
        self.font_mono = tkfont.Font(self, family=mono, size=12)

    def _init_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        for name, trough in (
            ("Card.Vertical.TScrollbar", Theme.card),
            ("Card.Horizontal.TScrollbar", Theme.card),
            ("Side.Vertical.TScrollbar", Theme.sidebar),
        ):
            style.configure(
                name,
                background=Theme.border,
                troughcolor=trough,
                bordercolor=trough,
                lightcolor=trough,
                darkcolor=trough,
                arrowcolor=Theme.muted,
                gripcount=0,
                relief="flat",
                borderwidth=0,
            )
            style.map(name, background=[("active", Theme.muted), ("!active", Theme.border)])
        trough_only = {
            "sticky": "ns",
            "children": [("Vertical.Scrollbar.thumb", {"expand": "1", "sticky": "nswe"})],
        }
        style.layout("Card.Vertical.TScrollbar", [("Vertical.Scrollbar.trough", trough_only)])
        style.layout("Side.Vertical.TScrollbar", [("Vertical.Scrollbar.trough", trough_only)])
        style.layout(
            "Card.Horizontal.TScrollbar",
            [
                (
                    "Horizontal.Scrollbar.trough",
                    {
                        "sticky": "ew",
                        "children": [("Horizontal.Scrollbar.thumb", {"expand": "1", "sticky": "nswe"})],
                    },
                )
            ],
        )

    def _center(self, width: int, height: int) -> None:
        self.update_idletasks()
        x = max(0, (self.winfo_screenwidth() - width) // 2)
        y = max(0, (self.winfo_screenheight() - height) // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _scan(self, path: Path, parent: Path | None) -> list[Node]:
        self._parent[path] = parent
        try:
            entries = sorted(
                (entry for entry in path.iterdir() if entry.is_dir() and not entry.name.startswith(".")),
                key=lambda entry: entry.name.lower(),
            )
        except OSError:
            return []
        nodes: list[Node] = []
        for entry in entries:
            is_model = (entry / "config.yaml").is_file() or any(entry.glob("*.pt"))
            if is_model:
                self._parent[entry] = path
                node = Node(entry, True, [], 1)
            else:
                children = self._scan(entry, parent=path)
                node = Node(entry, False, children, sum(child.model_count for child in children))
            self._nodes[entry] = node
            nodes.append(node)
        return nodes

    def _build_layout(self) -> None:
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        header = tk.Frame(self, bg=Theme.bg)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        titles = tk.Frame(header, bg=Theme.bg)
        titles.grid(row=0, column=0, sticky="ew", padx=22, pady=(18, 14))
        tk.Label(
            titles,
            text="Trained Models",
            bg=Theme.bg,
            fg=Theme.text,
            font=self.font_title,
            anchor="w",
        ).pack(anchor="w")
        subtitle = str(self.model_root) if self.model_root.is_dir() else f"Folder not found: {self.model_root}"
        tk.Label(
            titles,
            text=subtitle,
            bg=Theme.bg,
            fg=Theme.muted,
            font=self.font_small,
            anchor="w",
            justify="left",
            wraplength=980,
        ).pack(anchor="w", pady=(3, 0))
        tk.Frame(header, bg=Theme.border, height=1).grid(row=1, column=0, sticky="ew")

        body = tk.Frame(self, bg=Theme.bg)
        body.grid(row=1, column=0, sticky="nsew")
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, minsize=400, weight=0)
        body.grid_columnconfigure(2, weight=1)

        sidebar = tk.Frame(body, bg=Theme.sidebar)
        sidebar.grid(row=0, column=0, sticky="nsew")
        tk.Frame(body, bg=Theme.border, width=1).grid(row=0, column=1, sticky="ns")
        content = tk.Frame(body, bg=Theme.bg)
        content.grid(row=0, column=2, sticky="nsew")

        side_head = tk.Frame(sidebar, bg=Theme.sidebar)
        side_head.pack(fill="x", padx=16, pady=(16, 8))
        tk.Label(
            side_head,
            text="Models",
            bg=Theme.sidebar,
            fg=Theme.text,
            font=self.font_section,
        ).pack(side="left")
        count_text = f"{self._run_count} run" if self._run_count == 1 else f"{self._run_count} runs"
        tk.Label(
            side_head,
            text=count_text,
            bg=Theme.sidebar,
            fg=Theme.faint,
            font=self.font_small,
        ).pack(side="right")

        tree_wrap = tk.Frame(sidebar, bg=Theme.sidebar)
        tree_wrap.pack(fill="both", expand=True, padx=(8, 0), pady=(0, 8))
        self.tree_canvas = tk.Canvas(
            tree_wrap,
            bg=Theme.sidebar,
            highlightthickness=0,
            bd=0,
            takefocus=True,
        )
        side_scroll = ttk.Scrollbar(
            tree_wrap,
            orient="vertical",
            style="Side.Vertical.TScrollbar",
            command=self.tree_canvas.yview,
        )
        self.tree_canvas.configure(yscrollcommand=side_scroll.set)
        side_scroll.pack(side="right", fill="y")
        self.tree_canvas.pack(side="left", fill="both", expand=True)
        self.tree_inner = tk.Frame(self.tree_canvas, bg=Theme.sidebar)
        self._tree_window = self.tree_canvas.create_window((0, 0), window=self.tree_inner, anchor="nw")
        self.tree_canvas.bind("<Button-1>", lambda _event: self.tree_canvas.focus_set())
        self.tree_canvas.bind("<Up>", lambda _event: self._move_selection(-1))
        self.tree_canvas.bind("<Down>", lambda _event: self._move_selection(1))
        self.tree_canvas.bind("<Left>", lambda _event: self._collapse_or_ascend())
        self.tree_canvas.bind("<Right>", lambda _event: self._expand_selected())
        self.tree_canvas.bind("<Button-4>", self._scroll_tree)
        self.tree_canvas.bind("<Button-5>", self._scroll_tree)

        inner = tk.Frame(content, bg=Theme.bg)
        inner.pack(fill="both", expand=True, padx=22, pady=18)
        inner.grid_columnconfigure(0, weight=1)
        inner.grid_rowconfigure(2, weight=1)
        inner.bind("<Configure>", self._on_content_configure)

        self.detail_title = tk.Label(
            inner,
            text="Select a Model",
            bg=Theme.bg,
            fg=Theme.text,
            font=self.font_title,
            anchor="w",
        )
        self.detail_title.grid(row=0, column=0, sticky="ew")
        self.detail_path = tk.Label(
            inner,
            text="Open a folder in the library, then choose a run to view its config.",
            bg=Theme.bg,
            fg=Theme.muted,
            font=self.font_small,
            anchor="w",
            justify="left",
        )
        self.detail_path.grid(row=1, column=0, sticky="ew", pady=(4, 14))

        config_card = tk.Frame(inner, bg=Theme.card, highlightthickness=1, highlightbackground=Theme.border)
        config_card.grid(row=2, column=0, sticky="nsew")
        config_card.grid_columnconfigure(0, weight=1)
        config_card.grid_rowconfigure(1, weight=1)
        config_head = tk.Frame(config_card, bg=Theme.card)
        config_head.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 6))
        self.config_caption = tk.Label(
            config_head,
            text="Config",
            bg=Theme.card,
            fg=Theme.text,
            font=self.font_section,
        )
        self.config_caption.pack(side="left")
        self.readonly_badge = tk.Label(
            config_head,
            text="",
            bg=Theme.card,
            fg=Theme.faint,
            font=self.font_small,
        )
        self.readonly_badge.pack(side="right")

        viewer = tk.Frame(config_card, bg=Theme.card)
        viewer.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        viewer.grid_rowconfigure(0, weight=1)
        viewer.grid_columnconfigure(0, weight=1)
        self.config_text = tk.Text(
            viewer,
            bg=Theme.card,
            fg=Theme.text,
            insertbackground=Theme.accent,
            insertwidth=0,
            selectbackground=Theme.selected,
            selectforeground=Theme.white,
            relief="flat",
            bd=0,
            highlightthickness=0,
            wrap="word",
            font=self.font_ui,
            padx=8,
            pady=6,
            cursor="xterm",
            undo=False,
        )
        config_y = ttk.Scrollbar(
            viewer, orient="vertical", style="Card.Vertical.TScrollbar", command=self.config_text.yview
        )
        self.config_x = ttk.Scrollbar(
            viewer, orient="horizontal", style="Card.Horizontal.TScrollbar", command=self.config_text.xview
        )
        self.config_text.configure(
            yscrollcommand=config_y.set,
            xscrollcommand=self._on_config_xscroll,
            tabs=self.font_mono.measure("  "),
        )
        self.config_text.grid(row=0, column=0, sticky="nsew")
        config_y.grid(row=0, column=1, sticky="ns")
        for tag, color in (
            ("key", Theme.key),
            ("string", Theme.string),
            ("number", Theme.number),
            ("keyword", Theme.keyword),
            ("comment", Theme.comment),
            ("punct", Theme.punct),
            ("muted", Theme.muted),
        ):
            self.config_text.tag_configure(tag, foreground=color)
        self.config_text.tag_raise("string")
        self.config_text.tag_raise("keyword")
        self.config_text.tag_raise("number")
        self.config_text.tag_raise("comment")
        self.config_text.bind("<Key>", self._reject_config_edit)
        for sequence in ("<Button-2>", "<ButtonRelease-2>", "<<Paste>>", "<<Cut>>"):
            self.config_text.bind(sequence, lambda _event: "break")
        self.config_text.bind("<Button-4>", lambda _event: self.config_text.yview_scroll(-3, "units") or "break")
        self.config_text.bind("<Button-5>", lambda _event: self.config_text.yview_scroll(3, "units") or "break")
        self._set_config_text(
            "Config will appear here when you select a model run.",
            highlight=False,
        )

        pt_card = tk.Frame(inner, bg=Theme.card, highlightthickness=1, highlightbackground=Theme.border)
        pt_card.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        pt_head = tk.Frame(pt_card, bg=Theme.card)
        pt_head.pack(fill="x", padx=14, pady=(12, 6))
        tk.Label(pt_head, text="Checkpoint", bg=Theme.card, fg=Theme.text, font=self.font_section).pack(side="left")
        self.pt_caption = tk.Label(pt_head, text="", bg=Theme.card, fg=Theme.faint, font=self.font_small)
        self.pt_caption.pack(side="right")
        self.pt_box = tk.Frame(pt_card, bg=Theme.card)
        self.pt_box.pack(fill="x", padx=10, pady=(0, 12))
        self._show_checkpoint_placeholder("Checkpoint files will appear here.")

        footer = tk.Frame(self, bg=Theme.bg)
        footer.grid(row=2, column=0, sticky="ew")
        tk.Frame(footer, bg=Theme.border, height=1).pack(fill="x")
        footer_inner = tk.Frame(footer, bg=Theme.bg)
        footer_inner.pack(fill="x", padx=22, pady=16)
        footer_inner.grid_columnconfigure(0, weight=3, minsize=460)
        footer_inner.grid_columnconfigure(1, weight=2, minsize=220)
        self.folder_box = self._labeled_box(footer_inner, 0, "Model folder")
        self.checkpoint_box = self._labeled_box(footer_inner, 1, "Checkpoint")
        self.launch_button = RoundedButton(
            footer_inner,
            "Launch Model",
            self._open_launch,
            font=self.font_ui_bold,
            fill=Theme.accent_strong,
            fg=Theme.white,
            hover=Theme.accent_hover,
            disabled_fill=Theme.disabled,
            disabled_fg=Theme.disabled_text,
            bg=Theme.bg,
        )
        self.launch_button.grid(row=1, column=2, sticky="e", padx=(16, 0))
        self.launch_button.set_enabled(False)
        self._rebuild_tree()

    def _labeled_box(self, parent: tk.Frame, column: int, caption: str) -> tk.Text:
        tk.Label(
            parent,
            text=caption,
            bg=Theme.bg,
            fg=Theme.muted,
            font=self.font_small,
            anchor="w",
        ).grid(row=0, column=column, sticky="ew", padx=(0, 12))
        shell = tk.Frame(parent, bg=Theme.border)
        shell.grid(row=1, column=column, sticky="ew", padx=(0, 12))
        inner = tk.Frame(shell, bg=Theme.inset)
        inner.pack(fill="both", expand=True, padx=1, pady=1)
        box = tk.Text(
            inner,
            height=3,
            width=12,
            wrap="word",
            bg=Theme.inset,
            fg=Theme.text,
            font=self.font_ui,
            relief="flat",
            bd=0,
            highlightthickness=0,
            padx=10,
            pady=0,
            insertwidth=0,
            cursor="arrow",
            undo=False,
        )
        box.pack(fill="x", pady=8)
        box.bind("<Key>", self._reject_value_edit)
        for sequence in ("<Button-2>", "<ButtonRelease-2>", "<<Paste>>", "<<Cut>>"):
            box.bind(sequence, lambda _event: "break")
        return box

    def _set_value_box(self, box: tk.Text, value: str) -> None:
        box.delete("1.0", "end")
        if value:
            box.insert("1.0", value)
        box.see("1.0")

    def _reject_value_edit(self, event: tk.Event) -> str | None:
        if event.keysym in _READONLY_NAV:
            return None
        if (event.state & 0x4) and event.keysym.lower() in {"c", "a", "insert"}:
            return None
        return "break"

    def _on_config_xscroll(self, first: str, last: str) -> None:
        self.config_x.set(first, last)
        fits = float(last) - float(first) >= 0.999
        mapped = bool(self.config_x.winfo_ismapped())
        if fits and mapped:
            self.config_x.grid_remove()
        elif not fits and not mapped:
            self.config_x.grid(row=1, column=0, sticky="ew")

    def _on_content_configure(self, event: tk.Event) -> None:
        wrap = max(240, int(event.width) - 4)
        if wrap != self._path_wrap:
            self._path_wrap = wrap
            self.detail_title.configure(wraplength=wrap)
            self.detail_path.configure(wraplength=wrap)

    def _on_tree_configure(self, event: tk.Event) -> None:
        self.tree_canvas.itemconfigure(self._tree_window, width=event.width)
        if event.width < 50 or abs(event.width - self._tree_width) <= 2:
            return
        self._tree_width = event.width
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(30, self._rebuild_tree)

    def _scroll_tree(self, event: tk.Event) -> str:
        self.tree_canvas.yview_scroll(-3 if event.num == 4 else 3, "units")
        return "break"

    def _format_dir(self, path: Path) -> str:
        try:
            relative = path.relative_to(self.model_root)
        except ValueError:
            return str(path)
        if relative == Path("."):
            return self.model_root.name
        return str(Path(self.model_root.name) / relative)

    def _fit_name(self, text: str, max_px: int, font: tkfont.Font) -> str:
        if max_px <= 12 or font.measure(text) <= max_px:
            return text
        ellipsis = "..."
        best = ellipsis
        low = 0
        high = len(text)
        while low <= high:
            keep = (low + high) // 2
            candidate = text if keep >= len(text) else ellipsis + text[-keep:]
            if font.measure(candidate) <= max_px:
                best = candidate
                low = keep + 1
            else:
                high = keep - 1
        return best

    def _rebuild_tree(self) -> None:
        if self._rebuilding:
            return
        self._rebuilding = True
        self._resize_job = None
        try:
            try:
                y_fraction = self.tree_canvas.yview()[0]
            except tk.TclError:
                y_fraction = 0.0
            for child in self.tree_inner.winfo_children():
                child.destroy()
            self._rows.clear()
            self._visible.clear()
            if self._roots:
                self._render_nodes(self._roots, depth=0)
            else:
                tk.Label(
                    self.tree_inner,
                    text="No model runs found.",
                    bg=Theme.sidebar,
                    fg=Theme.muted,
                    font=self.font_ui,
                    anchor="w",
                    justify="left",
                    wraplength=320,
                ).pack(fill="x", padx=12, pady=12)
            self.tree_inner.update_idletasks()
            bbox = self.tree_canvas.bbox("all")
            self.tree_canvas.configure(scrollregion=bbox or (0, 0, 0, 0))
            self.tree_canvas.yview_moveto(y_fraction)
        finally:
            self._rebuilding = False

    def _render_nodes(self, nodes: list[Node], depth: int) -> None:
        for node in nodes:
            self._render_row(node, depth)
            if node.children and node.path in self._expanded:
                self._render_nodes(node.children, depth + 1)

    def _row_colors(self, node: Node) -> tuple[str, str, str, str]:
        selected = node.path == self._selected_path
        hover = node.path == self._hover_path and not selected
        if selected:
            return Theme.selected, Theme.white, Theme.white, Theme.accent_strong
        marker = Theme.accent if node.is_model else Theme.muted
        foreground = Theme.accent if node.is_model else Theme.text
        if hover:
            return Theme.hover, foreground, marker, Theme.hover
        return Theme.sidebar, foreground, marker, Theme.sidebar

    def _render_row(self, node: Node, depth: int) -> None:
        bg, fg, marker, bar_color = self._row_colors(node)
        row = tk.Frame(self.tree_inner, bg=bg, height=self._row_height, cursor="hand2")
        row.pack(fill="x")
        row.pack_propagate(False)
        bar = tk.Frame(row, bg=bar_color, width=3)
        bar.pack(side="left", fill="y")
        spacer = tk.Frame(row, bg=bg, width=10 + depth * 14)
        spacer.pack(side="left", fill="y")
        spacer.pack_propagate(False)
        marker_canvas = tk.Canvas(
            row,
            width=16,
            height=self._row_height,
            bg=bg,
            highlightthickness=0,
            bd=0,
        )
        marker_canvas.pack(side="left")
        self._draw_marker(marker_canvas, node, marker)
        available = max(80, self.tree_canvas.winfo_width() - (10 + depth * 14) - 72)
        font = self.font_ui_bold if node.is_model else self.font_ui
        label = tk.Label(
            row,
            text=self._fit_name(node.name, available, font),
            bg=bg,
            fg=fg,
            font=font,
            anchor="w",
        )
        label.pack(side="left", fill="x", expand=True, padx=(4, 8))
        count = None
        if not node.is_model:
            count = tk.Label(
                row,
                text=str(node.model_count),
                bg=bg,
                fg=Theme.white if node.path == self._selected_path else Theme.faint,
                font=self.font_small,
            )
            count.pack(side="right", padx=(0, 12))
        parts = [row, bar, spacer, marker_canvas, label]
        if count is not None:
            parts.append(count)
        self._rows[node.path] = {
            "parts": parts,
            "label": label,
            "count": count,
            "marker": marker_canvas,
            "bar": bar,
            "node": node,
        }
        self._visible.append(node)
        for widget in parts:
            widget.bind("<Button-1>", lambda _event, item=node: self._on_node_clicked(item))
            widget.bind("<Enter>", lambda _event, item=node: self._on_row_enter(item))
            widget.bind("<Leave>", lambda event, item=node: self._on_row_leave(item, event))
            widget.bind("<Button-4>", self._scroll_tree)
            widget.bind("<Button-5>", self._scroll_tree)
            widget.bind("<Double-Button-1>", lambda _event: "break")

    def _draw_marker(self, canvas: tk.Canvas, node: Node, color: str) -> None:
        canvas.delete("all")
        mid = self._row_height / 2
        if node.children:
            if node.path in self._expanded:
                canvas.create_polygon(3, mid - 3, 13, mid - 3, 8, mid + 3, fill=color, outline=color)
            else:
                canvas.create_polygon(5, mid - 5, 5, mid + 5, 12, mid, fill=color, outline=color)
        else:
            canvas.create_oval(5, mid - 3, 11, mid + 3, fill=color, outline=color)

    def _repaint_row(self, path: Path) -> None:
        record = self._rows.get(path)
        if record is None:
            return
        node: Node = record["node"]
        bg, fg, marker, bar_color = self._row_colors(node)
        for widget in record["parts"]:
            widget.configure(bg=bg)
        record["label"].configure(fg=fg)
        record["bar"].configure(bg=bar_color)
        if record["count"] is not None:
            record["count"].configure(fg=Theme.white if path == self._selected_path else Theme.faint)
        self._draw_marker(record["marker"], node, marker)

    def _on_row_enter(self, node: Node) -> None:
        if self._hover_path == node.path:
            return
        previous = self._hover_path
        self._hover_path = node.path
        if previous is not None:
            self._repaint_row(previous)
        self._repaint_row(node.path)

    def _on_row_leave(self, node: Node, event: tk.Event) -> None:
        record = self._rows.get(node.path)
        if record is None:
            return
        row = record["parts"][0]
        if row.winfo_rootx() <= event.x_root < row.winfo_rootx() + row.winfo_width() and (
            row.winfo_rooty() <= event.y_root < row.winfo_rooty() + row.winfo_height()
        ):
            return
        if self._hover_path == node.path:
            self._hover_path = None
            self._repaint_row(node.path)

    def _on_node_clicked(self, node: Node) -> None:
        self.tree_canvas.focus_set()
        self._activate(node, toggle=bool(node.children))

    def _show_directory(self, node: Node) -> None:
        runs = f"{node.model_count} model run" if node.model_count == 1 else f"{node.model_count} model runs"
        self.detail_title.configure(text=node.name)
        self.detail_path.configure(text=self._format_dir(node.path))
        self.config_caption.configure(text="Folder")
        self.readonly_badge.configure(text="")
        self._set_config_text(
            f"{runs} are nested in this folder.\n\nSelect a highlighted run to view config.yaml and choose a checkpoint.",
            highlight=False,
        )
        self.pt_caption.configure(text="")
        self._show_checkpoint_placeholder("Select a model run to list checkpoint files.")

    def _show_model(self, node: Node) -> None:
        self.selected_model_dir = node.path
        self._set_value_box(self.folder_box, self._format_dir(node.path))
        self.detail_title.configure(text=node.name)
        self.detail_path.configure(text=self._format_dir(node.path.parent))
        config_path = node.path / "config.yaml"
        self.config_caption.configure(text="config.yaml" if config_path.is_file() else "Config")
        self.readonly_badge.configure(text="Read only" if config_path.is_file() else "")
        if config_path.is_file():
            content = config_path.read_text(encoding="utf-8", errors="replace")
            self._set_config_text(
                content if content.strip() else "(config.yaml is empty)", highlight=bool(content.strip())
            )
        else:
            self._set_config_text("No config.yaml in this folder.", highlight=False)
        checkpoints = sorted(
            node.path.glob("*.pt"), key=lambda path: ("best" not in path.name.lower(), path.name.lower())
        )
        previous = self.selected_checkpoint.name if self.selected_checkpoint is not None else None
        match = next((path for path in checkpoints if path.name == previous), None)
        if match is not None:
            self.selected_checkpoint = match
            self._set_value_box(self.checkpoint_box, match.name)
        else:
            self.selected_checkpoint = None
            self._set_value_box(self.checkpoint_box, "")
        self._render_checkpoints(checkpoints)
        self._refresh_launch()

    def _set_config_text(self, content: str, *, highlight: bool) -> None:
        text = self.config_text
        text.configure(font=self.font_mono if highlight else self.font_ui, wrap="none" if highlight else "word")
        text.delete("1.0", "end")
        for tag in ("key", "string", "number", "keyword", "comment", "punct", "muted"):
            text.tag_remove(tag, "1.0", "end")
        text.insert("1.0", content)
        if highlight:
            for line_no, line in enumerate(content.splitlines(), start=1):
                self._highlight_line(line_no, line)
        else:
            text.tag_add("muted", "1.0", "end")
        text.see("1.0")
        text.update_idletasks()
        self._on_config_xscroll(*text.xview())

    def _highlight_line(self, line_no: int, line: str) -> None:
        comment_at = _comment_index(line)
        code = line if comment_at is None else line[:comment_at]
        if comment_at is not None:
            self._tag(line_no, comment_at, len(line), "comment")
        index = 0
        while index < len(code) and code[index] == " ":
            index += 1
        index = self._mark_list_items(line_no, code, index)
        key_match = re.match(r"([^\s:#][^:#]*?)(\s*)(:)", code[index:])
        if key_match:
            key_end = index + len(key_match.group(1))
            self._tag(line_no, index, key_end, "key")
            colon = key_end + len(key_match.group(2))
            self._tag(line_no, colon, colon + 1, "punct")
            index = self._mark_list_items(line_no, code, colon + 1)
        rest = code[index:]
        quoted = [(match.start(), match.end()) for match in _QUOTED.finditer(rest)]
        for match in _QUOTED.finditer(rest):
            self._tag(line_no, index + match.start(), index + match.end(), "string")
        for match in _NUMBER.finditer(rest):
            if _inside(match.start(), quoted):
                continue
            self._tag(line_no, index + match.start(), index + match.end(), "number")
        for match in _KEYWORD.finditer(rest):
            if _inside(match.start(), quoted):
                continue
            self._tag(line_no, index + match.start(), index + match.end(), "keyword")

    def _mark_list_items(self, line_no: int, code: str, index: int) -> int:
        length = len(code)
        while index < length and code[index] in " \t":
            index += 1
        while index < length and code[index] == "-" and (index + 1 == length or code[index + 1] in " \t"):
            self._tag(line_no, index, index + 1, "punct")
            index += 1
            while index < length and code[index] in " \t":
                index += 1
        return index

    def _tag(self, line_no: int, start: int, end: int, tag: str) -> None:
        if end > start:
            self.config_text.tag_add(tag, f"{line_no}.{start}", f"{line_no}.{end}")

    def _reject_config_edit(self, event: tk.Event) -> str | None:
        if event.keysym in _READONLY_NAV:
            return None
        if (event.state & 0x4) and event.keysym.lower() in {
            "c",
            "insert",
            "left",
            "right",
            "up",
            "down",
            "home",
            "end",
        }:
            return None
        if (event.state & 0x4) and event.keysym.lower() == "a":
            self.config_text.tag_add("sel", "1.0", "end-1c")
            return "break"
        return "break"

    def _show_checkpoint_placeholder(self, message: str) -> None:
        for child in self.pt_box.winfo_children():
            child.destroy()
        self._pt_rows.clear()
        tk.Label(
            self.pt_box,
            text=message,
            bg=Theme.card,
            fg=Theme.muted,
            font=self.font_ui,
            anchor="w",
            justify="left",
        ).pack(fill="x", padx=6, pady=8)

    def _render_checkpoints(self, files: list[Path]) -> None:
        for child in self.pt_box.winfo_children():
            child.destroy()
        self._pt_rows.clear()
        if not files:
            self.pt_caption.configure(text="")
            self._show_checkpoint_placeholder("No .pt checkpoints in this folder.")
            return
        noun = "file" if len(files) == 1 else "files"
        self.pt_caption.configure(text=f"{len(files)} {noun}")
        for path in files:
            self._add_checkpoint_row(path)

    def _add_checkpoint_row(self, path: Path) -> None:
        selected = self.selected_checkpoint == path
        bg = Theme.selected if selected else Theme.inset
        border = Theme.accent_strong if selected else Theme.border
        row = tk.Frame(self.pt_box, bg=bg, highlightthickness=1, highlightbackground=border, cursor="hand2")
        row.pack(fill="x", pady=4)
        name = tk.Label(
            row, text=path.name, bg=bg, fg=Theme.white if selected else Theme.text, font=self.font_ui, anchor="w"
        )
        name.pack(side="left", fill="x", expand=True, padx=12, pady=10)
        mark = tk.Label(
            row,
            text="Selected" if selected else "",
            bg=bg,
            fg=Theme.accent,
            font=self.font_small,
        )
        mark.pack(side="right", padx=12)
        self._pt_rows[path] = {"row": row, "name": name, "mark": mark}
        for widget in (row, name, mark):
            widget.bind("<Button-1>", lambda _event, item=path: self._select_checkpoint(item))
            widget.bind("<Enter>", lambda _event, item=path: self._hover_checkpoint(item, True))
            widget.bind("<Leave>", lambda _event, item=path: self._hover_checkpoint(item, False))

    def _select_checkpoint(self, path: Path) -> None:
        self.selected_checkpoint = path
        self._set_value_box(self.checkpoint_box, path.name)
        for candidate, record in self._pt_rows.items():
            selected = candidate == path
            bg = Theme.selected if selected else Theme.inset
            record["row"].configure(bg=bg, highlightbackground=Theme.accent_strong if selected else Theme.border)
            record["name"].configure(bg=bg, fg=Theme.white if selected else Theme.text)
            record["mark"].configure(bg=bg, text="Selected" if selected else "")
        self._refresh_launch()

    def _hover_checkpoint(self, path: Path, entering: bool) -> None:
        if path not in self._pt_rows or self.selected_checkpoint == path:
            return
        record = self._pt_rows[path]
        bg = Theme.hover if entering else Theme.inset
        record["row"].configure(bg=bg)
        record["name"].configure(bg=bg)
        record["mark"].configure(bg=bg)

    def _refresh_launch(self) -> None:
        self.launch_button.set_enabled(self.selected_model_dir is not None and self.selected_checkpoint is not None)

    def _move_selection(self, delta: int) -> str:
        if not self._visible:
            return "break"
        paths = [node.path for node in self._visible]
        if self._selected_path in paths:
            index = paths.index(self._selected_path)
        else:
            index = 0 if delta > 0 else -1
        index = max(0, min(len(self._visible) - 1, index + delta))
        self._activate(self._visible[index], toggle=False)
        return "break"

    def _expand_selected(self) -> str:
        node = self._nodes.get(self._selected_path) if self._selected_path else None
        if node and node.children and node.path not in self._expanded:
            self._expanded.add(node.path)
            self._rebuild_tree()
        return "break"

    def _collapse_or_ascend(self) -> str:
        if self._selected_path is None:
            return "break"
        if self._selected_path in self._expanded:
            self._expanded.remove(self._selected_path)
            self._rebuild_tree()
            return "break"
        parent = self._parent.get(self._selected_path)
        parent_node = self._nodes.get(parent) if parent is not None else None
        if parent_node is not None:
            self._activate(parent_node, toggle=False)
        return "break"

    def _activate(self, node: Node, *, toggle: bool) -> None:
        self._selected_path = node.path
        if toggle and node.children:
            if node.path in self._expanded:
                self._expanded.remove(node.path)
            else:
                self._expanded.add(node.path)
        if node.is_model:
            self._show_model(node)
        else:
            self._show_directory(node)
        self._rebuild_tree()
        self._reveal(node.path)

    def _reveal(self, path: Path) -> None:
        record = self._rows.get(path)
        if record is None:
            return
        self.tree_inner.update_idletasks()
        row = record["parts"][0]
        y = row.winfo_y()
        total = max(1, self.tree_inner.winfo_height())
        view_top = self.tree_canvas.canvasy(0)
        view_height = self.tree_canvas.winfo_height()
        if y < view_top or y + self._row_height > view_top + view_height:
            self.tree_canvas.yview_moveto(max(0, (y - 8) / total))

    def _open_launch(self) -> None:
        from mm_transfer.gui.launch_window import LaunchWindow

        if self.selected_model_dir is None or self.selected_checkpoint is None:
            return
        if self._launch_win is not None and self._launch_win.winfo_exists():
            self._launch_win.lift()
            return
        # Map the next window before hiding this one so it stays on screen.
        self._launch_win = LaunchWindow(self, self._close_launch)
        self._launch_win.update_idletasks()
        self._launch_win.deiconify()
        self._launch_win.lift()
        self._launch_win.focus_force()
        self.withdraw()

    def _close_launch(self) -> None:
        window = self._launch_win
        self._launch_win = None
        if window is not None:
            try:
                if window.winfo_exists():
                    window.protocol("WM_DELETE_WINDOW", lambda: None)
                    window.destroy()
            except tk.TclError:
                pass
        if self.winfo_exists():
            self.deiconify()
            self.lift()
            self.focus_force()

    def _quit(self) -> None:
        if getattr(self, "_quitting", False):
            return
        self._quitting = True
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
            self._resize_job = None
        window = self._launch_win
        self._launch_win = None
        if window is not None:
            try:
                if window.winfo_exists():
                    window._cancel.set()
                    window.protocol("WM_DELETE_WINDOW", lambda: None)
                    window.destroy()
            except tk.TclError:
                pass
        self.quit()
        self.destroy()
