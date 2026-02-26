# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unified plugin marketplace floating panel."""

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .marketplace import (
    MarketplacePluginEntry,
    PluginMarketplaceCatalog,
)
from .plugin import PluginInfo, PluginState
from .types import Panel

MAX_OUTPUT_LINES = 100
SUCCESS_DISMISS_SEC = 3.0

_PHASE_MILESTONES: List[Tuple[str, float]] = [
    ("cloning", 0.05),
    ("cloned", 0.30),
    ("downloading", 0.05),
    ("extracting", 0.35),
    ("syncing dependencies", 0.40),
    ("updating", 0.05),
    ("updated", 0.50),
    ("unloading", 0.20),
    ("uninstalling", 0.20),
]
_NUDGE_FRACTION = 0.08
_PROGRESS_CEILING = 0.95


class CardOpPhase(Enum):
    IDLE = "idle"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    ERROR = "error"


@dataclass
class CardOpState:
    phase: CardOpPhase = CardOpPhase.IDLE
    message: str = ""
    progress: float = 0.0
    output_lines: List[str] = field(default_factory=list)
    finished_at: float = 0.0


class PluginMarketplacePanel(Panel):
    """Floating plugin window for browsing, installing, and managing plugins."""

    idname = "lfs.plugin_marketplace"
    label = "Plugin Marketplace"
    space = "FLOATING"
    order = 91
    options = {"DEFAULT_CLOSED"}

    GRID_COLUMNS = 100
    CARD_WIDTH = 330
    CARD_HEIGHT = 200
    CARD_SPACING = 12
    FILTER_WIDTH = 140
    SORT_WIDTH = 170

    def __init__(self):
        self._catalog = PluginMarketplaceCatalog()
        self._url_plugin_names: Dict[str, str] = {}
        self._manual_url = ""
        self._install_filter_idx = 0
        self._sort_idx = 0

        self._card_ops: Dict[str, CardOpState] = {}
        self._lock = threading.Lock()
        self._pending_uninstall_name = ""
        self._pending_uninstall_card_id = ""
        self._pending_uninstall_open = False

        self._discover_cache: Optional[List[PluginInfo]] = None
        self._clear_manual_url = False

    def draw(self, layout):
        import lichtfeld as lf
        from .manager import PluginManager

        tr = lf.ui.tr
        theme = lf.ui.theme()
        palette = theme.palette
        mgr = PluginManager.instance()
        self._ensure_loaded()

        if self._clear_manual_url:
            self._manual_url = ""
            self._clear_manual_url = False

        scale = layout.get_dpi_scale()
        self._draw_uninstall_confirmation_modal(layout, mgr, scale)

        layout.text_colored(
            tr("plugin_marketplace.title_line"),
            palette.info,
        )
        layout.spacing()

        layout.text_colored(tr("plugin_marketplace.warning_body"), palette.warning)
        layout.spacing()

        self._draw_marketplace_controls(layout, mgr, scale)
        layout.spacing()

        entries, is_loading = self._catalog.snapshot()
        entries = self._with_local_plugins(entries, mgr)
        installed_lookup = self._get_installed_plugin_lookup(mgr)
        installed_versions = self._get_installed_plugin_versions(mgr)
        installed_names = set(installed_lookup.values())
        entries = self._filter_and_sort_entries(entries, set(installed_lookup.keys()), installed_names)

        layout.spacing()
        layout.separator()
        layout.spacing()

        if not entries:
            layout.text_disabled(tr("plugin_marketplace.no_plugins"))
            layout.text_disabled(tr("plugin_marketplace.edit_list_hint"))
            return

        card_w = self.CARD_WIDTH * scale
        card_h = self.CARD_HEIGHT * scale
        spacing = self.CARD_SPACING * scale

        avail_w, _ = layout.get_content_region_avail()
        visible_columns = self._visible_columns(avail_w, card_w, spacing)
        card_w = self._fit_card_width(avail_w, card_w, spacing, visible_columns, scale)

        if layout.begin_child("##plugin_marketplace_scroll", (0, 0), border=False):
            row_count = (len(entries) + visible_columns - 1) // visible_columns
            for row in range(row_count):
                base = row * visible_columns
                drawn = 0
                for col in range(visible_columns):
                    idx = base + col
                    if idx >= len(entries):
                        break

                    if drawn > 0:
                        layout.same_line(spacing=spacing)

                    self._draw_plugin_card(
                        layout,
                        mgr,
                        idx,
                        entries[idx],
                        installed_lookup,
                        installed_versions,
                        installed_names,
                        card_w,
                        card_h,
                        scale,
                    )
                    drawn += 1

                if drawn > 0:
                    layout.spacing()
        layout.end_child()

    def _ensure_loaded(self):
        self._catalog.refresh_async()

    def _invalidate_discover_cache(self):
        self._discover_cache = None

    def _get_discovered_plugins(self, mgr) -> List[PluginInfo]:
        cache = self._discover_cache
        if cache is None:
            cache = mgr.discover()
            self._discover_cache = cache
        return cache

    def _get_card_state(self, card_id: str) -> CardOpState:
        with self._lock:
            state = self._card_ops.get(card_id)
            if state is None:
                return CardOpState()
            if state.phase == CardOpPhase.SUCCESS and state.finished_at > 0:
                if time.monotonic() - state.finished_at >= SUCCESS_DISMISS_SEC:
                    state.phase = CardOpPhase.IDLE
                    state.message = ""
                    state.progress = 0.0
                    state.output_lines.clear()
                    state.finished_at = 0.0
            return CardOpState(
                phase=state.phase,
                message=state.message,
                progress=state.progress,
                output_lines=list(state.output_lines),
                finished_at=state.finished_at,
            )

    def _draw_manual_install_controls(self, layout, mgr, scale: float):
        import lichtfeld as lf

        tr = lf.ui.tr
        card_id = "__manual_url__"
        card_state = self._get_card_state(card_id)
        in_progress = card_state.phase == CardOpPhase.IN_PROGRESS

        layout.label(tr("plugin_manager.github_url_or_shorthand"))
        input_width = max(140.0, layout.get_content_region_avail()[0] - 104.0 * scale)
        layout.set_next_item_width(input_width)
        _, self._manual_url = layout.input_text("##marketplace_install_url", self._manual_url)

        layout.same_line(spacing=8.0 * scale)
        if in_progress:
            layout.begin_disabled()
        if layout.button_styled(tr("plugin_manager.button.install_plugin"), "success", (0, 28)):
            if not in_progress:
                self._install_plugin_from_url(mgr, self._manual_url, card_id)
        if in_progress:
            layout.end_disabled()

        self._draw_inline_feedback(layout, card_state, scale)

        if layout.tree_node(tr("plugin_manager.supported_formats")):
            layout.bullet_text("https://github.com/owner/repo")
            layout.bullet_text("github:owner/repo")
            layout.bullet_text("owner/repo")
            layout.tree_pop()

    def _draw_inline_feedback(self, layout, card_state: CardOpState, scale: float):
        import lichtfeld as lf

        palette = lf.ui.theme().palette
        tr = lf.ui.tr

        if card_state.phase == CardOpPhase.IN_PROGRESS:
            layout.progress_bar(
                card_state.progress,
                card_state.message or tr("plugin_manager.working"),
            )
        elif card_state.phase == CardOpPhase.SUCCESS:
            layout.progress_bar(1.0, card_state.message)
        elif card_state.phase == CardOpPhase.ERROR:
            layout.text_colored(card_state.message, palette.error)
            if layout.is_item_hovered() and card_state.message:
                layout.set_tooltip(card_state.message)

    @staticmethod
    def _calc_bottom_area_height(
        card_state: CardOpState,
        is_installed: bool,
        is_local_only: bool,
        button_height: float,
        button_spacing: float,
        scale: float,
    ) -> float:
        if card_state.phase != CardOpPhase.IDLE:
            return button_height

        if not is_installed:
            return button_height

        item_sp = 6 * scale
        return button_height + item_sp + button_height

    def _draw_plugin_card(
        self,
        layout,
        mgr,
        idx: int,
        entry: MarketplacePluginEntry,
        installed_lookup: Dict[str, str],
        installed_versions: Dict[str, str],
        installed_names: Set[str],
        card_w: float,
        card_h: float,
        scale: float,
    ):
        import lichtfeld as lf

        tr = lf.ui.tr
        theme = lf.ui.theme()
        palette = theme.palette
        card_rounding = max(theme.sizes.frame_rounding, theme.sizes.popup_rounding)
        layout.push_style_var("ChildRounding", card_rounding * scale)
        layout.push_style_color("ChildBg", palette.surface)

        plugin_name = self._resolve_entry_plugin_name(entry, installed_lookup, installed_names)
        plugin_state = mgr.get_state(plugin_name) if plugin_name else None
        is_installed = plugin_name is not None
        is_local = self._is_local_entry(entry)
        has_github = bool(entry.github_url)
        is_local_only = self._is_local_only_entry(entry)

        card_id = entry.registry_id or entry.name or str(idx)
        card_state = self._get_card_state(card_id)

        border_color = palette.border
        if card_state.phase == CardOpPhase.IN_PROGRESS:
            border_color = palette.info
        elif card_state.phase == CardOpPhase.SUCCESS:
            border_color = palette.success
        elif card_state.phase == CardOpPhase.ERROR:
            border_color = palette.error
        layout.push_style_color("Border", border_color)

        this_card_busy = card_state.phase == CardOpPhase.IN_PROGRESS

        if layout.begin_child(f"##plugin_card_{card_id}", (card_w, card_h), border=True):
            info_start = layout.get_cursor_pos()
            self._draw_card_info(
                layout, entry, plugin_name, plugin_state,
                is_installed, is_local, is_local_only, installed_versions, scale,
            )
            info_end_y = layout.get_cursor_pos()[1]

            if has_github:
                info_w = layout.get_content_region_avail()[0]
                info_h = info_end_y - info_start[1]
                saved = layout.get_cursor_pos()
                layout.set_cursor_pos(info_start)
                layout.invisible_button(f"##info_link_{card_id}", (info_w, info_h))
                if layout.is_item_hovered():
                    layout.set_mouse_cursor_hand()
                    layout.set_tooltip(entry.github_url)
                if layout.is_item_clicked():
                    lf.ui.open_url(entry.github_url)
                layout.set_cursor_pos(saved)

            button_spacing = 6 * scale
            button_height = 25 * scale
            avail_button_w, avail_h = layout.get_content_region_avail()
            pad = 4 * scale

            bottom_h = self._calc_bottom_area_height(
                card_state, is_installed, is_local_only,
                button_height, button_spacing, scale,
            )
            skip = avail_h - bottom_h - pad
            if skip > 0:
                cursor_x, cursor_y = layout.get_cursor_pos()
                layout.set_cursor_pos((cursor_x, cursor_y + skip))

            if card_state.phase != CardOpPhase.IDLE:
                self._draw_inline_feedback(layout, card_state, scale)
            elif is_installed:
                self._draw_card_buttons_installed(
                    layout, mgr, idx, entry, plugin_name, plugin_state,
                    is_local, is_local_only, this_card_busy,
                    card_id, avail_button_w, button_spacing, button_height, scale,
                )
            else:
                if is_local_only:
                    layout.spacing()
                    layout.end_child()
                    layout.pop_style_color(2)
                    layout.pop_style_var()
                    return

                self._draw_card_buttons_not_installed(
                    layout, mgr, idx, entry, this_card_busy,
                    card_id, avail_button_w, button_spacing, button_height, scale,
                )
        layout.end_child()

        layout.pop_style_color(2)
        layout.pop_style_var()

    def _draw_card_info(
        self, layout, entry, plugin_name, plugin_state,
        is_installed, is_local, is_local_only, installed_versions, scale,
    ):
        import lichtfeld as lf

        tr = lf.ui.tr
        palette = lf.ui.theme().palette

        short_name = entry.name or entry.repo or tr("plugin_marketplace.unknown_plugin")
        repo_label = f"{entry.owner}/{entry.repo}" if entry.owner and entry.repo else entry.repo
        desc = entry.description
        if not desc and plugin_name and self._discover_cache:
            for p in self._discover_cache:
                if p.name == plugin_name:
                    desc = p.description
                    break
        description = self._truncate_text(desc or tr("plugin_marketplace.no_description"), 90)

        layout.text_colored(short_name, palette.text)
        if plugin_name and plugin_state == PluginState.ACTIVE:
            version = installed_versions.get(plugin_name, "").strip()
            if version:
                version_label = version if version.lower().startswith("v") else f"v{version}"
                layout.same_line(spacing=6 * scale)
                layout.text_colored(version_label, palette.info)
        if repo_label:
            layout.text_disabled(repo_label)
        if not is_local_only:
            metrics = []
            if entry.stars > 0:
                metrics.append(f"{tr('plugin_marketplace.stars')}: {entry.stars}")
            if entry.downloads > 0:
                metrics.append(f"{tr('plugin_marketplace.downloads')}: {entry.downloads}")
            if metrics:
                layout.text_colored("  |  ".join(metrics), palette.warning)

        tags = self._entry_type_tags(entry)
        if tags:
            layout.text_disabled("  |  ".join(tags[:3]))
        if is_local:
            layout.text_colored(tr("plugin_marketplace.local_install"), palette.info)

        if is_installed:
            state_str = plugin_state.value if plugin_state else tr("plugin_manager.status_not_loaded")
            layout.text_colored(
                f"{tr('plugin_manager.status')}: {state_str}",
                palette.success if plugin_state == PluginState.ACTIVE else palette.text_dim,
            )

        if entry.error:
            layout.text_colored(tr("plugin_marketplace.invalid_link"), palette.error)
        else:
            layout.text_wrapped(description)

        layout.spacing()
        layout.separator()
        layout.spacing()

    def _draw_card_buttons_installed(
        self, layout, mgr, idx, entry, plugin_name, plugin_state,
        is_local, is_local_only, this_card_busy,
        card_id, avail_button_w, button_spacing, button_height, scale,
    ):
        import lichtfeld as lf
        from .settings import SettingsManager

        tr = lf.ui.tr

        if plugin_name:
            prefs = SettingsManager.instance().get(plugin_name)
            startup = prefs.get("load_on_startup", False)
            changed, startup = layout.checkbox(
                f"{tr('plugin_marketplace.load_on_startup')}##startup_{idx}", startup
            )
            if changed:
                prefs.set("load_on_startup", startup)

        if this_card_busy:
            layout.begin_disabled()

        bw = max(40.0, (avail_button_w - button_spacing * 2.0) / 3.0)

        if is_local_only:
            load_label = (
                tr("plugin_manager.button.unload")
                if plugin_state == PluginState.ACTIVE
                else tr("plugin_manager.button.load")
            )
            load_style = "warning" if plugin_state == PluginState.ACTIVE else "success"
            if layout.button_styled(
                f"{load_label}##loadtoggle_{idx}", load_style, (bw, button_height),
            ):
                if plugin_state == PluginState.ACTIVE:
                    self._unload_plugin(mgr, plugin_name, card_id)
                else:
                    self._load_plugin(mgr, plugin_name, card_id)
            layout.same_line(spacing=button_spacing)
            if layout.button_styled(
                f"{tr('plugin_manager.button.uninstall')}##uninstall_{idx}",
                "error", (bw, button_height),
            ):
                self._request_uninstall_confirmation(plugin_name, card_id)
        elif is_local and bool(entry.github_url):
            load_label = (
                tr("plugin_manager.button.unload")
                if plugin_state == PluginState.ACTIVE
                else tr("plugin_manager.button.load")
            )
            load_style = "warning" if plugin_state == PluginState.ACTIVE else "success"
            if layout.button_styled(
                f"{load_label}##loadtoggle_{idx}", load_style, (bw, button_height),
            ):
                if plugin_state == PluginState.ACTIVE:
                    self._unload_plugin(mgr, plugin_name, card_id)
                else:
                    self._load_plugin(mgr, plugin_name, card_id)
            layout.same_line(spacing=button_spacing)
            if layout.button_styled(
                f"{tr('plugin_manager.button.update')}##update_{idx}",
                "primary", (bw, button_height),
            ):
                self._update_plugin(mgr, plugin_name, card_id)
            layout.same_line(spacing=button_spacing)
            if layout.button_styled(
                f"{tr('plugin_manager.button.uninstall')}##uninstall_{idx}",
                "error", (bw, button_height),
            ):
                self._request_uninstall_confirmation(plugin_name, card_id)
        else:
            if plugin_state == PluginState.ACTIVE:
                if layout.button_styled(
                    f"{tr('plugin_manager.button.reload')}##reload_{idx}",
                    "primary", (bw, button_height),
                ):
                    self._reload_plugin(mgr, plugin_name, card_id)
                layout.same_line(spacing=button_spacing)
                if layout.button_styled(
                    f"{tr('plugin_manager.button.unload')}##unload_{idx}",
                    "warning", (bw, button_height),
                ):
                    self._unload_plugin(mgr, plugin_name, card_id)
            else:
                if layout.button_styled(
                    f"{tr('plugin_manager.button.load')}##load_{idx}",
                    "success", (bw, button_height),
                ):
                    self._load_plugin(mgr, plugin_name, card_id)
                layout.same_line(spacing=button_spacing)
                if layout.button_styled(
                    f"{tr('plugin_manager.button.update')}##update_{idx}",
                    "primary", (bw, button_height),
                ):
                    self._update_plugin(mgr, plugin_name, card_id)
            layout.same_line(spacing=button_spacing)
            if layout.button_styled(
                f"{tr('plugin_manager.button.uninstall')}##uninstall_{idx}",
                "error", (bw, button_height),
            ):
                self._request_uninstall_confirmation(plugin_name, card_id)

        if this_card_busy:
            layout.end_disabled()

    def _draw_card_buttons_not_installed(
        self, layout, mgr, idx, entry, this_card_busy,
        card_id, avail_button_w, button_spacing, button_height, scale,
    ):
        import lichtfeld as lf

        tr = lf.ui.tr
        bw = max(40.0, (avail_button_w - button_spacing * 2.0) / 3.0)
        disable_install = this_card_busy or bool(entry.error)

        if disable_install:
            layout.begin_disabled()
        if layout.button_styled(
            f"{tr('plugin_marketplace.button.install')}##install_{idx}",
            "success",
            (bw, button_height),
        ):
            if not disable_install:
                self._install_plugin_from_marketplace(mgr, entry, card_id)
        if disable_install:
            layout.end_disabled()

    def _draw_marketplace_controls(self, layout, mgr, scale: float):
        import lichtfeld as lf

        tr = lf.ui.tr
        avail_w, _ = layout.get_content_region_avail()
        filter_items = [
            tr("plugin_marketplace.filter.all"),
            tr("plugin_marketplace.filter.installed"),
            tr("plugin_marketplace.filter.not_installed"),
        ]
        sort_items = [
            tr("plugin_marketplace.sort.popularity_desc"),
            tr("plugin_marketplace.sort.popularity_asc"),
            tr("plugin_marketplace.sort.name_asc"),
            tr("plugin_marketplace.sort.name_desc"),
        ]

        filter_w = max(1.0, min(self.FILTER_WIDTH * scale, avail_w))
        remaining = max(1.0, avail_w - filter_w - 8 * scale)
        sort_w = max(1.0, min(self.SORT_WIDTH * scale, remaining))

        layout.text_disabled(tr("plugin_marketplace.filter_label"))
        layout.same_line(spacing=6 * scale)
        layout.set_next_item_width(filter_w)
        _, self._install_filter_idx = layout.combo(
            "##install_filter",
            self._install_filter_idx,
            filter_items,
        )
        layout.same_line(spacing=10 * scale)
        layout.text_disabled(tr("plugin_marketplace.sort_label"))
        layout.same_line(spacing=6 * scale)
        layout.set_next_item_width(sort_w)
        _, self._sort_idx = layout.combo("##sort_filter", self._sort_idx, sort_items)

        layout.spacing()
        self._draw_manual_install_controls(layout, mgr, scale)

    @staticmethod
    def _visible_columns(avail_w: float, card_w: float, spacing: float) -> int:
        if avail_w <= 0:
            return 1
        return max(1, min(PluginMarketplacePanel.GRID_COLUMNS, int((avail_w + spacing) // (card_w + spacing))))

    @staticmethod
    def _fit_card_width(
        avail_w: float,
        preferred_card_w: float,
        spacing: float,
        columns: int,
        scale: float,
    ) -> float:
        if columns <= 0:
            return preferred_card_w
        usable = max(1.0, avail_w - (columns - 1) * spacing - 2.0 * scale)
        return max(1.0, min(preferred_card_w, usable / columns))

    def _filter_and_sort_entries(
        self,
        entries: List[MarketplacePluginEntry],
        installed_keys: Set[str],
        installed_names: Set[str],
    ) -> List[MarketplacePluginEntry]:
        filtered = []
        for entry in entries:
            is_installed = self._is_marketplace_entry_installed(entry, installed_keys, installed_names)
            if self._install_filter_idx == 1 and not is_installed:
                continue
            if self._install_filter_idx == 2 and is_installed:
                continue
            filtered.append(entry)

        def popularity(e):
            return (e.stars + e.downloads, e.name.lower())

        if self._sort_idx == 1:
            return sorted(filtered, key=popularity)
        if self._sort_idx == 2:
            return sorted(filtered, key=lambda e: e.name.lower())
        if self._sort_idx == 3:
            return sorted(filtered, key=lambda e: e.name.lower(), reverse=True)
        return sorted(filtered, key=popularity, reverse=True)

    @staticmethod
    def _advance_progress(state: CardOpState, msg: str):
        lower = msg.lower()
        for keyword, milestone in _PHASE_MILESTONES:
            if keyword in lower:
                state.progress = max(state.progress, milestone)
                return
        remaining = _PROGRESS_CEILING - state.progress
        if remaining > 0.01:
            state.progress += remaining * _NUDGE_FRACTION

    def _run_async(self, card_id: str, operation, success_msg: str, error_prefix: str):
        with self._lock:
            existing = self._card_ops.get(card_id)
            if existing and existing.phase == CardOpPhase.IN_PROGRESS:
                return
            state = CardOpState(phase=CardOpPhase.IN_PROGRESS)
            self._card_ops[card_id] = state

        def on_progress(msg: str):
            with self._lock:
                self._advance_progress(state, msg)
                state.message = msg
                state.output_lines.append(msg)
                if len(state.output_lines) > MAX_OUTPUT_LINES:
                    state.output_lines = state.output_lines[-MAX_OUTPUT_LINES:]

        def worker():
            try:
                result = operation(on_progress)
                if result is False:
                    raise RuntimeError(error_prefix)
                with self._lock:
                    state.progress = 1.0
                    if isinstance(result, str):
                        state.message = success_msg.format(result)
                    else:
                        state.message = success_msg
                    state.phase = CardOpPhase.SUCCESS
                    state.finished_at = time.monotonic()
            except Exception as e:
                detail = str(e).strip()
                with self._lock:
                    if detail:
                        state.message = f"{error_prefix}: {detail}"
                    else:
                        state.message = error_prefix
                    state.phase = CardOpPhase.ERROR

        threading.Thread(target=worker, daemon=True).start()

    def _install_plugin_from_marketplace(self, mgr, entry: MarketplacePluginEntry, card_id: str):
        import lichtfeld as lf

        tr = lf.ui.tr

        def do_install(on_progress):
            if entry.registry_id:
                name = mgr.install_from_registry(entry.registry_id, on_progress=on_progress)
            else:
                name = mgr.install(entry.source_url, on_progress=on_progress)
            if mgr.get_state(name) == PluginState.ERROR:
                err = mgr.get_error(name) or tr("plugin_manager.status.load_failed")
                raise RuntimeError(err)
            norm_url = self._normalize_url(entry.source_url)
            if norm_url:
                with self._lock:
                    self._url_plugin_names[norm_url] = name
            self._invalidate_discover_cache()
            return name

        self._run_async(
            card_id,
            do_install,
            tr("plugin_manager.status.installed"),
            tr("plugin_manager.status.install_failed"),
        )

    def _install_plugin_from_url(self, mgr, url: str, card_id: str):
        import lichtfeld as lf

        tr = lf.ui.tr
        clean_url = url.strip()
        if not clean_url:
            with self._lock:
                self._card_ops[card_id] = CardOpState(
                    phase=CardOpPhase.ERROR,
                    message=tr("plugin_manager.error.enter_github_url"),
                )
            return

        def do_install(on_progress):
            name = mgr.install(clean_url, on_progress=on_progress)
            if mgr.get_state(name) == PluginState.ERROR:
                err = mgr.get_error(name) or tr("plugin_manager.status.load_failed")
                raise RuntimeError(err)
            self._clear_manual_url = True
            with self._lock:
                self._url_plugin_names[self._normalize_url(clean_url)] = name
            self._invalidate_discover_cache()
            return name

        self._run_async(
            card_id,
            do_install,
            tr("plugin_manager.status.installed"),
            tr("plugin_manager.status.install_failed"),
        )

    def _load_plugin(self, mgr, name: str, card_id: str):
        import lichtfeld as lf

        tr = lf.ui.tr

        def do_load(on_progress):
            ok = mgr.load(name, on_progress=on_progress)
            if not ok:
                err = mgr.get_error(name) or tr("plugin_manager.status.load_failed")
                raise RuntimeError(err)
            self._invalidate_discover_cache()
            return True

        self._run_async(
            card_id,
            do_load,
            tr("plugin_manager.status.loaded").format(name=name),
            tr("plugin_manager.status.load_failed"),
        )

    def _unload_plugin(self, mgr, name: str, card_id: str):
        import lichtfeld as lf

        tr = lf.ui.tr

        def do_unload(on_progress):
            on_progress(tr("plugin_manager.status.unloading").format(name=name))
            if not mgr.unload(name):
                raise RuntimeError(tr("plugin_manager.status.unload_failed"))
            self._invalidate_discover_cache()

        self._run_async(
            card_id,
            do_unload,
            tr("plugin_manager.status.unloaded").format(name=name),
            tr("plugin_manager.status.unload_failed"),
        )

    def _reload_plugin(self, mgr, name: str, card_id: str):
        import lichtfeld as lf

        tr = lf.ui.tr

        def do_reload(on_progress):
            mgr.unload(name)
            ok = mgr.load(name, on_progress=on_progress)
            if not ok:
                err = mgr.get_error(name) or tr("plugin_manager.status.reload_failed")
                raise RuntimeError(err)
            self._invalidate_discover_cache()
            return True

        self._run_async(
            card_id,
            do_reload,
            tr("plugin_manager.status.reloaded").format(name=name),
            tr("plugin_manager.status.reload_failed"),
        )

    def _update_plugin(self, mgr, name: str, card_id: str):
        import lichtfeld as lf

        tr = lf.ui.tr

        def do_update(on_progress):
            mgr.update(name, on_progress=on_progress)
            self._invalidate_discover_cache()
            return True

        self._run_async(
            card_id,
            do_update,
            tr("plugin_manager.status.updated").format(name=name),
            tr("plugin_manager.status.update_failed"),
        )

    def _uninstall_plugin(self, mgr, name: str, card_id: str):
        import lichtfeld as lf

        tr = lf.ui.tr

        def do_uninstall(on_progress):
            on_progress(tr("plugin_manager.status.uninstalling").format(name=name))
            if not mgr.uninstall(name):
                raise RuntimeError(tr("plugin_manager.status.uninstall_failed"))
            self._invalidate_discover_cache()

        self._run_async(
            card_id,
            do_uninstall,
            tr("plugin_manager.status.uninstalled").format(name=name),
            tr("plugin_manager.status.uninstall_failed"),
        )

    def _request_uninstall_confirmation(self, name: str, card_id: str):
        if not name:
            return
        self._pending_uninstall_name = name
        self._pending_uninstall_card_id = card_id
        self._pending_uninstall_open = True

    def _draw_uninstall_confirmation_modal(self, layout, mgr, scale: float):
        import lichtfeld as lf

        tr = lf.ui.tr
        if not self._pending_uninstall_name and not self._pending_uninstall_open:
            return

        popup_title = tr("plugin_marketplace.confirm_uninstall_title")
        popup_id = f"{popup_title}##plugin_marketplace_uninstall_confirm"

        if self._pending_uninstall_open:
            layout.set_next_window_pos_viewport_center(always=True)
            layout.set_next_window_size((380 * scale, 0))
            layout.open_popup(popup_id)
            self._pending_uninstall_open = False

        layout.push_modal_style()
        if layout.begin_popup_modal(popup_id):
            avail_width = layout.get_content_region_avail()[0]
            text_width = layout.calc_text_size(
                tr("plugin_marketplace.confirm_uninstall_message").format(name=self._pending_uninstall_name)
            )[0]
            layout.set_cursor_pos_x(layout.get_cursor_pos()[0] + max(0.0, (avail_width - text_width) * 0.5))
            layout.text_wrapped(
                tr("plugin_marketplace.confirm_uninstall_message").format(name=self._pending_uninstall_name)
            )
            layout.spacing()
            layout.separator()
            layout.spacing()

            button_width = 92 * scale
            button_spacing = 8 * scale
            avail_width = layout.get_content_region_avail()[0]
            total_width = button_width * 2 + button_spacing
            layout.set_cursor_pos_x(layout.get_cursor_pos()[0] + max(0.0, (avail_width - total_width) * 0.5))

            if layout.button_styled(
                tr("plugin_marketplace.confirm_uninstall_no"),
                "secondary",
                (button_width, 0),
            ) or lf.ui.is_key_pressed(lf.ui.Key.ESCAPE):
                self._pending_uninstall_name = ""
                layout.close_current_popup()

            layout.same_line(0, button_spacing)
            if layout.button_styled(
                tr("plugin_marketplace.confirm_uninstall_yes"),
                "error",
                (button_width, 0),
            ):
                uninstall_name = self._pending_uninstall_name
                uninstall_card_id = self._pending_uninstall_card_id
                self._pending_uninstall_name = ""
                layout.close_current_popup()
                self._uninstall_plugin(mgr, uninstall_name, uninstall_card_id)

            layout.end_popup_modal()
        layout.pop_modal_style()


    def _with_local_plugins(self, entries: List[MarketplacePluginEntry], mgr) -> List[MarketplacePluginEntry]:
        merged = list(entries)
        known_keys: Set[str] = set()
        catalog_urls: Set[str] = set()
        for entry in merged:
            known_keys.update(self._entry_keys(entry))
            norm = self._normalize_url(entry.source_url)
            if norm:
                catalog_urls.add(norm)

        for plugin in self._get_discovered_plugins(mgr):
            plugin_keys = self._plugin_keys(plugin.name, plugin.path.name)
            if any(k in known_keys for k in plugin_keys):
                continue

            remote_url = self._git_remote_url(plugin.path)
            if remote_url:
                norm_remote = self._normalize_url(remote_url)
                if norm_remote in catalog_urls:
                    with self._lock:
                        self._url_plugin_names[norm_remote] = plugin.name
                    known_keys.update(plugin_keys)
                    continue

            source_path = str(plugin.path)
            merged.append(
                MarketplacePluginEntry(
                    source_url=source_path,
                    github_url=remote_url or "",
                    owner="",
                    repo=plugin.path.name,
                    name=plugin.name,
                    description=plugin.description or "",
                )
            )
            with self._lock:
                self._url_plugin_names[self._normalize_url(source_path)] = plugin.name
                if remote_url:
                    self._url_plugin_names[self._normalize_url(remote_url)] = plugin.name
            known_keys.update(plugin_keys)

        return merged

    @staticmethod
    def _git_remote_url(plugin_path: Path) -> str:
        import subprocess
        if not (plugin_path / ".git").exists():
            return ""
        try:
            result = subprocess.run(
                ["git", "-C", str(plugin_path), "remote", "get-url", "origin"],
                capture_output=True, text=True, timeout=3,
            )
            url = result.stdout.strip()
            if url.endswith(".git"):
                url = url[:-4]
            return url
        except Exception:
            return ""

    def _get_installed_plugin_lookup(self, mgr) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        for plugin in self._get_discovered_plugins(mgr):
            for key in self._plugin_keys(plugin.name, plugin.path.name):
                lookup[key] = plugin.name
        return lookup

    def _get_installed_plugin_versions(self, mgr) -> Dict[str, str]:
        return {plugin.name: plugin.version for plugin in self._get_discovered_plugins(mgr)}

    def _resolve_entry_plugin_name(
        self,
        entry: MarketplacePluginEntry,
        installed_lookup: Dict[str, str],
        installed_names: Set[str],
    ):
        norm_url = self._normalize_url(entry.source_url)
        by_url = None
        if norm_url:
            with self._lock:
                by_url = self._url_plugin_names.get(norm_url)
        if by_url and by_url in installed_names:
            return by_url
        for key in self._entry_keys(entry):
            plugin_name = installed_lookup.get(key)
            if plugin_name:
                return plugin_name
        return None

    @staticmethod
    def _normalize_url(url: str) -> str:
        return str(url or "").strip().rstrip("/")

    def _is_marketplace_entry_installed(
        self,
        entry: MarketplacePluginEntry,
        installed_keys: Set[str],
        installed_names: Set[str],
    ) -> bool:
        if any(key in installed_keys for key in self._entry_keys(entry)):
            return True
        norm_url = self._normalize_url(entry.source_url)
        if not norm_url:
            return False
        with self._lock:
            by_url = self._url_plugin_names.get(norm_url)
        return by_url is not None and by_url in installed_names

    @staticmethod
    def _is_local_entry(entry: MarketplacePluginEntry) -> bool:
        source = str(entry.source_url or "").strip()
        if not source:
            return False
        if source.startswith(("http://", "https://", "github:")):
            return False
        return Path(source).is_absolute() or source.startswith("~")

    @staticmethod
    def _is_local_only_entry(entry: MarketplacePluginEntry) -> bool:
        return PluginMarketplacePanel._is_local_entry(entry) and not bool(entry.github_url)

    def _entry_keys(self, entry: MarketplacePluginEntry) -> Set[str]:
        from .installer import normalize_repo_name

        normalized_repo = normalize_repo_name(entry.repo) if entry.repo else ""
        return self._plugin_keys(
            entry.repo,
            entry.name,
            normalized_repo,
            f"{entry.owner}-{entry.repo}" if entry.owner and entry.repo else "",
            f"{entry.owner}_{entry.repo}" if entry.owner and entry.repo else "",
        )

    @staticmethod
    def _plugin_keys(*values: str) -> Set[str]:
        keys = set()
        for value in values:
            raw = str(value or "").strip()
            if not raw:
                continue
            lower = raw.lower()
            keys.add(lower)
            normalized = "".join(ch for ch in lower if ch.isalnum())
            if normalized:
                keys.add(normalized)
        return keys

    @staticmethod
    def _entry_type_tags(entry: MarketplacePluginEntry) -> List[str]:
        tags: List[str] = []
        for topic in entry.topics:
            clean = topic.replace("_", " ").replace("-", " ").strip()
            if not clean:
                continue
            pretty = " ".join(part.capitalize() for part in clean.split())
            if pretty and pretty not in tags:
                tags.append(pretty)
        if entry.language and entry.language not in tags and entry.language.lower() != "python":
            tags.append(entry.language)
        return tags

    @staticmethod
    def _truncate_text(value: str, max_chars: int) -> str:
        if len(value) <= max_chars:
            return value
        return value[: max_chars - 3].rstrip() + "..."
