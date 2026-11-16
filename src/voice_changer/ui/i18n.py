"""In-place localization for the bilingual desktop UI."""

from collections import defaultdict
from typing import Callable

from PySide6.QtCore import QEvent, QObject, QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QAbstractButton, QComboBox, QLabel, QLineEdit, QListWidget, QPlainTextEdit, QWidget,
)

from .language_preferences import (
    DEFAULT_LANGUAGE,
    load_language,
    save_language,
    settings_path,
)

_COMMON_BUTTONS = {
    "OK": ("确定", "OK"),
    "Yes": ("是", "Yes"),
    "No": ("否", "No"),
    "Cancel": ("取消", "Cancel"),
    "Save": ("保存", "Save"),
    "Discard": ("丢弃", "Discard"),
    "Close": ("关闭", "Close"),
    "Retry": ("重试", "Retry"),
    "Apply": ("应用", "Apply"),
}


def _contains_cjk(value: str) -> bool:
    return any("\u3400" <= character <= "\u9fff" for character in value)


def _localized_line(language: str, line: str) -> str:
    if " / " not in line:
        return line
    local, translated = line.rsplit(" / ", 1)
    if _contains_cjk(local) and not _contains_cjk(translated):
        if language != "zh":
            return translated
        if ": " in translated and "：" not in local and ": " not in local:
            return f"{local}：{translated}"
        return local
    return line


def _localize_dynamic_choice(language: str, text: str) -> str:
    for source_prefix, zh_prefix, en_prefix in (
        ("预置 / Preset · ", "预置音色 · ", "Preset · "),
        ("启动配置 / Configured · ", "启动配置 · ", "Configured · "),
    ):
        if text.startswith(source_prefix):
            prefix = zh_prefix if language == "zh" else en_prefix
            return prefix + text[len(source_prefix):]

    profile_prefix = "我的音色 / Mine · "
    if text.startswith(profile_prefix):
        suffixes = (
            (" — 损坏 / damaged", " — 损坏", " — damaged"),
            (" — 暂不可用 / not usable yet", " — 暂不可用", " — not usable yet"),
        )
        for source_suffix, zh_suffix, en_suffix in suffixes:
            if text.endswith(source_suffix):
                suffix = zh_suffix if language == "zh" else en_suffix
                return (
                    ("我的音色" if language == "zh" else "My voice")
                    + " · " + text[len(profile_prefix):-len(source_suffix)] + suffix
                )
        return (
            ("我的音色" if language == "zh" else "My voice")
            + " · " + text[len(profile_prefix):]
        )
    return localize_text(language, text)


def localize_text(language: str, text: str) -> str:
    """Choose one side of known bilingual UI copy without translating user data."""
    button_label = text.lstrip("&")
    if button_label in _COMMON_BUTTONS:
        localized = _COMMON_BUTTONS[button_label][0 if language == "zh" else 1]
        return localized if language == "zh" else text.replace(button_label, localized, 1)
    lines = text.split("\n")
    result: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        translated = _localized_line(language, line)
        if translated != line:
            result.append(translated)
            index += 1
            continue

        if _contains_cjk(line):
            local_end = index
            while local_end < len(lines) and _contains_cjk(lines[local_end]):
                local_end += 1
            english_end = local_end
            while english_end < len(lines) and not _contains_cjk(lines[english_end]):
                english_end += 1
            local_block = lines[index:local_end]
            english_block = lines[local_end:english_end]
            if local_block and english_block:
                result.extend(local_block if language == "zh" else english_block)
                if language == "zh" and len(english_block) > len(local_block):
                    result.extend(english_block[len(local_block):])
                index = english_end
                continue
        result.append(line)
        index += 1

    return "\n".join(result)


class LanguageFilter(QObject):
    """Translate visible Qt copy on paint while retaining its bilingual source."""

    _TEXT_WIDGETS = (QAbstractButton, QLabel)

    def __init__(self, root: QWidget, language: Callable[[], str]) -> None:
        super().__init__(root)
        self.root = root
        self.language = language

    def _is_in_scope(self, widget: QWidget) -> bool:
        current = widget
        while current is not None:
            if current is self.root:
                return True
            current = current.parentWidget()
        return False

    def set_text(self, widget: QLabel | QAbstractButton, source: str) -> None:
        """Localize dynamic copy before Qt measures it, retaining the source."""
        if not hasattr(widget, "_i18n_sources"):
            widget._i18n_sources = {}
            widget._i18n_last = {}
        translated = localize_text(self.language(), source)
        widget._i18n_sources["text"] = source
        widget._i18n_last["text"] = translated
        if widget.text() != translated:
            widget.setText(translated)

    @staticmethod
    def _localize_attribute(
        widget: QWidget, key: str, getter: Callable[[], str], setter: Callable[[str], None],
        language: str,
    ) -> None:
        sources = getattr(widget, "_i18n_sources", None)
        if sources is None:
            sources = {}
            widget._i18n_sources = sources
            widget._i18n_last = {}
        current = getter()
        last = widget._i18n_last.get(key)
        if key not in sources or current != last:
            sources[key] = current
        translated = localize_text(language, sources[key])
        if current != translated:
            setter(translated)
        widget._i18n_last[key] = translated

    def _localize_combo(self, combo: QComboBox, language: str) -> None:
        device_names = combo.objectName() in {"inputDeviceCombo", "outputDeviceCombo"}
        sources = getattr(combo, "_i18n_item_sources", {})
        last_values = getattr(combo, "_i18n_item_last", {})
        counts: defaultdict[object, int] = defaultdict(int)
        items: list[tuple[object, str]] = []
        for index in range(combo.count()):
            data = combo.itemData(index)
            base_key = data if data is not None else ("empty", index)
            occurrence = counts[base_key]
            counts[base_key] += 1
            key = (base_key, occurrence)
            current = combo.itemText(index)
            if key not in sources or current != last_values.get(key):
                sources[key] = current
            text = (
                sources[key] if device_names and data is not None
                else _localize_dynamic_choice(language, sources[key])
            )
            items.append((key, text))

        if any(combo.itemText(index) != text for index, (_, text) in enumerate(items)):
            with QSignalBlocker(combo):
                for index, (_, text) in enumerate(items):
                    combo.setItemText(index, text)
        combo._i18n_item_sources = sources
        combo._i18n_item_last = {key: text for key, text in items}

    def _localize_list(self, listing: QListWidget, language: str) -> None:
        sources = getattr(listing, "_i18n_item_sources", {})
        last_values = getattr(listing, "_i18n_item_last", {})
        counts: defaultdict[object, int] = defaultdict(int)
        items: list[tuple[object, str]] = []
        for index in range(listing.count()):
            item = listing.item(index)
            data = item.data(Qt.ItemDataRole.UserRole)
            base_key = data if data is not None else ("empty", index)
            occurrence = counts[base_key]
            counts[base_key] += 1
            key = (base_key, occurrence)
            current = item.text()
            if key not in sources or current != last_values.get(key):
                sources[key] = current
            source_lines = sources[key].split("\n")
            if len(source_lines) > 1:
                translated = "\n".join(
                    [
                        _localize_dynamic_choice(language, source_lines[0]),
                        localize_text(language, "\n".join(source_lines[1:])),
                    ],
                )
            elif source_lines[0].startswith(("预置 /", "暂无音色，")):
                translated = _localize_dynamic_choice(language, source_lines[0])
            else:
                translated = source_lines[0]
            items.append((key, translated))

        if any(listing.item(index).text() != text for index, (_, text) in enumerate(items)):
            with QSignalBlocker(listing):
                for index, (_, text) in enumerate(items):
                    listing.item(index).setText(text)
        listing._i18n_item_sources = sources
        listing._i18n_item_last = {key: text for key, text in items}

    def localize_widget(self, widget: QWidget) -> None:
        if not self._is_in_scope(widget):
            return
        language = self.language()
        if isinstance(widget, self._TEXT_WIDGETS) and not widget.property("i18nPreserveText"):
            shortcut = (
                widget.shortcut()
                if isinstance(widget, QAbstractButton)
                and widget.property("i18nPreserveShortcut")
                else None
            )
            self._localize_attribute(
                widget, "text", widget.text, widget.setText, language,
            )
            if shortcut is not None:
                widget.setShortcut(shortcut)
        if isinstance(widget, QComboBox):
            self._localize_combo(widget, language)
        if isinstance(widget, QListWidget):
            self._localize_list(widget, language)
        if isinstance(widget, (QLineEdit, QPlainTextEdit)):
            self._localize_attribute(
                widget, "placeholder", widget.placeholderText,
                widget.setPlaceholderText, language,
            )
        self._localize_attribute(
            widget, "tooltip", widget.toolTip, widget.setToolTip, language,
        )
        self._localize_attribute(
            widget, "statusTip", widget.statusTip, widget.setStatusTip, language,
        )
        self._localize_attribute(
            widget, "whatsThis", widget.whatsThis, widget.setWhatsThis, language,
        )
        self._localize_attribute(
            widget, "accessibleName", widget.accessibleName, widget.setAccessibleName, language,
        )
        self._localize_attribute(
            widget, "accessibleDescription", widget.accessibleDescription,
            widget.setAccessibleDescription, language,
        )
        if widget.isWindow():
            self._localize_attribute(
                widget, "windowTitle", widget.windowTitle, widget.setWindowTitle, language,
            )

    def localize_subtree(self, root: QWidget) -> None:
        self.localize_widget(root)
        for widget in root.findChildren(QWidget):
            self.localize_widget(widget)

    def localize_tree(self) -> None:
        self.localize_subtree(self.root)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() in (QEvent.Type.Paint, QEvent.Type.Show) and isinstance(watched, QWidget):
            self.localize_widget(watched)
        return False


__all__ = [
    "DEFAULT_LANGUAGE",
    "LanguageFilter",
    "load_language",
    "localize_text",
    "save_language",
    "settings_path",
]
