"""Shared, tokenized Qt styling for the desktop window and silent prototype."""

from PySide6.QtCore import QCoreApplication, QEvent

COLORS = {
    "canvas": "#F0EEF7",
    "surface": "#FEFEFF",
    "surfaceRaised": "#F7F5FC",
    "navigation": "#FAF8FE",
    "text": "#302D46",
    "muted": "#605C77",
    "line": "#D8D2E8",
    "active": "#624E96",
    "activeInk": "#FFFFFF",
    "activeSurface": "#EBE6F7",
    "danger": "#A12E46",
    "dangerSurface": "#FBEAEC",
    "warning": "#835025",
    "warningSurface": "#FFF3E7",
    "disabled": "#EFEAF5",
    "meterTrack": "#E5E0EF",
    "focus": "#493375",
}

TYPE = {
    "ui": '"Noto Sans CJK SC", "Noto Sans", sans-serif',
    "numeric": '"DejaVu Sans Mono", monospace',
}

SPACING = {
    "none": 0,
    "xs": 4,
    "sm": 8,
    "md": 12,
    "lg": 16,
}

SIZES = {
    "body": "10pt",
    "small": "9pt",
    "title": "20pt",
    "section": "12pt",
    "currentValue": "15pt",
    "panelRadius": 7,
    "inputRadius": 4,
    "buttonRadius": 5,
    "meterRadius": 3,
    "sliderTrackRadius": 2,
    "sliderHandleRadius": 8,
    "scrollbarHandleRadius": 4,
    "checkboxIndicatorRadius": 3,
    "borderWidth": 1,
    "focusBorderWidth": 2,
    "stateBorderWidth": 3,
    "inputPadding": "3px 7px",
    "buttonPadding": "3px 10px",
    "toolPadding": "3px 7px",
    "statePadding": "5px 8px",
    "checkPadding": 2,
    "checkSpacing": 7,
    "dropdownWidth": 22,
    "checkboxIndicatorSize": 17,
    "buttonHeight": 32,
    "meterHeight": 18,
    "sliderTrackHeight": 5,
    "sliderHandleWidth": 16,
    "parameterValueWidth": 108,
    "sliderHandleMargin": "-6px 0",
    "scrollbarWidth": 10,
    "scrollbarHandleMinHeight": 24,
    "footerMinHeight": 96,
    "footerHeight": 120,
    "compactFooterHeight": 100,
    "sidebarWidth": 205,
    "sidebarBreakpoint": 900,
}


def stylesheet() -> str:
    c = COLORS
    t = TYPE
    s = SIZES
    return f"""
    QWidget {{
        color: {c['text']};
        font-family: {t['ui']};
        font-size: {s['body']};
    }}
    QMainWindow, QScrollArea, QWidget#appRoot, QWidget#canvas {{
        background-color: {c['canvas']};
    }}
    QScrollArea {{ border: 0; }}
    QLabel {{ background-color: transparent; }}
    QWidget#workspaceHeader, QWidget#workspaceNavigation {{
        background-color: {c['navigation']};
    }}
    QWidget#workspaceHeader {{
        border-bottom: 1px solid {c['line']};
    }}
    QWidget#workspaceNavigation {{
        border-right: 1px solid {c['line']};
        border-bottom: 1px solid {c['line']};
    }}
    QWidget#fixedFooter {{
        background-color: {c['surface']};
        border-top: {s['borderWidth']}px solid {c['line']};
    }}
    QWidget#footerStatus {{
        background-color: {c['surface']};
    }}
    QFrame#panel, QWidget[role="section"] {{
        background-color: {c['surface']};
        border: {s['borderWidth']}px solid {c['line']};
        border-radius: {s['panelRadius']}px;
    }}
    QLabel#appTitle {{
        color: {c['text']};
        font-size: {s['section']};
        font-weight: 700;
    }}
    QLabel#pageTitle {{
        font-size: {s['title']};
        font-weight: 700;
    }}
    QDialog {{
        background-color: {c['canvas']};
    }}
    QListWidget {{
        background-color: {c['surface']};
        border: 1px solid {c['line']};
        border-radius: {s['panelRadius']}px;
        padding: 8px;
    }}
    QListWidget::item {{
        padding: 10px 8px;
        border-bottom: 1px solid {c['line']};
    }}
    QListWidget::item:selected {{
        background-color: {c['activeSurface']};
        color: {c['active']};
    }}
    QLabel#sectionTitle, QLabel#heading, QLabel[role="sectionHeading"] {{
        color: {c['text']};
        font-size: {s['section']};
        font-weight: 650;
    }}
    QLabel[role="muted"], QLabel#muted {{
        color: {c['muted']};
        font-size: {s['small']};
    }}
    QLabel#current {{
        color: {c['active']};
        font-size: {s['currentValue']};
        font-weight: 700;
    }}
    QLabel[state="active"] {{
        color: {c['active']};
        background-color: {c['activeSurface']};
        border-left: {s['stateBorderWidth']}px solid {c['active']};
        padding: {s['statePadding']};
        font-weight: 650;
    }}
    QLabel[state="recording"], QLabel[state="fault"], QLabel#warning {{
        color: {c['danger']};
        background-color: {c['dangerSurface']};
        border-left: {s['stateBorderWidth']}px solid {c['danger']};
        padding: {s['statePadding']};
        font-weight: 650;
    }}
    QLabel[state="warning"] {{
        color: {c['warning']};
        background-color: {c['warningSurface']};
        border-left: {s['stateBorderWidth']}px solid {c['warning']};
        padding: {s['statePadding']};
        font-weight: 650;
    }}
    QLineEdit, QComboBox, QDoubleSpinBox, QPlainTextEdit {{
        background-color: {c['surfaceRaised']};
        color: {c['text']};
        border: {s['borderWidth']}px solid {c['line']};
        border-radius: {s['inputRadius']}px;
        padding: {s['inputPadding']};
        selection-background-color: {c['active']};
        selection-color: {c['activeInk']};
        min-height: {s['buttonHeight']}px;
    }}
    QComboBox::drop-down {{ border: 0; width: {s['dropdownWidth']}px; }}
    QDoubleSpinBox[role="parameterValue"] {{
        min-height: 24px;
        padding: 2px 7px;
    }}
    QPushButton {{
        background-color: {c['surface']};
        color: {c['text']};
        border: {s['borderWidth']}px solid {c['line']};
        border-radius: {s['buttonRadius']}px;
        padding: {s['buttonPadding']};
        min-height: {s['buttonHeight']}px;
    }}
    QPushButton[role="primary"] {{
        background-color: {c['active']};
        color: {c['activeInk']};
        border-color: {c['active']};
        font-weight: 650;
    }}
    QPushButton[role="navigation"] {{
        background-color: transparent;
        border-color: transparent;
        color: {c['muted']};
        text-align: left;
        font-weight: 650;
    }}
    QPushButton[role="navigation"]:checked {{
        background-color: {c['activeSurface']};
        color: {c['active']};
    }}
    QPushButton[role="preset"]:checked, QPushButton[state="active"] {{
        background-color: {c['active']};
        color: {c['activeInk']};
        border-color: {c['active']};
        font-weight: 700;
    }}
    QPushButton[state="recording"], QPushButton[state="fault"] {{
        background-color: {c['dangerSurface']};
        color: {c['danger']};
        border-color: {c['danger']};
        font-weight: 700;
    }}
    QToolButton {{
        background-color: transparent;
        color: {c['text']};
        border: {s['borderWidth']}px solid transparent;
        border-radius: {s['inputRadius']}px;
        padding: {s['toolPadding']};
        min-height: {s['buttonHeight']}px;
    }}
    QToolButton:hover:enabled {{
        background-color: {c['surfaceRaised']};
    }}
    QPushButton:disabled, QComboBox:disabled, QLineEdit:disabled,
    QDoubleSpinBox:disabled, QToolButton:disabled {{
        background-color: {c['disabled']};
        color: {c['muted']};
    }}
    QPushButton:hover:enabled, QComboBox:hover:enabled,
    QLineEdit:hover:enabled, QDoubleSpinBox:hover:enabled,
    QToolButton:hover:enabled {{
        border-color: {c['active']};
    }}
    QComboBox QAbstractItemView {{
        background-color: {c['surface']};
        color: {c['text']};
        selection-background-color: {c['activeSurface']};
        selection-color: {c['text']};
        border: {s['borderWidth']}px solid {c['line']};
    }}
    QPushButton:focus, QComboBox:focus, QLineEdit:focus,
    QDoubleSpinBox:focus, QPlainTextEdit:focus,
    QToolButton:focus {{
        border: {s['focusBorderWidth']}px solid {c['focus']};
    }}
    QCheckBox {{
        spacing: {s['checkSpacing']}px;
        border: {s['focusBorderWidth']}px solid transparent;
        border-radius: {s['inputRadius']}px;
        padding: {s['checkPadding']}px;
    }}
    QCheckBox::indicator {{
        width: {s['checkboxIndicatorSize']}px;
        height: {s['checkboxIndicatorSize']}px;
        background-color: {c['surface']};
        border: {s['borderWidth']}px solid {c['line']};
        border-radius: {s['checkboxIndicatorRadius']}px;
    }}
    QCheckBox::indicator:checked {{
        background-color: {c['active']};
        border-color: {c['active']};
    }}
    QCheckBox:focus {{
        border-color: {c['focus']};
    }}
    QProgressBar {{
        background-color: {c['meterTrack']};
        color: {c['text']};
        border: {s['borderWidth']}px solid {c['line']};
        border-radius: {s['meterRadius']}px;
        min-height: {s['meterHeight']}px;
        text-align: center;
        font-family: {t['numeric']};
        font-size: {s['small']};
    }}
    QProgressBar::chunk {{ background-color: {c['active']}; }}
    QSlider {{
        border: none;
        background-color: transparent;
        min-height: {s['buttonHeight']}px;
    }}
    QSlider::groove:horizontal {{
        height: {s['sliderTrackHeight']}px;
        background-color: {c['meterTrack']};
        border-radius: {s['sliderTrackRadius']}px;
    }}
    QSlider::sub-page:horizontal {{
        background-color: {c['active']};
        border-radius: {s['sliderTrackRadius']}px;
    }}
    QSlider::handle:horizontal {{
        width: {s['sliderHandleWidth']}px;
        margin: {s['sliderHandleMargin']};
        background-color: {c['active']};
        border: {s['focusBorderWidth']}px solid {c['surface']};
        border-radius: {s['sliderHandleRadius']}px;
    }}
    QSlider::handle:horizontal:focus {{
        border: {s['focusBorderWidth']}px solid {c['focus']};
    }}
    QScrollBar:vertical {{
        width: {s['scrollbarWidth']}px;
        background-color: {c['canvas']};
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        min-height: {s['scrollbarHandleMinHeight']}px;
        background-color: {c['line']};
        border-radius: {s['scrollbarHandleRadius']}px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    """


def set_visual_state(widget, state: str) -> None:
    if widget.property("state") == state:
        return
    widget.setProperty("state", state)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    QCoreApplication.sendEvent(widget, QEvent(QEvent.Type.StyleChange))
    widget.updateGeometry()
    widget.update()
