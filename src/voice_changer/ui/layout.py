"""Shared surface and heading geometry for the desktop workspaces."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QScrollArea, QVBoxLayout, QWidget

from .theme import SPACING


def new_panel() -> tuple[QFrame, QVBoxLayout]:
    panel = QFrame()
    panel.setObjectName("panel")
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(SPACING["lg"], SPACING["md"], SPACING["lg"], SPACING["md"])
    layout.setSpacing(SPACING["sm"])
    return panel, layout


def page_heading(layout: QVBoxLayout, title: str, description: str) -> None:
    container = QWidget()
    container.setObjectName("pageHeading")
    heading_layout = QVBoxLayout(container)
    heading_layout.setContentsMargins(0, 0, 0, 4)
    heading_layout.setSpacing(4)
    heading = QLabel(title)
    heading.setObjectName("pageTitle")
    heading_layout.addWidget(heading)
    note = QLabel(description)
    note.setWordWrap(True)
    note.setProperty("role", "muted")
    heading_layout.addWidget(note)
    layout.addWidget(container)


def footer_status(layout: QVBoxLayout) -> QVBoxLayout:
    """Keep long runtime messages scrollable without displacing stop controls."""
    scroll = QScrollArea()
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setMinimumSize(0, 0)
    scroll.setAccessibleName("操作状态 / Operation status")
    content = QWidget()
    content.setObjectName("footerStatus")
    status_layout = QVBoxLayout(content)
    status_layout.setContentsMargins(0, 0, 0, 0)
    status_layout.setSpacing(0)
    scroll.setWidget(content)
    layout.addWidget(scroll, 1)
    return status_layout
