from PyQt6.QtGui import QFont


class Typography:
    # Use a stable system font for Windows-based industrial UI
    FAMILY = "Segoe UI"

    # Font sizes
    PAGE_TITLE = 24
    SECTION_TITLE = 14
    CARD_TITLE = 12
    BODY = 10
    DESCRIPTION = 14
    BUTTON = 16
    KPI_VALUE = 28
    KPI_LABEL = 12
    TABLE_HEADER = 10
    TABLE_BODY = 10
    SIDEBAR = 10
    CAPTION = 9

    # Weights
    LIGHT = QFont.Weight.Light
    NORMAL = QFont.Weight.Normal
    MEDIUM = QFont.Weight.Medium
    SEMIBOLD = QFont.Weight.DemiBold
    BOLD = QFont.Weight.Bold

    @classmethod
    def page_title(cls):
        font = QFont(cls.FAMILY, cls.PAGE_TITLE)
        font.setWeight(cls.BOLD)
        return font

    @classmethod
    def section_title(cls):
        font = QFont(cls.FAMILY, cls.SECTION_TITLE)
        font.setWeight(cls.BOLD)
        return font

    @classmethod
    def card_title(cls):
        font = QFont(cls.FAMILY, cls.CARD_TITLE)
        font.setWeight(cls.SEMIBOLD)
        return font

    @classmethod
    def body(cls):
        font = QFont(cls.FAMILY, cls.BODY)
        font.setWeight(cls.NORMAL)
        return font

    @classmethod
    def description(cls):
        font = QFont(cls.FAMILY, cls.DESCRIPTION)
        font.setWeight(cls.NORMAL)
        return font

    @classmethod
    def button(cls):
        font = QFont(cls.FAMILY, cls.BUTTON)
        font.setWeight(cls.SEMIBOLD)
        return font

    @classmethod
    def kpi_value(cls):
        font = QFont(cls.FAMILY, cls.KPI_VALUE)
        font.setWeight(cls.BOLD)
        return font

    @classmethod
    def kpi_label(cls):
        font = QFont(cls.FAMILY, cls.KPI_LABEL)
        font.setWeight(cls.MEDIUM)
        return font

    @classmethod
    def table_header(cls):
        font = QFont(cls.FAMILY, cls.TABLE_HEADER)
        font.setWeight(cls.BOLD)
        return font

    @classmethod
    def table_body(cls):
        font = QFont(cls.FAMILY, cls.TABLE_BODY)
        font.setWeight(cls.NORMAL)
        return font

    @classmethod
    def sidebar(cls):
        font = QFont(cls.FAMILY, cls.SIDEBAR)
        font.setWeight(cls.SEMIBOLD)
        return font

    @classmethod
    def caption(cls):
        font = QFont(cls.FAMILY, cls.CAPTION)
        font.setWeight(cls.NORMAL)
        return font