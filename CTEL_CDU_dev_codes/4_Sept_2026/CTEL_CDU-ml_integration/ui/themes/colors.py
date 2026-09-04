from dataclasses import dataclass


@dataclass(frozen=True)
class Colors:

    # ======================
    # Brand
    # ======================

    PRIMARY = "#04D627"
    PRIMARY_DARK = "#145221"
    PRIMARY_LIGHT = "#EAF7EE"

    # ======================
    # Backgrounds
    # ======================

    # WINDOW_BG = "#F4F7F9"
    WINDOW_BG = "#FFFFFF"

    CARD_BG = "#FFFFFF"

    SIDEBAR_BG = "#0D1B12"

    # ======================
    # Borders
    # ======================

    BORDER = "#DCE3EA"

    BORDER_HOVER = "#B8C5D1"

    # ======================
    # Text
    # ======================

    TEXT_PRIMARY = "#111827"

    TEXT_SECONDARY = "#667085"

    TEXT_LIGHT = "#FFFFFF"

    # ======================
    # Alarm Colors
    # ======================

    CRITICAL = "#E53935"

    WARNING = "#FB8C00"

    INFO = "#1E88E5"

    HEALTHY = "#2EAD4A"

    UNKNOWN = "#6B7280"

    LOW = "#1E88E5"

    # ======================
    # Charts
    # ======================

    TREND_RED = "#FF4D4F"

    TREND_GREEN = "#2ECC71"

    TREND_BLUE = "#3B82F6"

    TREND_PURPLE = "#8B5CF6"

    # fonts colors
    HIGH = "#EF6C00"
    TEXT_PRIMARY = "#111827"
    TEXT_SECONDARY = "#6B7280"
    TEXT_MUTED = "#9CA3AF"
    TEXT_LIGHT = "#FFFFFF"
    OFFLINE = "#6B7280"
    