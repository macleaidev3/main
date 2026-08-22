from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QGraphicsDropShadowEffect


class Effects:

    @staticmethod
    def card_shadow():

        shadow = QGraphicsDropShadowEffect()

        shadow.setBlurRadius(24)

        shadow.setOffset(0, 4)

        shadow.setColor(QColor(0, 0, 0, 25))

        return shadow
    

