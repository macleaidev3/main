
from .typography import Typography
from .colors import Colors
from .metrics import Metrics

from src.utils.core_utility_functions import resource_path
class Styles:
    #============================
    @staticmethod
    def text_style():
        return f"""
        QLabel#PageTitle {{
            font-family: "{Typography.FAMILY}";
            font-size: {Typography.PAGE_TITLE}px;
            font-weight: bold;
            color: {Colors.TEXT_PRIMARY};
        }}

        QLabel#SectionTitle {{
            font-family: "{Typography.FAMILY}";
            font-size: {Typography.SECTION_TITLE}px;
            font-weight: bold;
            color: {Colors.TEXT_PRIMARY};
        }}

        QLabel#CardTitle {{
            font-family: "{Typography.FAMILY}";
            font-size: {Typography.CARD_TITLE}px;
            font-weight: 600; /* Semibold */
            color: {Colors.TEXT_PRIMARY};
        }}

        QLabel#BodyText {{
            font-family: "{Typography.FAMILY}";
            font-size: {Typography.BODY}px;
            color: {Colors.TEXT_PRIMARY};
        }}

        QLabel#DescriptionText {{
            font-family: "{Typography.FAMILY}";
            font-size: {Typography.DESCRIPTION}px;
            color: {Colors.TEXT_SECONDARY};
        }}

        QLabel#CaptionText {{
            font-family: "{Typography.FAMILY}";
            font-size: {Typography.CAPTION}px;
            color: {Colors.TEXT_MUTED};
        }}

        QLabel#KPIValue {{
            font-family: "{Typography.FAMILY}";
            font-size: {Typography.KPI_VALUE}px;
            font-weight: bold;
            color: {Colors.TEXT_PRIMARY};
        }}

        QLabel#KPILabel {{
            font-family: "{Typography.FAMILY}";
            font-size: {Typography.KPI_LABEL}px;
            font-weight: 500; /* Medium */
            color: {Colors.TEXT_SECONDARY};
        }}

        QLabel#SplashScreenMessageLabel {{
            background-color: transparent;
        }}

        QLabel[severity="healthy"] {{
            color: {Colors.HEALTHY};
            font-weight: 600; /* Semibold */
        }}

        QLabel[severity="low"] {{
            color: {Colors.INFO};
            font-weight: 600; /* Semibold */
        }}

        QLabel[severity="warning"] {{
            color: {Colors.WARNING};
            font-weight: 600; /* Semibold */
        }}

        QLabel[severity="high"] {{
            color: {Colors.HIGH};
            font-weight: 600; /* Semibold */
        }}

        QLabel[severity="critical"] {{
            color: {Colors.CRITICAL};
            font-weight: bold;
        }}

        QLabel[severity="offline"] {{
            color: {Colors.OFFLINE};
            font-weight: 500; /* Medium */
        }}

        QPushButton {{
            font-family: "{Typography.FAMILY}";
            font-size: {Typography.BUTTON}px;
            font-weight: 600; /* Semibold */
            outline: none;
        }}
    

        QHeaderView::section {{
            font-family: "{Typography.FAMILY}";
            font-size: {Typography.TABLE_HEADER}px;
            font-weight: bold;
        }}

        QTableWidget {{
            font-family: "{Typography.FAMILY}";
            font-size: {Typography.TABLE_BODY}px;
        }}

        QFrame#Sidebar QLabel {{
            font-family: "{Typography.FAMILY}";
            font-size: {Typography.SIDEBAR}px;
        }}
        """
    #============================

    @staticmethod
    def global_style():

        return f"""
        QWidget {{
            background: {Colors.WINDOW_BG};
            color: {Colors.TEXT_PRIMARY};
            font-family: "Segoe UI";
            font-size: 12px;
        }}


        QMainWindow {{
            background: {Colors.WINDOW_BG};
        }}
        """
    
    @staticmethod
    def card_style():

        return f"""
        QWidget#Card
        {{
            background: {Colors.CARD_BG};

            border:  1px solid {Colors.BORDER};

            border-radius: {Metrics.BORDER_RADIUS}px;
        }}

        /*QWidget#Card:hover
         {{
             border: 1px solid {Colors.BORDER_HOVER};
         }}*/
        """

    @staticmethod
    def sidebar_style():

        return f"""
        QWidget#Sidebar
        {{
            background: {Colors.SIDEBAR_BG};
        }}
        """

    @staticmethod
    def navigation_button():

        return f"""
        QPushButton[nav="true"]
        {{
            color: black;

            border:none;

            border-radius:10px;

            padding-left:12px;

            text-align:left;

            min-height:45px;
        }}

        QPushButton[nav="true"]:hover
        {{
            background:#173220;
            color: white;
        }}

        QPushButton[nav="true"]:pressed
        {{
            background:#004d00;

            font-weight:600;
        }}
        """

    @staticmethod
    def nav_tab_button_style():

        return """
        QPushButton[navTab="true"]
        {
            background: transparent;

            border: 1px solid #DCE3EA;

            border-bottom: 2px solid transparent;

            border-top: 2px solid transparent;

            color: #667085;

            padding: 8px 14px;

            font-size: 14px;

            font-weight: 500;
        }

        QPushButton[navTab="true"]:hover
        {   
            background: #ebf9eb;
            color: #184e18;
        }

        QPushButton[navTab="true"]:checked
        {
            color: #1E7A2E;

            border-bottom: 2px solid #1E7A2E;

            font-weight: 505;
        }
        """
    
    @staticmethod
    def month_tab_button_style():

        return """
        QPushButton[monthTab="true"]
        {
            background: transparent;

            border: none;

            color: #667085;

            padding: 6px 12px;

            font-size: 14px;

            font-weight: 500;
        }

        QPushButton[monthTab="true"]:hover
        {
            color: #1E7A2E;
        }

        QPushButton[monthTab="true"]:checked
        {
            color: #111827;

            font-weight: 700;

            border-bottom: 2px solid #1E7A2E;
        }

        QPushButton[monthTab="true"]:pressed
        {
            color: #145221;
        }
        """

    @staticmethod
    def display_button():

        return f"""
        QPushButton[display_button="true"]
        {{
            background:white;

            border:1px solid #e4ede6;

            border-radius:8px;

            padding:8px 14px;

            text-align:left;

            font-size:12px;

            font-weight: 400;

        }}

        QPushButton[display_button="true"]:hover
        {{
            background:{Colors.PRIMARY_LIGHT};
        }}
        """
    
    @staticmethod
    def primary_button():

        return f"""
        QPushButton[variant="primary"]
        {{
            background: #173220;

            color:white;

            border:none;

            border-radius:8px;

            padding: 8px 14px;

            
        }}

        QPushButton[variant="primary"]:hover
        {{
            background:{Colors.PRIMARY_DARK};
        }}

        QPushButton[variant="primary"]:pressed
        {{
            background:#0D3A17;
        }}
        """

    @staticmethod
    def secondary_button():

        return f"""
        QPushButton[variant="secondary"]
        {{
            background:white;

            color:{Colors.PRIMARY};

            border:1px solid {Colors.PRIMARY};

            border-radius:10px;

            padding:10px 18px;
        }}

        QPushButton[variant="secondary"]:hover
        {{
            background:{Colors.PRIMARY_LIGHT};
        }}
        """

    @staticmethod
    def line_edit():

        return f"""
        QLineEdit
        {{
            background:white;

            border:1px solid {Colors.BORDER};

            border-radius:10px;

            padding-left:12px;
        }}

        QLineEdit:focus
        {{
            border:1px solid {Colors.PRIMARY};
        }}
        """

    @staticmethod
    def table_style():

        return f"""
        QTableWidget
        {{
            background:white;

            border:none;

            gridline-color:#E5E7EB;
        }}

        QHeaderView::section
        {{
            background:#F8FAFC;

            border:none;

            padding:10px;

            font-weight:600;
        }}
        """
    
    @staticmethod
    def compact_combobox_style():

        return """
        QComboBox[compact="true"]
        {
            background: white;

            border: 1px solid #DCE3EA;

            border-radius: 6px;

            padding: 2px 6px;

            min-height: 22px;

            font-size: 14px;

            color: #344054;
        }

        QComboBox[compact="true"]:hover
        {
            border: 1px solid #1E7A2E;
        }

        QComboBox[compact="true"]:focus
        {
            border: 1px solid #1E7A2E;
        }

        QComboBox[compact="true"]::drop-down
        {
            border: none;

            width: 18px;
        }

        QComboBox[compact="true"] QAbstractItemView
        {
            background: white;

            border: 1px solid #DCE3EA;

            selection-background-color: #EAF7EE;

            selection-color: #111827;

            outline: none;
        }
        """
    
    @staticmethod
    def checkbox_style():
        unchecked = resource_path("assets/checkbox_unchecked.png")
        checked = resource_path("assets/checkbox_checked.png")
        partial = resource_path("assets/checkbox_partial.png")

        return f"""
        /* Normal QCheckBox */
        QCheckBox
        {{
            color: #344054;
            spacing: 8px;
            font-size: 11px;
        }}

        QCheckBox::indicator
        {{
            width: 16px;
            height: 16px;
        }}

        QCheckBox::indicator:unchecked
        {{
            image: url("{unchecked}");
        }}

        QCheckBox::indicator:checked
        {{
            image: url("{checked}");
        }}

        QCheckBox::indicator:indeterminate
        {{
            image: url("{partial}");
        }}

        /* TreeView checkbox indicators */
        QTreeView::indicator
        {{
            width: 16px;
            height: 16px;
        }}

        QTreeView::indicator:unchecked
        {{
            image: url("{unchecked}");
        }}

        QTreeView::indicator:checked
        {{
            image: url("{checked}");
        }}

        QTreeView::indicator:indeterminate
        {{
            image: url("{partial}");
        }}
        """
    
    @staticmethod
   
    def tree_style():
        closed_arrow = resource_path("assets/chevron_right.png")
        open_arrow = resource_path("assets/chevron_down.png")

        return f"""
        QTreeView
        {{
            background: #FFFFFF;
            border: none;
            outline: 0;
            font-size: 11px;
            color: #344054;
            show-decoration-selected: 0;
            alternate-background-color: #FAFBFC;
        }}

        QTreeView::item
        {{
            padding-top: 5px;
            padding-bottom: 5px;
            padding-left: 4px;
            min-height: 24px;
            border: none;
        }}

        QTreeView::item:hover
        {{
            background: #F5FAF6;
        }}

        QTreeView::item:selected
        {{
            background: transparent;
            color: #111827;
        }}

        QTreeView::item:focus
        {{
            background: transparent;
            outline: none;
        }}

        QTreeView::item:has-children
        {{
            font-weight: 600;
            color: #111827;
        }}

        QTreeView::item:has-children:open
        {{
            background: #F3FBF6;
        }}

        QTreeView::branch
        {{
            background: transparent;
        }}

        QTreeView::branch:has-children
        {{
            width: 12px;
            padding-left: 2px;
        }}

        QTreeView::branch:closed:has-children
        {{
            image: url("{closed_arrow}");
        }}

        QTreeView::branch:open:has-children
        {{
            image: url("{open_arrow}");
        }}
        """ 
    
    @staticmethod
    def calendar_style():
        return """
        QCalendarWidget
        {
            background: #FFFFFF;
            color: #344054;
            border: none;
            border-radius: 12px;
            font-size: 11px;
        }

        QCalendarWidget QWidget#qt_calendar_navigationbar
        {
            background: #F8FAFC;
            border: none;
            border-top-left-radius: 12px;
            border-top-right-radius: 12px;
            min-height: 36px;
        }

        QCalendarWidget QToolButton
        {
            background: transparent;
            color: #111827;
            border: none;
            font-size: 12px;
            font-weight: 600;
            padding: 4px 8px;
        }

        QCalendarWidget QToolButton:hover
        {
            background: #EAF7EE;
            color: #1E7A2E;
            border-radius: 6px;
        }

        QCalendarWidget QMenu
        {
            background: white;
            border: 1px solid #DCE3EA;
        }

        QCalendarWidget QSpinBox
        {
            background: white;
            border: 1px solid #DCE3EA;
            border-radius: 6px;
            padding: 2px 4px;
            min-width: 60px;
            color: #111827;
        }

        QCalendarWidget QAbstractItemView
        {
            selection-background-color: #EAF7EE;
            selection-color: #111827;
            outline: 0;
            background: white;
            alternate-background-color: #FAFBFC;
            gridline-color: #EEF2F6;
        }

        QCalendarWidget QAbstractItemView:enabled
        {
            font-size: 10px;
        }

        QCalendarWidget QAbstractItemView:item
        {
            padding: 4px;
        }

        QCalendarWidget QAbstractItemView:item:hover
        {
            background: #F5FAF6;
        }

        QCalendarWidget QAbstractItemView:item:selected
        {
            background: #EAF7EE;
            color: #111827;
            border-radius: 4px;
        }

        QCalendarWidget QWidget
        {
            alternate-background-color: #FAFBFC;
        }

        QCalendarWidget QAbstractItemView:disabled
        {
            color: #98A2B3;
        }

        QCalendarWidget QLabel
        {
            color: #344054;
        }

        QCalendarWidget QAbstractItemView:enabled:selected
        {
            background: #EAF7EE;
            color: #111827;
        }

        QCalendarWidget QAbstractItemView:enabled:focus
        {
            outline: none;
        }

        QCalendarWidget QAbstractItemView::item
        {
            border: none;
        }

        QCalendarWidget QAbstractItemView::item:enabled
        {
            color: #111827;
        }

        QCalendarWidget QAbstractItemView::item:disabled
        {
            color: #98A2B3;
        }

        QCalendarWidget QAbstractItemView::item:selected
        {
            background: #EAF7EE;
            color: #111827;
            border-radius: 4px;
        }
        """
   
    @staticmethod
    def list_item_style():
        return """
        QWidget#listItemWidget
        {
            background: #F8FAFC;
            border-radius: 3px;
            border: 1px solid #DCE3EA;
        
        }
        """
    
    @classmethod
    def build(cls):

        return "\n".join([
            cls.global_style(),
            cls.card_style(),
            cls.sidebar_style(),
            cls.navigation_button(),
            # cls.normal_button(),
            cls.primary_button(),
            cls.secondary_button(),
            cls.line_edit(),
            cls.table_style(),
            cls.text_style(),
            cls.nav_tab_button_style(),
            cls.display_button(),
            cls.month_tab_button_style(),
            cls.compact_combobox_style(),

            cls.checkbox_style(),
            cls.tree_style(),
            cls.calendar_style(),
            cls.list_item_style(),
            
        ])