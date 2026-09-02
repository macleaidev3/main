
import pandas as pd
from PyQt6 import QtCore
import os
import pandas as pd
from PyQt6 import QtCore
class ExcelLoaderThread(QtCore.QThread):
    finished_success = QtCore.pyqtSignal() 
    finished_limit_error = QtCore.pyqtSignal(str)  
    finished_validation_error = QtCore.pyqtSignal(str)
    finished_error = QtCore.pyqtSignal(str)

    def __init__(self, file_path, table_columns, nrows_limit=750):
        super().__init__()
        self.file_path = file_path
        self.table_columns = table_columns
        self.nrows_limit = nrows_limit
        self.loaded_rows = []

    def run(self):
        try:
            
            # 1. Determine file type based on extension
            _, file_extension = os.path.splitext(self.file_path)
            file_extension = file_extension.lower()

            # 2. Read a maximum of limit + 1 data rows to check the limit instantly
            if file_extension == '.csv':
                df = pd.read_csv(self.file_path, nrows=self.nrows_limit + 1)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(self.file_path, nrows=self.nrows_limit + 1)
            else:
                self.finished_error.emit(f"Unsupported file format: {file_extension}. Please upload a CSV.")
                return

            # Check if it exceeded the limit
            if len(df) > self.nrows_limit:
                self.finished_limit_error.emit(
                    f"The selected file contains more than {self.nrows_limit} rows.\n\n"
                    f"Support for large datasets is limited to {self.nrows_limit} rows."
                )
                return

            df.columns = [str(col).strip() for col in df.columns]
            expected = [str(c).strip() for c in self.table_columns]

            missing = [c for c in expected if c not in df.columns]
            extra = [c for c in df.columns if c not in expected]

            if missing or extra:
                message = ""
                if missing:
                    message += f"Missing required columns:\n  • " + "\n  • ".join(missing) + "\n\n"
                if extra:
                    message += f"Unexpected extra columns:\n  • " + "\n  • ".join(extra) + "\n\n"
                
                self.finished_validation_error.emit(message)
                return

            # If we pass the checks, df already contains the full dataset
            self.loaded_rows = [tuple(row) for row in df[expected].itertuples(index=False, name=None)]
            
            self.finished_success.emit()
            print(f"Loaded {len(self.loaded_rows)} rows successfully.")

        except Exception as e:
            self.finished_error.emit(str(e))
