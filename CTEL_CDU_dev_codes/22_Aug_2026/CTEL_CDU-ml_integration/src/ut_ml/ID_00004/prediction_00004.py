from src.ut_ml.ut_thickness_prediction import UTThicknessPrediction
from ml_module.corrosion_probes.ID_00004.prediction_lstm import ID00004Model

class UTThicknessPrediction00004():
    
    def __init__(self, month: str, year: str, yesterday_date: str, parent = None):
        probe_id = "00004"
        self.ut_thickness_prediction = UTThicknessPrediction(month, year, yesterday_date, probe_id, parent)
        self.ut_thickness_prediction.model = ID00004Model()
        self.ut_thickness_prediction.set_up()

   