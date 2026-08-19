from src.ut_ml.ut_thickness_prediction import UTThicknessPrediction
from ml_module.corrosion_probes.ID_00003.prediction_lstm import ID00003Model

class UTThicknessPrediction00003():
    
    def __init__(self, month: str, year: str, yesterday_date: str, parent = None):
        probe_id = "00003"
        self.ut_thickness_prediction = UTThicknessPrediction(month, year, yesterday_date, probe_id, parent)
        self.ut_thickness_prediction.model = ID00003Model()
        self.ut_thickness_prediction.set_up()
   