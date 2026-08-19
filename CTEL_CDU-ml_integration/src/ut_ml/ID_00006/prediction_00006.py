from src.ut_ml.ut_thickness_prediction import UTThicknessPrediction
from ml_module.corrosion_probes.ID_00006.prediction_lstm import ID00006Model

class UTThicknessPrediction00006():
    
    def __init__(self, month: str, year: str, yesterday_date: str, parent = None):
        probe_id = "00006"
        self.ut_thickness_prediction = UTThicknessPrediction(month, year, yesterday_date, probe_id, parent)
        self.ut_thickness_prediction.model = ID00006Model()
        self.ut_thickness_prediction.set_up()
   