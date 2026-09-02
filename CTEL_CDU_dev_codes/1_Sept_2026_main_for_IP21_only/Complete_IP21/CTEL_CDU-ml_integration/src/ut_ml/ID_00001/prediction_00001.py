from src.ut_ml.ut_thickness_prediction import UTThicknessPrediction
from ml_module.corrosion_probes.ID_00001.prediction_lstm import ID00001Model

class UTThicknessPrediction00001():
    
    def __init__(self, month: str, year: str, yesterday_date: str, parent = None):
        probe_id = "00001"
        self.ut_thickness_prediction = UTThicknessPrediction(month, year, yesterday_date, probe_id, parent)
        self.ut_thickness_prediction.model = ID00001Model()
        self.ut_thickness_prediction.set_up()