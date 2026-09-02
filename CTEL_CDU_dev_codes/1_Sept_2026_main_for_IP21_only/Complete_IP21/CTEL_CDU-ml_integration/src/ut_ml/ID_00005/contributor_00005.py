from src.ut_ml.ut_thickness_contributor import UTThicknessContributor

class UTThicknessContributor00005():
    
    def __init__(self, month: str, year: str, yesterday_date: str,  parent = None):
        probe_id = "00005"
        self.ut_thickness_contributor = UTThicknessContributor(month, year, yesterday_date, probe_id, parent)
        self.ut_thickness_contributor.set_up()
