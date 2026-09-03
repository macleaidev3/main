### Edited by Anurag

from PyQt6 import QtCore
from multiprocessing import Process, Queue
import itertools
import traceback
import datetime
import logging


# ============================================================
# Application logger
# ============================================================

logger = logging.getLogger("SentinelApp")


# ============================================================
# Worker process entry point
# ============================================================      

def worker_loop(task_q: Queue, result_q: Queue):
    """
    Persistent ML worker process.

    Responsibilities:
        - Receive manual/automatic jobs.
        - Run contributor.
        - Collect contributor combined_flag.
        - Run prediction.
        - Return flags to the Qt process.
    """

    logger.info("[Worker] Started ML worker process.")

    while True:

        job = task_q.get()

        # ----------------------------------------------------
        # Shutdown
        # ----------------------------------------------------

        if job is None:
            logger.info(
                "[Worker] Shutdown signal received. "
                "Terminating worker loop."
            )
            break

        job_id = job["job_id"]
        job_type = job["type"]
        instruments = job["instruments"]
        dates = job["dates"]

        try:

            logger.debug(
                "[Worker] Processing job_id=%s, type=%s",
                job_id,
                job_type
            )

            # ------------------------------------------------
            # This dictionary contains the ACTUAL flag message.
            #
            # Structure:
            #
            # {
            #     "00001": {
            #         "15/08/2025": "actual flag message"
            #     }
            # }
            # ------------------------------------------------

            flagged_dates = {}

            # ------------------------------------------------
            # Automatic job
            # ------------------------------------------------

            if job_type == "auto":

                flagged_dates = _run_automatic_job(dates)

            # ------------------------------------------------
            # Manual job
            # ------------------------------------------------

            elif job_type == "manual":

                flagged_dates = _run_manual_job(
                    instruments,
                    dates
                )

            else:

                raise ValueError(
                    f"Unknown job type: {job_type}"
                )

            # ------------------------------------------------
            # Send result back to Qt process
            # ------------------------------------------------

            result_q.put(
                {
                    "job_id": job_id,
                    "type": job_type,
                    "instruments": instruments,
                    "status": "success",

                    # IMPORTANT:
                    # Send the actual flag messages.
                    "flagged_dates": flagged_dates,
                }
            )

            logger.info(
                "[Worker] Job %s completed successfully. "
                "Flagged dates: %s",
                job_id,
                flagged_dates
            )

        except Exception as e:

            logger.error(
                "[Worker] Job %s failed: %s",
                job_id,
                str(e)
            )

            result_q.put(
                {
                    "job_id": job_id,
                    "type": job_type,
                    "instruments": instruments,
                    "status": "failed",
                    "error": traceback.format_exc(),
                }
            )

    logger.info("[Worker] Exiting worker process.")


# ============================================================
# Automatic ML job
# ============================================================

def _run_automatic_job(dates):
    """
    Automatic job.

    Runs ML for all automatic UT instruments.

    Returns
    -------
    dict
        Structure:

        {
            "00001": {
                "15/08/2025": "flag message"
            },
            "00003": {
                "15/08/2025": "flag message"
            }
        }
    """

    from src.utils.core_utility_functions import month_short_name
    from src.crude_blend.updated_calculated_blend_properties import (
        BlendPropertiesCalculation
    )

    flagged_dates = {}

    yesterday_date = dates[0]

    logger.debug(
        "[Worker] Starting AUTOMATIC job for date: %s",
        yesterday_date
    )

    # --------------------------------------------------------
    # Update calculated blend properties
    # --------------------------------------------------------

    BlendPropertiesCalculation().update_blend_properties(
        yesterday_date
    )

    day, month_i, year = map(
        int,
        yesterday_date.split("/")
    )

    month = month_short_name()[month_i - 1]

    # ========================================================
    # ID 00001
    # ========================================================

    logger.debug(
        "[Worker] Running ML predictions for ID 00001"
    )

    from src.ut_ml.ID_00001.contributor_00001 import (
        UTThicknessContributor00001
    )

    from src.ut_ml.ID_00001.prediction_00001 import (
        UTThicknessPrediction00001
    )

    contributor_instance = UTThicknessContributor00001(
        month=month,
        year=year,
        yesterday_date=yesterday_date
    )

    _collect_combined_flag(
        flagged_dates,
        "00001",
        yesterday_date,
        contributor_instance
    )

    UTThicknessPrediction00001(
        month=month,
        year=year,
        yesterday_date=yesterday_date
    )

    # ========================================================
    # ID 00003
    # ========================================================

    logger.debug(
        "[Worker] Running ML predictions for ID 00003"
    )

    from src.ut_ml.ID_00003.contributor_00003 import (
        UTThicknessContributor00003
    )

    from src.ut_ml.ID_00003.prediction_00003 import (
        UTThicknessPrediction00003
    )

    contributor_instance = UTThicknessContributor00003(
        month=month,
        year=year,
        yesterday_date=yesterday_date
    )

    _collect_combined_flag(
        flagged_dates,
        "00003",
        yesterday_date,
        contributor_instance
    )

    UTThicknessPrediction00003(
        month=month,
        year=year,
        yesterday_date=yesterday_date
    )

    # ========================================================
    # ID 00004
    # ========================================================

    logger.debug(
        "[Worker] Running ML predictions for ID 00004"
    )

    from src.ut_ml.ID_00004.contributor_00004 import (
        UTThicknessContributor00004
    )

    from src.ut_ml.ID_00004.prediction_00004 import (
        UTThicknessPrediction00004
    )

    contributor_instance = UTThicknessContributor00004(
        month=month,
        year=year,
        yesterday_date=yesterday_date
    )

    _collect_combined_flag(
        flagged_dates,
        "00004",
        yesterday_date,
        contributor_instance
    )

    UTThicknessPrediction00004(
        month=month,
        year=year,
        yesterday_date=yesterday_date
    )

    # ========================================================
    # ID 00005
    # ========================================================

    logger.debug(
        "[Worker] Running ML predictions for ID 00005"
    )

    from src.ut_ml.ID_00005.contributor_00005 import (
        UTThicknessContributor00005
    )

    from src.ut_ml.ID_00005.prediction_00005 import (
        UTThicknessPrediction00005
    )

    contributor_instance = UTThicknessContributor00005(
        month=month,
        year=year,
        yesterday_date=yesterday_date
    )

    _collect_combined_flag(
        flagged_dates,
        "00005",
        yesterday_date,
        contributor_instance
    )

    UTThicknessPrediction00005(
        month=month,
        year=year,
        yesterday_date=yesterday_date
    )

    # ========================================================
    # ID 00006
    # ========================================================

    logger.debug(
        "[Worker] Running ML predictions for ID 00006"
    )

    from src.ut_ml.ID_00006.contributor_00006 import (
        UTThicknessContributor00006
    )

    from src.ut_ml.ID_00006.prediction_00006 import (
        UTThicknessPrediction00006
    )

    contributor_instance = UTThicknessContributor00006(
        month=month,
        year=year,
        yesterday_date=yesterday_date
    )

    _collect_combined_flag(
        flagged_dates,
        "00006",
        yesterday_date,
        contributor_instance
    )

    UTThicknessPrediction00006(
        month=month,
        year=year,
        yesterday_date=yesterday_date
    )

    # ========================================================
    # ID 00029
    # ========================================================

    logger.debug(
        "[Worker] Running ML predictions for ID 00029"
    )

    from src.ut_ml.ID_00029.contributor_00029 import (
        UTThicknessContributor00029
    )

    from src.ut_ml.ID_00029.prediction_00029 import (
        UTThicknessPrediction00029
    )

    contributor_instance = UTThicknessContributor00029(
        month=month,
        year=year,
        yesterday_date=yesterday_date
    )

    _collect_combined_flag(
        flagged_dates,
        "00029",
        yesterday_date,
        contributor_instance
    )

    UTThicknessPrediction00029(
        month=month,
        year=year,
        yesterday_date=yesterday_date
    )

    # ========================================================
    # ID 00030
    # ========================================================

    logger.debug(
        "[Worker] Running ML predictions for ID 00030"
    )

    from src.ut_ml.ID_00030.contributor_00030 import (
        UTThicknessContributor00030
    )

    from src.ut_ml.ID_00030.prediction_00030 import (
        UTThicknessPrediction00030
    )

    contributor_instance = UTThicknessContributor00030(
        month=month,
        year=year,
        yesterday_date=yesterday_date
    )

    _collect_combined_flag(
        flagged_dates,
        "00030",
        yesterday_date,
        contributor_instance
    )

    UTThicknessPrediction00030(
        month=month,
        year=year,
        yesterday_date=yesterday_date
    )

    # --------------------------------------------------------
    # Return all collected flags
    # --------------------------------------------------------

    logger.info(
        "[Worker] AUTOMATIC FINAL flagged_dates = %s",
        flagged_dates
    )

    return flagged_dates


# ============================================================
# Flag collection helper
# ============================================================

def _collect_combined_flag(
    flagged_dates,
    instrument,
    date,
    contributor_instance
):
    """
    Collect the combined flag generated by the contributor.

    The contributor is responsible for creating the actual
    IP21 + Lab Report flag.

    Example returned flag:

    15/08/2025 IP21 data was not available, so Sentinal has
    averaged the data of the last 30 days to predict the
    Cr/Thickness

    OR:

    15/08/2025 Lab Report data was not available in section
    "AD Stage-1", so Sentinal has averaged the data of the
    last 30 days to predict the Cr/Thickness

    OR both messages together.
    """

    combined_flag = getattr(
        contributor_instance,
        "combined_flag",
        None
    )

    if combined_flag:

        instrument = str(instrument)
        date = str(date).strip()

        flagged_dates.setdefault(
            instrument,
            {}
        )[date] = combined_flag

        logger.warning(
            "[Worker] FLAGGED | Instrument=%s | Date=%s | %s",
            instrument,
            date,
            combined_flag
        )

    else:

        logger.info(
            "[Worker] No combined flag | "
            "Instrument=%s | Date=%s",
            instrument,
            date
        )


# ============================================================
# Manual ML job
# ============================================================

def _run_manual_job(instruments, dates):
    """
    Manual job.

    Runs ML only for selected instruments and dates.

    Returns
    -------
    dict
        Actual combined flag messages generated by contributors.
    """

    from src.crude_blend.updated_calculated_blend_properties import (
        BlendPropertiesCalculation
    )

    from src.utils.core_utility_functions import (
        month_short_name
    )

    # --------------------------------------------------------
    # Existing database manager used by the existing
    # prediction-input checking logic.
    # --------------------------------------------------------

    from src.server_manager.operation_manager import (
        DatabaseManager
    )

    _blend_properties = BlendPropertiesCalculation()

    db_manager = DatabaseManager()

    db_name = "SentinelDB"

    # ========================================================
    # IMPORTANT
    #
    # Do NOT use:
    #
    # flagged_dates = {
    #     str(instrument): []
    #     for instrument in instruments
    # }
    #
    # because we now need the actual flag message.
    # ========================================================

    flagged_dates = {}

    logger.info(
        "[Worker] Starting MANUAL job for instruments: %s",
        instruments
    )

    # ========================================================
    # Existing prediction-input checking function
    #
    # This is retained.
    # It is NOT the Lab Report recovery logic.
    # ========================================================

    def check_missing_prediction_data(
        instrument,
        date
    ):
        """
        Check whether the existing UT prediction inputs
        are missing for the selected prediction date.

        This existing check is retained and does not modify
        the Lab Report recovery logic.
        """

        required_properties = [
            "Density(g/ml)",
            "API",
            "Sulphur%"
        ]

        day, month_i, year = map(
            int,
            date.split("/")
        )

        month = month_short_name()[month_i - 1]

        contributor_table = (
            f"ut_{instrument}_{year}_{month}_contributor"
        )

        missing_properties = []

        for prop in required_properties:

            try:

                value = db_manager.get_cell_value(
                    db_name,
                    contributor_table,
                    prop,
                    "Date",
                    date
                )

                if (
                    value is None
                    or (
                        isinstance(value, str)
                        and value.strip().lower()
                        in {
                            "",
                            "nan",
                            "none",
                            "null",
                            "na",
                            "n/a",
                        }
                    )
                    or (
                        not isinstance(value, str)
                        and value != value
                    )
                ):
                    missing_properties.append(prop)

            except Exception:

                logger.exception(
                    "[Worker] Failed to check prediction "
                    "input '%s' for instrument %s on %s.",
                    prop,
                    instrument,
                    date
                )

                missing_properties.append(prop)

        if missing_properties:

            logger.warning(
                "[Worker] Missing prediction data detected. "
                "Instrument: %s | Date: %s | Missing: %s",
                instrument,
                date,
                missing_properties
            )

            return True

        return False

    # ========================================================
    # Update blend properties for all selected dates
    # ========================================================

    for date in dates:

        _blend_properties.update_blend_properties(
            date
        )

    # ========================================================
    # Process each instrument
    # ========================================================

    for instrument in instruments:

        _contributor = None
        _calculator = None

        # ====================================================
        # ID 00001
        # ====================================================

        if instrument == "00001":

            from src.ut_ml.ID_00001.contributor_00001 import (
                UTThicknessContributor00001
            )

            from src.ut_ml.ID_00001.prediction_00001 import (
                UTThicknessPrediction00001
            )

            _contributor = UTThicknessContributor00001
            _calculator = UTThicknessPrediction00001

        # ====================================================
        # ID 00003
        # ====================================================

        elif instrument == "00003":

            from src.ut_ml.ID_00003.contributor_00003 import (
                UTThicknessContributor00003
            )

            from src.ut_ml.ID_00003.prediction_00003 import (
                UTThicknessPrediction00003
            )

            _contributor = UTThicknessContributor00003
            _calculator = UTThicknessPrediction00003

        # ====================================================
        # ID 00004
        # ====================================================

        elif instrument == "00004":

            from src.ut_ml.ID_00004.contributor_00004 import (
                UTThicknessContributor00004
            )

            from src.ut_ml.ID_00004.prediction_00004 import (
                UTThicknessPrediction00004
            )

            _contributor = UTThicknessContributor00004
            _calculator = UTThicknessPrediction00004

        # ====================================================
        # ID 00005
        # ====================================================

        elif instrument == "00005":

            from src.ut_ml.ID_00005.contributor_00005 import (
                UTThicknessContributor00005
            )

            from src.ut_ml.ID_00005.prediction_00005 import (
                UTThicknessPrediction00005
            )

            _contributor = UTThicknessContributor00005
            _calculator = UTThicknessPrediction00005

        # ====================================================
        # ID 00006
        # ====================================================

        elif instrument == "00006":

            from src.ut_ml.ID_00006.contributor_00006 import (
                UTThicknessContributor00006
            )

            from src.ut_ml.ID_00006.prediction_00006 import (
                UTThicknessPrediction00006
            )

            _contributor = UTThicknessContributor00006
            _calculator = UTThicknessPrediction00006

        # ====================================================
        # ID 00029
        # ====================================================

        elif instrument == "00029":

            from src.ut_ml.ID_00029.contributor_00029 import (
                UTThicknessContributor00029
            )

            from src.ut_ml.ID_00029.prediction_00029 import (
                UTThicknessPrediction00029
            )

            _contributor = UTThicknessContributor00029
            _calculator = UTThicknessPrediction00029

        # ====================================================
        # ID 00030
        # ====================================================

        elif instrument == "00030":

            from src.ut_ml.ID_00030.contributor_00030 import (
                UTThicknessContributor00030
            )

            from src.ut_ml.ID_00030.prediction_00030 import (
                UTThicknessPrediction00030
            )

            _contributor = UTThicknessContributor00030
            _calculator = UTThicknessPrediction00030

        # ====================================================
        # IC-V-112
        # ====================================================

        elif instrument == "IC-V-112":

            from src.ml_instrument.ICV112.icv112_contributor import (
                CRContributorICV112
            )

            from src.ml_instrument.ICV112.icv112_prediction import (
                CRPredictionICV112
            )

            _contributor = CRContributorICV112
            _calculator = CRPredictionICV112

        # ====================================================
        # IC-V-113
        # ====================================================

        elif instrument == "IC-V-113":

            from src.ml_instrument.ICV113.icv113_contributor import (
                CRContributorICV113
            )

            from src.ml_instrument.ICV113.icv113_prediction import (
                CRPredictionICV113
            )

            _contributor = CRContributorICV113
            _calculator = CRPredictionICV113

        # ====================================================
        # IC-E-126
        # ====================================================

        elif instrument == "IC-E-126":

            from src.ml_instrument.ICE126.ice126_contributor import (
                CRContributorICE126
            )

            from src.ml_instrument.ICE126.ice126_prediction import (
                CRPredictionICE126
            )

            _contributor = CRContributorICE126
            _calculator = CRPredictionICE126

        # ====================================================
        # IC-E-102 trial
        # ====================================================

        elif instrument == "IC-E-102 trial":

            from src.ml_instrument.ICE102.ice102_contributor import (
                CRContributorICE102
            )

            _contributor = CRContributorICE102
            _calculator = None

        # ====================================================
        # IC-E-161 A~H
        # ====================================================

        elif instrument == "IC-E-161 A~H":

            from src.ml_instrument.ICE161.ice161_contributor import (
                CRContributorICE161
            )

            from src.ml_instrument.ICE161.ice161_prediction import (
                CRPredictionICE161
            )

            _contributor = CRContributorICE161
            _calculator = CRPredictionICE161

        # ====================================================
        # IC-E-162 A~P
        # ====================================================

        elif instrument == "IC-E-162 A~P":

            from src.ml_instrument.ICE162.ice162_contributor import (
                CRContributorICE162
            )

            from src.ml_instrument.ICE162.ice162_prediction import (
                CRPredictionICE162
            )

            _contributor = CRContributorICE162
            _calculator = CRPredictionICE162

        # ====================================================
        # Pipeline:
        # IC-E-102 -> IC-E-161 A~H
        # ====================================================

        elif instrument == "Pipeline(IC-E-102 to IC-E-161 A~H)":

            from src.ml_pipelines.ICE102_ICE161.ice102_161_contributor import (
                CRContributorICE102toICE161
            )

            from src.ml_pipelines.ICE102_ICE161.ice102_161_prediction import (
                CRPredictionICE101TO161
            )

            _contributor = CRContributorICE102toICE161
            _calculator = CRPredictionICE101TO161

        # ====================================================
        # Pipeline:
        # IC-V-101 -> IC-E-102
        # ====================================================

        elif instrument == "Pipeline(IC-V-101 to IC-E-102)":

            from src.ml_pipelines.ICV101_ICE102.icv101_102_contributor import (
                CRContributorICV101toICE102
            )

            from src.ml_pipelines.ICV101_ICE102.icv101_102_prediction import (
                CRPredictionICV101TO102
            )

            _contributor = CRContributorICV101toICE102
            _calculator = CRPredictionICV101TO102

        # ====================================================
        # Pipeline:
        # IC-V-112 -> IC-E-162 A~P
        # ====================================================

        elif instrument == "Pipeline(IC-V-112 to IC-E-162 A~P)":

            from src.ml_pipelines.ICEV112_ICE162.icv112_162_contributor import (
                CRContributorICV112toICE162
            )

            from src.ml_pipelines.ICEV112_ICE162.icv112_162_prediction import (
                CRPredictionICV112TO162
            )

            _contributor = CRContributorICV112toICE162
            _calculator = CRPredictionICV112TO162

        # ====================================================
        # Pipeline:
        # IC-E-126 A~D -> IC-V-113
        # ====================================================

        elif instrument == "Pipeline(IC-E-126 A~D to IC-V-113)":

            from src.ml_pipelines.ICE126_ICV113.ice126_113_contributor import (
                CRContributorICE126toICV113
            )

            from src.ml_pipelines.ICE126_ICV113.ice126_113_prediction import (
                CRPredictionICE126TO113
            )

            _contributor = CRContributorICE126toICV113
            _calculator = CRPredictionICE126TO113

        # ====================================================
        # Pipeline:
        # IC-E-161 A~H -> IC-V-112
        # ====================================================

        elif instrument == "Pipeline(IC-E-161 A~H to IC-V-112)":

            from src.ml_pipelines.ICE161_ICV112.ice161_112_contributor import (
                CRContributorICE161toICV112
            )

            from src.ml_pipelines.ICE161_ICV112.ice161_112_prediction import (
                CRPredictionICE161TO112
            )

            _contributor = CRContributorICE161toICV112
            _calculator = CRPredictionICE161TO112

        # ====================================================
        # Pipeline:
        # IC-E-162 A~P -> IC-E-126 A~D
        # ====================================================

        elif instrument == "Pipeline(IC-E-162 A~P to IC-E-126 A~D)":

            from src.ml_pipelines.ICE162_ICE126.ice162_126_contributor import (
                CRContributorICE162toICE126
            )

            from src.ml_pipelines.ICE162_ICE126.ice162_126_prediction import (
                CRPredictionICE162TO126
            )

            _contributor = CRContributorICE162toICE126
            _calculator = CRPredictionICE162TO126

        else:

            logger.warning(
                "[Worker] No contributor configured for instrument: %s",
                instrument
            )

            continue

        # ====================================================
        # Process selected dates
        # ====================================================

        for date in dates:

            logger.debug(
                "[Worker] Executing MANUAL job component: "
                "Instrument=%s | Date=%s",
                instrument,
                date
            )

            day, month_i, year = map(
                int,
                date.split("/")
            )

            month = month_short_name()[month_i - 1]

            # ------------------------------------------------
            # Existing prediction-input check
            #
            # Keep this because it was already part of your
            # application.
            # ------------------------------------------------

            if instrument in {
                "00001",
                "00003",
                "00004",
                "00005",
                "00006",
                "00029",
                "00030",
            }:

                was_missing_before_contributor = (
                    check_missing_prediction_data(
                        instrument,
                        date
                    )
                )

                logger.info(
                    "[Worker] Missing-data check BEFORE "
                    "contributor | Instrument=%s | Date=%s | "
                    "was_missing=%s",
                    instrument,
                    date,
                    was_missing_before_contributor
                )

            # ------------------------------------------------
            # Run contributor
            #
            # IMPORTANT:
            #
            # We now KEEP the returned object.
            # ------------------------------------------------

            contributor_instance = _contributor(
                month=month,
                year=year,
                yesterday_date=date,
            )

            # ------------------------------------------------
            # Get combined flag generated by contributor.
            #
            # This is where the Lab Report flag and existing
            # IP21 flag reach ml_job.py.
            # ------------------------------------------------

            _collect_combined_flag(
                flagged_dates,
                instrument,
                date,
                contributor_instance
            )

            # ------------------------------------------------
            # Existing prediction-input check AFTER contributor
            # ------------------------------------------------

            if instrument in {
                "00001",
                "00003",
                "00004",
                "00005",
                "00006",
                "00029",
                "00030",
            }:

                was_missing_after_contributor = (
                    check_missing_prediction_data(
                        instrument,
                        date
                    )
                )

                logger.info(
                    "[Worker] Missing-data check AFTER "
                    "contributor | Instrument=%s | Date=%s | "
                    "was_missing=%s",
                    instrument,
                    date,
                    was_missing_after_contributor
                )

            # ------------------------------------------------
            # Run prediction
            #
            # This is unchanged.
            # ------------------------------------------------

            if _calculator is not None:

                _calculator(
                    month=month,
                    year=year,
                    yesterday_date=date,
                )

            else:

                logger.warning(
                    "[Worker] No prediction calculator configured "
                    "for Instrument %s | Date %s",
                    instrument,
                    date
                )

    # ========================================================
    # Return flags to worker_loop
    # ========================================================

    logger.info(
        "[Worker] MANUAL FINAL flagged_dates = %s",
        flagged_dates
    )

    return flagged_dates


# ============================================================
# Qt-side controller
# ============================================================

class MidnightFetcherMultiprocess(QtCore.QObject):
    """
    Central job controller.

    - Automatic jobs triggered by timer
    - Manual jobs triggered by UI
    - One persistent worker process
    """

    manual_job_started = QtCore.pyqtSignal(dict)
    manual_job_finished = QtCore.pyqtSignal(dict)

    def __init__(
        self,
        interval_minutes: float,
        parent=None
    ):

        super().__init__(parent)

        # ----------------------------------------------------
        # Job queues
        # ----------------------------------------------------

        self._high_priority_queue = []
        self._low_priority_queue = []

        self._worker_busy = False

        self._job_counter = itertools.count()

        # ----------------------------------------------------
        # Multiprocessing
        # ----------------------------------------------------

        self.task_q = Queue()
        self.result_q = Queue()

        self.worker = Process(
            target=worker_loop,
            args=(
                self.task_q,
                self.result_q
            ),
            daemon=True,
        )

        self.worker.start()

        # ----------------------------------------------------
        # Result polling timer
        # ----------------------------------------------------

        self.result_timer = QtCore.QTimer(self)

        self.result_timer.setInterval(200)

        self.result_timer.timeout.connect(
            self._poll_results
        )

        self.result_timer.start()

        # ----------------------------------------------------
        # Automatic midnight timer
        # ----------------------------------------------------

        self.auto_timer = QtCore.QTimer(self)

        self.auto_timer.setSingleShot(True)

        self.auto_timer.timeout.connect(
            self._on_midnight_trigger
        )

        self._schedule_next_midnight()

    # ========================================================
    # Automatic scheduling
    # ========================================================

    def _schedule_next_midnight(self):

        now = datetime.datetime.now()

        tomorrow = now + datetime.timedelta(
            days=1
        )

        next_midnight = datetime.datetime(
            year=tomorrow.year,
            month=tomorrow.month,
            day=tomorrow.day,
            hour=0,
            minute=0,
            second=0,
            microsecond=0
        )

        time_to_midnight = (
            next_midnight - now
        )

        ms_to_midnight = int(
            time_to_midnight.total_seconds() * 1000
        )

        self.auto_timer.start(
            ms_to_midnight
        )

        logger.info(
            "[Fetcher] Controller initialized. "
            "Next automatic job scheduled for %s "
            "(in %s ms).",
            next_midnight.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            ms_to_midnight
        )

    # ========================================================
    # Midnight trigger
    # ========================================================

    def _on_midnight_trigger(self):

        logger.info(
            "[Fetcher] Midnight reached. "
            "Triggering automatic job."
        )

        self._enqueue_automatic_job()

        self._schedule_next_midnight()

    # ========================================================
    # Automatic job
    # ========================================================

    def _enqueue_automatic_job(self):

        yesterday = (
            QtCore.QDate.currentDate()
            .addDays(-1)
            .toString("dd/MM/yyyy")
        )

        job = {
            "job_id": next(self._job_counter),
            "type": "auto",
            "priority": 0,
            "instruments": None,
            "dates": [yesterday],
        }

        logger.info(
            "[Fetcher] Queued AUTOMATIC job. "
            "Job ID: %s",
            job["job_id"]
        )

        self._low_priority_queue.append(job)

        self._try_start_next_job()

    # ========================================================
    # Manual job
    # ========================================================

    def enqueue_manual_job(
        self,
        instruments: list[str],
        dates: list[str]
    ):

        """
        Called when the user clicks START.
        """

        if not instruments or not dates:

            logger.warning(
                "[Fetcher] Attempted to queue MANUAL job "
                "without valid instruments or dates."
            )

            return

        job = {
            "job_id": next(self._job_counter),
            "type": "manual",
            "priority": 1,
            "instruments": instruments,
            "dates": dates,
        }

        logger.info(
            "[Fetcher] Queued MANUAL job. "
            "Job ID: %s",
            job["job_id"]
        )

        self._high_priority_queue.append(job)

        self._try_start_next_job()

    # ========================================================
    # Job dispatch
    # ========================================================

    def _try_start_next_job(self):

        if self._worker_busy:
            return

        if self._high_priority_queue:

            job = self._high_priority_queue.pop(0)

        elif self._low_priority_queue:

            job = self._low_priority_queue.pop(0)

        else:

            return

        self._worker_busy = True

        self.task_q.put(job)

        if job["type"] == "manual":

            self.manual_job_started.emit(
                job
            )

        logger.info(
            "[Fetcher] Started job %s "
            "(Type: %s)",
            job["job_id"],
            job["type"]
        )

    # ========================================================
    # Result handling
    # ========================================================

    def _poll_results(self):

        """
        Receive completed jobs from the worker.

        IMPORTANT:
        The result dictionary now contains:

            result["flagged_dates"]

        which contains the actual Lab/IP21 flag messages.
        """

        while not self.result_q.empty():

            result = self.result_q.get()

            self._worker_busy = False

            # ------------------------------------------------
            # Manual result
            #
            # Send complete result, including flagged_dates,
            # to the UI.
            # ------------------------------------------------

            if result["type"] == "manual":

                logger.info(
                    "[Fetcher] Sending manual job result "
                    "to UI. Flagged dates: %s",
                    result.get(
                        "flagged_dates",
                        {}
                    )
                )

                self.manual_job_finished.emit(
                    result
                )

            # ------------------------------------------------
            # Success / failure logging
            # ------------------------------------------------

            if result["status"] == "success":

                logger.info(
                    "[Fetcher] Job %s (%s) "
                    "completed successfully.",
                    result["job_id"],
                    result["type"]
                )

            else:

                logger.error(
                    "[Fetcher] Job %s (%s) FAILED. "
                    "Error details:\n%s",
                    result["job_id"],
                    result["type"],
                    result["error"]
                )

            # ------------------------------------------------
            # Start next queued job
            # ------------------------------------------------

            self._try_start_next_job()

    # ========================================================
    # Cleanup
    # ========================================================

    def cleanup(self):

        logger.info(
            "[Fetcher] ML Job Cleanup sequence initiated."
        )

        try:

            self.auto_timer.stop()

            self.result_timer.stop()

        except Exception:

            logger.exception(
                "[Fetcher] Failed to stop timers "
                "during cleanup."
            )

        try:

            self.task_q.put(None)

            self.worker.join(
                timeout=5
            )

        except Exception:

            logger.exception(
                "[Fetcher] Failed to gracefully join "
                "worker process."
            )

        logger.info(
            "[Fetcher] ML Job Cleanup complete."
        )