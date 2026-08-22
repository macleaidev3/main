from PyQt6 import QtCore
from multiprocessing import Process, Queue
import itertools
import traceback
import datetime
import logging

# Retrieve the dedicated application logger
logger = logging.getLogger("SentinelApp")

# ============================================================
# Worker process entry point
# ============================================================

def worker_loop(task_q: Queue, result_q: Queue):
    """
    Persistent worker process.
    Executes ML jobs sequentially.
    """

    logger.info("[Worker] Started ML worker process.")

    while True:
        job = task_q.get()  # blocking

        if job is None:
            logger.info("[Worker] Shutdown signal received. Terminating worker loop.")
            break

        job_id = job["job_id"]
        job_type = job["type"]
        instruments = job["instruments"]
        dates = job["dates"]

        try:
            logger.debug("[Worker] Processing job_id=%s, type=%s", job_id, job_type)

            flagged_dates = {} ###me

            if job_type == "auto":
                _run_automatic_job(dates)

            elif job_type == "manual":
                # _run_manual_job(instruments, dates) ###Sunarjit
                flagged_dates = _run_manual_job(instruments, dates) ###me
            else:
                raise ValueError(f"Unknown job type: {job_type}")

            result_q.put({
                "job_id": job_id,
                "type": job_type,
                "instruments": instruments,
                "status": "success",
                "flagged_dates": flagged_dates, ###me
            })

        except Exception as e:
            logger.error("[Worker] Job %s failed: %s", job_id, str(e))
            result_q.put({
                "job_id": job_id,
                "type": job_type,
                "instruments": instruments,
                "status": "failed",
                "error": traceback.format_exc(),
            })

    logger.info("[Worker] Exiting worker process.")


# ============================================================
# Job implementations (PLACEHOLDERS)
# Replace with real ML logic
# ============================================================

def _run_automatic_job(dates):
    """
    Automatic job:
    Run ML for ALL instruments for the given dates.
    """

    from src.utils.core_utility_functions import month_short_name
    from src.crude_blend.updated_calculated_blend_properties import BlendPropertiesCalculation

    yesterday_date = dates[0]
    logger.debug("[Worker] Starting AUTOMATIC job for date: %s", yesterday_date)

    BlendPropertiesCalculation().update_blend_properties(yesterday_date)
    
    day, month_i, year = map(int, yesterday_date.split("/"))
    month = month_short_name()[month_i - 1]
    
    # for ID 00001
    logger.debug("[Worker] Running ML predictions for ID 00001")
    from src.ut_ml.ID_00001.contributor_00001 import UTThicknessContributor00001
    from src.ut_ml.ID_00001.prediction_00001 import UTThicknessPrediction00001
    UTThicknessContributor00001(month=month, year=year, yesterday_date=yesterday_date)
    UTThicknessPrediction00001(month=month, year=year, yesterday_date=yesterday_date)

    # for ID 00003
    logger.debug("[Worker] Running ML predictions for ID 00003")
    from src.ut_ml.ID_00003.contributor_00003 import UTThicknessContributor00003
    from src.ut_ml.ID_00003.prediction_00003 import UTThicknessPrediction00003
    UTThicknessContributor00003(month=month, year=year, yesterday_date=yesterday_date)
    UTThicknessPrediction00003(month=month, year=year, yesterday_date=yesterday_date)

    # for ID 00004
    logger.debug("[Worker] Running ML predictions for ID 00004")
    from src.ut_ml.ID_00004.contributor_00004 import UTThicknessContributor00004
    from src.ut_ml.ID_00004.prediction_00004 import UTThicknessPrediction00004
    UTThicknessContributor00004(month=month, year=year, yesterday_date=yesterday_date)
    UTThicknessPrediction00004(month=month, year=year, yesterday_date=yesterday_date)

    # for ID 00005
    logger.debug("[Worker] Running ML predictions for ID 00005")
    from src.ut_ml.ID_00005.contributor_00005 import UTThicknessContributor00005
    from src.ut_ml.ID_00005.prediction_00005 import UTThicknessPrediction00005
    UTThicknessContributor00005(month=month, year=year, yesterday_date=yesterday_date)
    UTThicknessPrediction00005(month=month, year=year, yesterday_date=yesterday_date)

    # for ID 00006
    logger.debug("[Worker] Running ML predictions for ID 00006")
    from src.ut_ml.ID_00006.contributor_00006 import UTThicknessContributor00006
    from src.ut_ml.ID_00006.prediction_00006 import UTThicknessPrediction00006
    UTThicknessContributor00006(month=month, year=year, yesterday_date=yesterday_date)
    UTThicknessPrediction00006(month=month, year=year, yesterday_date=yesterday_date)

    # for ID 00029
    logger.debug("[Worker] Running ML predictions for ID 00029")
    from src.ut_ml.ID_00029.contributor_00029 import UTThicknessContributor00029
    from src.ut_ml.ID_00029.prediction_00029 import UTThicknessPrediction00029
    UTThicknessContributor00029(month=month, year=year, yesterday_date=yesterday_date)
    UTThicknessPrediction00029(month=month, year=year, yesterday_date=yesterday_date)

    # for ID 00030
    logger.debug("[Worker] Running ML predictions for ID 00030")
    from src.ut_ml.ID_00030.contributor_00030 import UTThicknessContributor00030
    from src.ut_ml.ID_00030.prediction_00030 import UTThicknessPrediction00030
    UTThicknessContributor00030(month=month, year=year, yesterday_date=yesterday_date)    
    UTThicknessPrediction00030(month=month, year=year, yesterday_date=yesterday_date)

def _run_manual_job(instruments, dates):
    """
    Manual job:
    Run ML only for selected instruments and dates.
    args:
        instruments: list[str]
        dates: list[str]
    """
    from src.crude_blend.updated_calculated_blend_properties import BlendPropertiesCalculation
    from src.utils.core_utility_functions import month_short_name
    from src.server_manager.operation_manager import DatabaseManager ###me



    _contributor = None
    _calculator = None
    _blend_properties = BlendPropertiesCalculation()

    db_manager = DatabaseManager() ###me
    db_name = "SentinelDB" ###me

    ## Stores dates for which prediction input data was missing.

    flagged_dates = {
        str(instrument): []
        for instrument in instruments
    } ###me
    
    logger.info("[Worker] Starting MANUAL job for instruments: %s", instruments)

    def check_missing_prediction_data(instrument, date): ###me
            """
            Check whether any prediction input required by the UT thickness
            model is missing for the selected prediction date.
            """
    
            # These are the three inputs used by UTThicknessPrediction.
            required_properties = [
                "Density(g/ml)",
                "API",
                "Sulphur%"
            ]
    
            day, month_i, year = map(int, date.split("/"))
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
                            and value.strip().lower() in {
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
                        "[Worker] Failed to check prediction input '%s' "
                        "for instrument %s on %s.",
                        prop,
                        instrument,
                        date
                    )
    
                    # Treat an unavailable value as missing.
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


    for date in dates: # first compute blend properties for all dates
        _blend_properties.update_blend_properties(date)
    
    for instrument in instruments:
        _contributor = None
        _calculator = None
        
        if instrument == "00001":
            from src.ut_ml.ID_00001.contributor_00001 import UTThicknessContributor00001
            from src.ut_ml.ID_00001.prediction_00001 import UTThicknessPrediction00001
            _contributor = UTThicknessContributor00001
            _calculator = UTThicknessPrediction00001
            
        elif instrument == "00003":
            from src.ut_ml.ID_00003.contributor_00003 import UTThicknessContributor00003
            from src.ut_ml.ID_00003.prediction_00003 import UTThicknessPrediction00003
            _contributor = UTThicknessContributor00003
            _calculator = UTThicknessPrediction00003

        elif instrument == "00004":
            from src.ut_ml.ID_00004.contributor_00004 import UTThicknessContributor00004
            from src.ut_ml.ID_00004.prediction_00004 import UTThicknessPrediction00004
            _contributor = UTThicknessContributor00004
            _calculator = UTThicknessPrediction00004

        elif instrument == "00005":
            from src.ut_ml.ID_00005.contributor_00005 import UTThicknessContributor00005
            from src.ut_ml.ID_00005.prediction_00005 import UTThicknessPrediction00005
            _contributor = UTThicknessContributor00005
            _calculator = UTThicknessPrediction00005

        elif instrument == "00006":
            from src.ut_ml.ID_00006.contributor_00006 import UTThicknessContributor00006
            from src.ut_ml.ID_00006.prediction_00006 import UTThicknessPrediction00006
            _contributor = UTThicknessContributor00006
            _calculator = UTThicknessPrediction00006

        elif instrument == "00029":
            from src.ut_ml.ID_00029.contributor_00029 import UTThicknessContributor00029
            from src.ut_ml.ID_00029.prediction_00029 import UTThicknessPrediction00029
            _contributor = UTThicknessContributor00029
            _calculator = UTThicknessPrediction00029

        elif instrument == "00030":
            from src.ut_ml.ID_00030.contributor_00030 import UTThicknessContributor00030
            from src.ut_ml.ID_00030.prediction_00030 import UTThicknessPrediction00030
            _contributor = UTThicknessContributor00030
            _calculator = UTThicknessPrediction00030

        elif instrument == "IC-V-112":
            from src.ml_instrument.ICV112.icv112_contributor import CRContributorICV112
            from src.ml_instrument.ICV112.icv112_prediction import CRPredictionICV112
            _contributor = CRContributorICV112
            _calculator = CRPredictionICV112

        elif instrument == "IC-V-113":
            from src.ml_instrument.ICV113.icv113_contributor import CRContributorICV113
            from src.ml_instrument.ICV113.icv113_prediction import CRPredictionICV113
            _contributor = CRContributorICV113
            _calculator = CRPredictionICV113

        elif instrument == "IC-E-126":
            from src.ml_instrument.ICE126.ice126_contributor import CRContributorICE126
            from src.ml_instrument.ICE126.ice126_prediction import CRPredictionICE126
            _contributor = CRContributorICE126
            _calculator = CRPredictionICE126

        elif instrument == "IC-E-102 trial":
            from src.ml_instrument.ICE102.ice102_contributor import CRContributorICE102
            # from src.ml_instrument.ICE102.ice102_prediction import CRPredictionICE102
            _contributor = CRContributorICE102
            _calculator = None ###me

        elif instrument == "IC-E-161 A~H":
            from src.ml_instrument.ICE161.ice161_contributor  import CRContributorICE161
            from src.ml_instrument.ICE161.ice161_prediction import CRPredictionICE161
            _contributor = CRContributorICE161
            _calculator = CRPredictionICE161

        elif instrument == "IC-E-162 A~P":
            from src.ml_instrument.ICE162.ice162_contributor import CRContributorICE162
            from src.ml_instrument.ICE162.ice162_prediction import CRPredictionICE162
            _contributor = CRContributorICE162
            _calculator = CRPredictionICE162

        elif instrument == "Pipeline(IC-E-102 to IC-E-161 A~H)":
            from src.ml_pipelines.ICE102_ICE161.ice102_161_contributor import CRContributorICE102toICE161
            from src.ml_pipelines.ICE102_ICE161.ice102_161_prediction import CRPredictionICE101TO161
            _contributor = CRContributorICE102toICE161
            _calculator = CRPredictionICE101TO161

        elif instrument == "Pipeline(IC-V-101 to IC-E-102)":
            from src.ml_pipelines.ICV101_ICE102.icv101_102_contributor import CRContributorICV101toICE102
            from src.ml_pipelines.ICV101_ICE102.icv101_102_prediction import CRPredictionICV101TO102
            _contributor = CRContributorICV101toICE102
            _calculator = CRPredictionICV101TO102


        elif instrument == "Pipeline(IC-V-112 to IC-E-162 A~P)":
            from src.ml_pipelines.ICEV112_ICE162.icv112_162_contributor import CRContributorICV112toICE162
            from src.ml_pipelines.ICEV112_ICE162.icv112_162_prediction import CRPredictionICV112TO162
            _contributor = CRContributorICV112toICE162
            _calculator = CRPredictionICV112TO162

        elif instrument == "Pipeline(IC-E-126 A~D to IC-V-113)":
            from src.ml_pipelines.ICE126_ICV113.ice126_113_contributor import CRContributorICE126toICV113
            from src.ml_pipelines.ICE126_ICV113.ice126_113_prediction import CRPredictionICE126TO113 
            _contributor = CRContributorICE126toICV113
            _calculator = CRPredictionICE126TO113


        elif instrument == "Pipeline(IC-E-161 A~H to IC-V-112)":
            from src.ml_pipelines.ICE161_ICV112.ice161_112_contributor import CRContributorICE161toICV112
            from src.ml_pipelines.ICE161_ICV112.ice161_112_prediction import CRPredictionICE161TO112 
            _contributor = CRContributorICE161toICV112
            _calculator = CRPredictionICE161TO112

        elif instrument =="Pipeline(IC-E-162 A~P to IC-E-126 A~D)":
            from src.ml_pipelines.ICE162_ICE126.ice162_126_contributor import CRContributorICE162toICE126
            from src.ml_pipelines.ICE162_ICE126.ice162_126_prediction import CRPredictionICE162TO126
        
            _contributor = CRContributorICE162toICE126
            _calculator = CRPredictionICE162TO126

        for date in dates:
            logger.debug("[Worker] Executing MANUAL job component: Instrument %s | Date: %s", instrument, date)
            day, month_i, year = map(int, date.split("/"))
            month = month_short_name()[month_i - 1]         

            # ------------------------------------------------------------
            # 1. Determine whether the required prediction input data
            #    is missing for the selected date BEFORE the contributor
            #    modifies/prepares the prediction input data.
            # ------------------------------------------------------------            
            

            was_missing_before_contributor = False
            was_missing_after_contributor = False

            # Check whether the prediction inputs for this date contain missing values.

            if instrument in {   ###me
                "00001",
                "00003",
                "00004",
                "00005",
                "00006",
                "00029",
                "00030",
            }:
                was_missing_before_contributor= check_missing_prediction_data(
                    instrument,
                    date
                )

                logger.info(
                    "[Worker] Missing-data check BEFORE contributor | "
                    "Instrument=%s | Date=%s | was_missing=%s",
                    instrument,
                    date,
                    was_missing_before_contributor
                )
            # ------------------------------------------------------------
            # 2. Run contributor.
            # ------------------------------------------------------------
            _contributor(
                    month=month,
                    year=year,
                    yesterday_date=date,
                )
                
            # ------------------------------------------------------------
            # 3. Check prediction input data AGAIN
            #    AFTER the contributor runs.
            # ------------------------------------------------------------
            if instrument in {
                "00001",
                "00003",
                "00004",
                "00005",
                "00006",
                "00029",
                "00030",
            }:
                was_missing_after_contributor = check_missing_prediction_data(
                    instrument,
                    date
                )
                logger.info(
                    "[Worker] Missing-data check AFTER contributor | "
                    "Instrument=%s | Date=%s | was_missing=%s",
                    instrument,
                    date,
                    was_missing_after_contributor
                )
            # ------------------------------------------------------------
            # 4. Flag ONLY when data is STILL missing
            #    AFTER the contributor has finished.
            # ------------------------------------------------------------

            if was_missing_after_contributor:
                instrument_flags = flagged_dates.setdefault(
                    str(instrument),
                    []
                )
                date = str(date).strip()

                if date not in instrument_flags:
                    instrument_flags.append(date)

                logger.warning(
                    "[Worker] FLAGGED date %s for instrument %s "
                    "because prediction input data is STILL missing "
                    "after contributor execution.",
                    date,
                    instrument
                )
            else:
                 logger.info(
                    "[Worker] Date %s for instrument %s has valid prediction "
                    "input data after contributor execution. No flag generated.",
                    date,
                    instrument
                )      

            # ------------------------------------------------------------
            # 4. Run prediction.
            # ------------------------------------------------------------

            if _calculator is not None:
                _calculator(
                    month=month,
                    year=year,
                    yesterday_date=date,
                )
            else:
                logger.warning(
                    "[Worker] No prediction calculator configured for "
                    "Instrument %s | Date %s",
                    instrument,
                    date
                )

    logger.info(
        "[Worker] FINAL flagged_dates = %s",
        flagged_dates
    )

    return flagged_dates ###me

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

    def __init__(self, interval_minutes: float, parent=None):
        super().__init__(parent)

        # ----------------------------------------------------
        # Job queues (Qt side)
        # ----------------------------------------------------
        self._high_priority_queue = []  # manual
        self._low_priority_queue = []   # automatic
        self._worker_busy = False
        self._job_counter = itertools.count()

        # ----------------------------------------------------
        # Multiprocessing
        # ----------------------------------------------------
        self.task_q = Queue()
        self.result_q = Queue()

        self.worker = Process(
            target=worker_loop,
            args=(self.task_q, self.result_q),
            daemon=True,
        )
        self.worker.start()

        # ----------------------------------------------------
        # Result polling timer (Qt-safe)
        # ----------------------------------------------------
        self.result_timer = QtCore.QTimer(self)
        self.result_timer.setInterval(200)
        self.result_timer.timeout.connect(self._poll_results)
        self.result_timer.start()

        # # ----------------------------------------------------
        # # Automatic job timer
        # # ----------------------------------------------------
        # self.interval_ms = int(max(1,interval_minutes) * 60 * 1000)
        # self.auto_timer = QtCore.QTimer(self)
        # self.auto_timer.setInterval(self.interval_ms)
        # self.auto_timer.timeout.connect(self._enqueue_automatic_job)
        # self.auto_timer.start()

        # ----------------------------------------------------
        # Automatic job timer (Midnight scheduling)
        # ----------------------------------------------------
        self.auto_timer = QtCore.QTimer(self)
        self.auto_timer.setSingleShot(True)
        self.auto_timer.timeout.connect(self._on_midnight_trigger)
        
        # Kick off the first midnight calculation
        self._schedule_next_midnight()
    
    # ====================================================
    # Automatic job scheduling
    # ====================================================
    
    def _schedule_next_midnight(self):
        """Calculates milliseconds until the next midnight and starts the timer."""
        now = datetime.datetime.now()
        tomorrow = now + datetime.timedelta(days=1)
        
        # Build a datetime object for exactly 00:00:00 tomorrow
        next_midnight = datetime.datetime(
            year=tomorrow.year,
            month=tomorrow.month,
            day=tomorrow.day,
            hour=0, minute=0, second=0, microsecond=0
        )

        # Calculate time difference in milliseconds
        time_to_midnight = next_midnight - now
        ms_to_midnight = int(time_to_midnight.total_seconds() * 1000)

        # Start the single-shot timer
        self.auto_timer.start(ms_to_midnight)
        
        logger.info("[Fetcher] Controller initialized. Next automatic job scheduled for %s (in %s ms).", 
                    next_midnight.strftime("%Y-%m-%d %H:%M:%S"), ms_to_midnight)

    def _on_midnight_trigger(self):
        """Executes the automatic job and schedules the next one."""
        logger.info("[Fetcher] Midnight reached. Triggering automatic job.")
        self._enqueue_automatic_job()
        
        # Reschedule for the following midnight to ensure continuous daily operation
        self._schedule_next_midnight()

    # ====================================================
    # Automatic job
    # ====================================================

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
            "instruments": None,   # ALL instruments
            "dates": [yesterday],
        }

        logger.info("[Fetcher] Queued AUTOMATIC job. Job ID: %s", job["job_id"])
        self._low_priority_queue.append(job)
        self._try_start_next_job()

    # ====================================================
    # Manual job (call from UI)
    # ====================================================

    def enqueue_manual_job(self, instruments: list[str], dates: list[str]):
        """
        Called when user clicks START.
        """
        if not instruments or not dates:
            logger.warning("[Fetcher] Attempted to queue MANUAL job without valid instruments or dates.")
            return

        job = {
            "job_id": next(self._job_counter),
            "type": "manual",
            "priority": 1,
            "instruments": instruments,
            "dates": dates,
        }

        logger.info("[Fetcher] Queued MANUAL job. Job ID: %s", job["job_id"])
        self._high_priority_queue.append(job)
        self._try_start_next_job()

    # ====================================================
    # Job dispatch
    # ====================================================

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
            self.manual_job_started.emit(job)
        
        logger.info("[Fetcher] Started job %s (Type: %s)", job['job_id'], job['type'])

    # ====================================================
    # Result handling
    # ====================================================

    def _poll_results(self):
        """
        After a job is completed(success/failure),the status is added to self.result_q
        This function is triggered by self.result_timer in every 200ms.
        If there are any results in self.result_q, that means a job is completed.
        So the worker is free to start the next job.
        """
        while not self.result_q.empty():
            result = self.result_q.get()
            self._worker_busy = False

            if result["type"] == "manual":
                self.manual_job_finished.emit(result)

            if result["status"] == "success":
                logger.info("[Fetcher] Job %s (%s) completed successfully.", result['job_id'], result['type'])
            else:
                logger.error("[Fetcher] Job %s (%s) FAILED. Error details:\n%s", 
                             result['job_id'], result['type'], result['error'])

            self._try_start_next_job()

    # ====================================================
    # Cleanup
    # ====================================================

    def cleanup(self):
        logger.info("[Fetcher] ML Job Cleanup sequence initiated.")

        try:
            self.auto_timer.stop()
            self.result_timer.stop()
        except Exception:
            logger.exception("[Fetcher] Failed to stop timers during cleanup.")

        try:
            self.task_q.put(None)
            self.worker.join(timeout=5)
        except Exception:
            logger.exception("[Fetcher] Failed to gracefully join worker process.")

        logger.info("[Fetcher] ML Job Cleanup complete.")