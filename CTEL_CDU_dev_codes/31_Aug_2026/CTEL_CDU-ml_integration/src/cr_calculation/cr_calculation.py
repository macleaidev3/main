
class CrCalculation():
    def __init__(self, sub_process = None, **kwargs):
          
        self.sub_process_ml_job = sub_process

        self.high_priority_queue = self.sub_process_ml_job._high_priority_queue
        self.manual_job_started_signal = self.sub_process_ml_job.manual_job_started
        self.manual_job_completed_signal = self.sub_process_ml_job.manual_job_finished
        

    def manual_calculation(self, instrument:list, dates:list, **kwargs):

        self.sub_process_ml_job.enqueue_manual_job(instrument, dates)
        
    def is_worker_busy(self):
        return self.sub_process_ml_job._worker_busy
