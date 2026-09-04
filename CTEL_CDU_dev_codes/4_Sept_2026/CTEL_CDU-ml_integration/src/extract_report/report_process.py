import gc
import io
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile
from datetime import datetime
from src.utils.core_utility_functions import resource_path


def run_report_job(from_date, to_date, output_folder, checked_checkboxes, result_queue):
    try:
        from src.extract_report.define_cr_probes import CorrosionProbeExporter
        from src.extract_report.define_1p21 import IP21Exporter
        from src.extract_report.define_crude_blend import CrudeBlendExporter
        from src.extract_report.define_crude_general import CrudeExporter
        from src.extract_report.define_lab_report import LabReportExporter

        selected_probes = checked_checkboxes.get("selected_corrosion_probes", [])
        selected_lab_results = checked_checkboxes.get("selected_lab_reports", [])
        selected_ip21 = checked_checkboxes.get("selected_ip21", [])
        selected_crude = checked_checkboxes.get("selected_general_crude", [])
        selected_blend = checked_checkboxes.get("selected_crude_blend", [])

        start_year = int(from_date.split("/")[2])
        start_month = int(from_date.split("/")[1])
        end_year = int(to_date.split("/")[2])
        end_month = int(to_date.split("/")[1])

        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        zip_path = output_folder / f"Sentinel_reports_{datetime.now():%Y%m%d_%H%M%S}.zip"

        template_path = resource_path("assets/template_excel.xlsx")
        with open(template_path, "rb") as f:
            template_bytes = f.read()

        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            exporters = [
                CorrosionProbeExporter(
                    selected_probes,
                    start_year,
                    end_year,
                    start_month,
                    end_month,
                    tmpdir,
                    template_bytes
                ),
                
                IP21Exporter(
                    selected_ip21,
                    start_year,
                    end_year,
                    start_month,
                    end_month,
                    tmpdir,
                    template_bytes
                ),
                
                CrudeBlendExporter(
                    selected_blend,
                    start_year,
                    end_year,
                    start_month,
                    end_month,
                    tmpdir,
                    template_bytes
                ),

                CrudeExporter(
                    selected_crude,
                    start_year,
                    end_year,
                    start_month,
                    end_month,
                    tmpdir,
                    template_bytes
                ),
                
                LabReportExporter(
                    selected_lab_results,
                    start_year,
                    end_year,
                    start_month,
                    end_month,
                    tmpdir,
                    template_bytes
                )
            
            ]

            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for exporter in exporters:
                    for archive_name, file_path in exporter.generate_files():
                        zf.write(file_path, arcname=archive_name)
                        file_path.unlink(missing_ok=True)
                        gc.collect()

        result_queue.put(("finished", str(zip_path)))

    except Exception:
        result_queue.put(("error", traceback.format_exc()))