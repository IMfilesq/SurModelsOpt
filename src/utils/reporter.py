import base64
import io
from typing import Optional

import matplotlib.figure
import pandas as pd
from jinja2 import Template

from src.schemas.analysis import DatasetAnalysis
from src.schemas.boundaries import Boundaries
from src.schemas.optimization import OptResult
from src.schemas.protocols import TupleProtocol


def fig_to_base64(fig: Optional[matplotlib.figure.Figure]) -> Optional[str]:
    """Konwertuje obiekt Matplotlib Figure na string data-URL w formacie Base64 PNG."""
    if fig is None:
        return None
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def protocol_to_dataframe(protocol: TupleProtocol) -> pd.DataFrame:
    return pd.DataFrame(protocol, columns=["Time", "Dose"])


REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Optimization Report - {{ model_name }}</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <style>
        body { background-color: #f4f6f9; padding: 30px; font-family: system-ui, -apple-system, sans-serif; }
        .card { border: none; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
        .kpi-title { font-size: 0.8rem; color: #6c757d; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; }
        .kpi-value { font-size: 1.5rem; font-weight: bold; color: #2c3e50; }
        .badge-model { font-size: 0.9rem; background-color: #0d6efd; }
        .badge-opt { font-size: 0.9rem; background-color: #6c757d; }
        .table-custom th { background-color: #f8f9fa; }
        .section-header { border-bottom: 2px solid #e9ecef; padding-bottom: 8px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container" style="max-width: 1000px;">
        <!-- Header -->
        <div class="d-flex justify-content-between align-items-center mb-4">
            <h2>📊 Optimization Report</h2>
            <div>
                <span class="badge badge-model px-3 py-2 me-1">Model: {{ model_name }}</span>
                <span class="badge badge-opt px-3 py-2">Algorithm: {{ opt_result.opt_name }}</span>
            </div>
        </div>

        <!-- KPI Metrics (Optimization) -->
        <div class="row g-3 mb-4">
            <div class="col-md-3">
                <div class="card p-3 text-center">
                    <div class="kpi-title">Found Model Surface Min</div>
                    <div class="kpi-value text-success">{{ "%.6f"|format(opt_result.min_val) }}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card p-3 text-center">
                    <div class="kpi-title">Simulation Val</div>
                    <div class="kpi-value text-primary">{{ "%.6f"|format(sim_val) }}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card p-3 text-center">
                    <div class="kpi-title">Search Time [s]</div>
                    <div class="kpi-value">{{ "%.2f"|format(opt_result.search_time) }}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card p-3 text-center">
                    <div class="kpi-title">Iterations / Calls</div>
                    <div class="kpi-value">{{ opt_result.n_iter }} / {{ opt_result.n_calls }}</div>
                </div>
            </div>
        </div>

        <!-- ================================================================= -->
        <!-- SECTION: Analysis of Constraint-Satisfying Data                   -->
        <!-- ================================================================= -->
        <div class="card p-4 mb-4 border-start border-4 border-info">
            <h4 class="card-title section-header text-dark fw-bold">
                📈 Analysis of Constraint-Satisfying Data
            </h4>

            <!-- Dataset Summary KPIs -->
            <div class="row g-3 mb-4">
                <div class="col-md-4">
                    <div class="card p-3 text-center bg-light border-0">
                        <div class="kpi-title">Leftover Protocol Count</div>
                        <div class="kpi-value text-secondary">{{ analysis_result.total_series_count }}</div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card p-3 text-center bg-light border-0">
                        <div class="kpi-title">Min Cancer Cells</div>
                        <div class="kpi-value text-danger">
                            {{ "%.4e"|format(analysis_result.min_cancer_cells) if analysis_result.min_cancer_cells < 0.001 else "%.2f"|format(analysis_result.min_cancer_cells) }}
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card p-3 text-center bg-light border-0">
                        <div class="kpi-title">Mean Cancer Cells</div>
                        <div class="kpi-value text-dark">
                            {{ "%.4e"|format(analysis_result.mean_cancer_cells) if analysis_result.mean_cancer_cells < 0.001 else "%.2f"|format(analysis_result.mean_cancer_cells) }}
                        </div>
                    </div>
                </div>
            </div>

            <!-- Histograms Grid -->
            {% if cancer_cells_hist_base64 or dose_hist_base64 or total_dose_hist_base64 %}
            <div class="row g-3">
                {% if cancer_cells_hist_base64 %}
                <div class="col-md-4">
                    <div class="card p-3 text-center h-100 bg-white border">
                        <div class="kpi-title mb-2">Cancer Cells Distribution</div>
                        <img src="{{ cancer_cells_hist_base64 }}" class="img-fluid rounded" alt="Cancer Cells Histogram">
                    </div>
                </div>
                {% endif %}

                {% if dose_hist_base64 %}
                <div class="col-md-4">
                    <div class="card p-3 text-center h-100 bg-white border">
                        <div class="kpi-title mb-2">Single Dose Distribution</div>
                        <img src="{{ dose_hist_base64 }}" class="img-fluid rounded" alt="Single Dose Histogram">
                    </div>
                </div>
                {% endif %}

                {% if total_dose_hist_base64 %}
                <div class="col-md-4">
                    <div class="card p-3 text-center h-100 bg-white border">
                        <div class="kpi-title mb-2">Total Dose Distribution</div>
                        <img src="{{ total_dose_hist_base64 }}" class="img-fluid rounded" alt="Total Dose Histogram">
                    </div>
                </div>
                {% endif %}
            </div>
            {% endif %}
        </div>
        <!-- ================================================================= -->

        <!-- Constraints Validation Table -->
        <div class="card p-4 mb-4">
            <h5 class="card-title mb-3">⚙️ Constraints Validation (Limits vs Actual)</h5>
            <div class="table-responsive">
                <table class="table table-sm table-bordered align-middle text-center mb-0">
                    <thead class="table-light">
                        <tr>
                            <th class="text-start">Metric</th>
                            <th>Min Interval</th>
                            <th>Max Interval</th>
                            <th>Max Total Time</th>
                            <th>Min Single Dose</th>
                            <th>Max Single Dose</th>
                            <th>Max Total Dose</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td class="text-start fw-bold text-muted">Limit (Boundaries)</td>
                            <td>{{ boundaries.min_interval }}</td>
                            <td>{{ boundaries.max_interval }}</td>
                            <td>{{ boundaries.max_total_time }}</td>
                            <td>{{ boundaries.min_single_dose }}</td>
                            <td>{{ boundaries.max_single_dose }}</td>
                            <td>{{ boundaries.max_total_dose }}</td>
                        </tr>
                        <tr class="table-white">
                            <td class="text-start fw-bold text-primary">Actual (Protocol)</td>
                            <td>{{ "%.2f"|format(bounds_check.min_interval) }}</td>
                            <td>{{ "%.2f"|format(bounds_check.max_interval) }}</td>
                            <td>{{ "%.2f"|format(bounds_check.max_total_time) }}</td>
                            <td>{{ "%.4f"|format(bounds_check.min_single_dose) }}</td>
                            <td>{{ "%.4f"|format(bounds_check.max_single_dose) }}</td>
                            <td>{{ "%.4f"|format(bounds_check.max_total_dose) }}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Optimal Protocol -->
        <div class="card p-4 mb-4">
            <h5 class="card-title mb-3 text-success">Optimal Protocol (min_protocol)</h5>
            <div class="table-responsive">
                {{ min_protocol_table }}
            </div>
        </div>
    </div>
</body>
</html>
"""


def generate_report(
    model_name: str,
    opt_result: OptResult,
    sim_val: float,
    boundaries: Boundaries,
    bounds_check: Boundaries,
    analysis_result: DatasetAnalysis,
    filename: str = "opt_report.html",
) -> None:
    # 1. Convert optimal protocol to HTML DataFrame
    df_min = protocol_to_dataframe(opt_result.min_protocol)
    min_protocol_table_html = df_min.to_html(
        classes="table table-sm table-striped table-hover align-middle table-custom",
        index=True,
        index_names=True,
    )

    # 2. Convert matplotlib figures to base64
    cancer_cells_hist_base64 = fig_to_base64(analysis_result.cancer_cells_histogram)
    dose_hist_base64 = fig_to_base64(analysis_result.dose_histogram)
    total_dose_hist_base64 = fig_to_base64(analysis_result.total_dose_histogram)

    # 3. Render Jinja2 template
    template = Template(REPORT_TEMPLATE)
    rendered_html = template.render(
        model_name=model_name,
        opt_result=opt_result,
        sim_val=sim_val,
        boundaries=boundaries,
        bounds_check=bounds_check,
        analysis_result=analysis_result,
        cancer_cells_hist_base64=cancer_cells_hist_base64,
        dose_hist_base64=dose_hist_base64,
        total_dose_hist_base64=total_dose_hist_base64,
        min_protocol_table=min_protocol_table_html,
    )

    # 4. Write output HTML file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(rendered_html)