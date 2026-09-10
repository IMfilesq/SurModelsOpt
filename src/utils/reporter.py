from jinja2 import Template
from src.schemas.optimization import OptResult
from src.schemas.protocols import TupleProtocol

import pandas as pd
# Pomocnicza funkcja do zamiany protokołu na DataFrame
def protocol_to_dataframe(protocol: TupleProtocol) -> pd.DataFrame:
    return pd.DataFrame(protocol, columns=["Parametr 1", "Parametr 2"])

REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <title>OptResult Report</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <style>
        body { background-color: #f4f6f9; padding: 30px; font-family: sans-serif; }
        .card { border: none; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
        .kpi-title { font-size: 0.85rem; color: #6c757d; text-transform: uppercase; font-weight: 600; }
        .kpi-value { font-size: 1.6rem; font-weight: bold; color: #2c3e50; }
    </style>
</head>
<body>
    <div class="container" style="max-width: 900px;">
        <h2 class="mb-4 text-center">📊 Raport Optymalizacji</h2>

        <!-- Kafelki KPI z metrykami z OptResult -->
        <div class="row g-3 mb-4">
            <div class="col-md-3">
                <div class="card p-3 text-center">
                    <div class="kpi-title">Min Val</div>
                    <div class="kpi-value text-success">{{ "%.6f"|format(result.min_val) }}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card p-3 text-center">
                    <div class="kpi-title">Czas [s]</div>
                    <div class="kpi-value">{{ "%.2f"|format(result.search_time) }}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card p-3 text-center">
                    <div class="kpi-title">Liczba Iteracji</div>
                    <div class="kpi-value">{{ result.n_iter }}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card p-3 text-center">
                    <div class="kpi-title">Liczba Wywołań</div>
                    <div class="kpi-value">{{ result.n_calls }}</div>
                </div>
            </div>
        </div>

        <!-- Tabela z min_protocol -->
        <div class="card p-4">
            <h5 class="card-title mb-3">Optymalny Protokół (min_protocol)</h5>
            <div class="table-responsive">
                {{ protocol_table }}
            </div>
        </div>
    </div>
</body>
</html>
"""

def generate_opt_report(result: OptResult, filename="opt_report.html"):
    # 1. Konwersja min_protocol na tabelę HTML
    df_protocol = protocol_to_dataframe(result.min_protocol)
    protocol_table_html = df_protocol.to_html(
        classes="table table-sm table-striped table-hover align-middle", 
        index=True,
        index_names=True
    )

    # 2. Renderowanie szablonu Jinja2
    template = Template(REPORT_TEMPLATE)
    rendered_html = template.render(
        result=result,
        protocol_table=protocol_table_html
    )

    # 3. Zapis do pliku HTML
    with open(filename, "w", encoding="utf-8") as f:
        f.write(rendered_html)