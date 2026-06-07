"""
Generate GitHub Pages Index for Monitoring Reports
Author: Yuvraaj M N
"""

import json
from pathlib import Path
from datetime import datetime

REPORTS_PATH = Path("reports")


def generate_index_html():
    """Create index.html for GitHub Pages"""

    # Load summary
    summary_path = REPORTS_PATH / "monitoring_summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    metrics = summary["model_performance"]
    top_features = summary["top_churn_features"]
    feature_html = "".join(
        [
            f"""
            <div class="feature-bar">
                <div class="feature-name">{feat}</div>
                <div class="bar-bg">
                    <div class="bar-fill" style="width:{100 - i*9}%"></div>
                </div>
            </div>
            """
            for i, feat in enumerate(top_features[:8])
        ]
    )
    html = f"""<!DOCTYPE html>

<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Churn Predictor — ML Monitoring Dashboard</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            padding: 24px;
        }}
        .header {{
            text-align: center;
            padding: 40px 0 32px;
            border-bottom: 1px solid #1e293b;
            margin-bottom: 32px;
        }}
        h1 {{ font-size: 28px; color: #f8fafc; margin-bottom: 8px; }}
        .subtitle {{ color: #94a3b8; font-size: 14px; }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            margin: 4px;
        }}
        .badge-green {{ background: #166534; color: #86efac; }}
        .badge-blue {{ background: #1e3a5f; color: #93c5fd; }}
        .badge-purple {{ background: #3b1f6a; color: #c4b5fd; }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}
        .metric-card {{
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: 700;
            color: #38bdf8;
            margin-bottom: 4px;
        }}
        .metric-label {{
            font-size: 12px;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .section {{
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }}
        .section h2 {{
            font-size: 16px;
            color: #f1f5f9;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid #334155;
        }}
        .report-link {{
            display: block;
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
            text-decoration: none;
            color: #e2e8f0;
            transition: border-color 0.2s;
        }}
        .report-link:hover {{ border-color: #38bdf8; }}
        .report-link .title {{ font-weight: 600; margin-bottom: 4px; }}
        .report-link .desc {{ font-size: 12px; color: #64748b; }}
        .feature-bar {{
            display: flex;
            align-items: center;
            margin-bottom: 10px;
            gap: 12px;
        }}
        .feature-name {{
            width: 240px;
            font-size: 13px;
            color: #cbd5e1;
            flex-shrink: 0;
        }}
        .bar-bg {{
            flex: 1;
            height: 8px;
            background: #0f172a;
            border-radius: 4px;
            overflow: hidden;
        }}
        .bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, #3b82f6, #38bdf8);
            border-radius: 4px;
        }}
        .model-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        .model-table th {{
            text-align: left;
            padding: 8px 12px;
            color: #64748b;
            font-size: 11px;
            text-transform: uppercase;
            border-bottom: 1px solid #334155;
        }}
        .model-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #1e293b;
        }}
        .champion {{ color: #fbbf24; font-weight: 600; }}
        .footer {{
            text-align: center;
            padding: 24px 0;
            color: #475569;
            font-size: 12px;
        }}
    </style>
</head>
<body>

<div class="header">
    <h1>🤖 Churn Predictor — ML Monitoring Dashboard</h1>
    <p class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</p>
    <div style="margin-top: 12px;">
        <span class="badge badge-green">✅ Model Healthy</span>
        <span class="badge badge-blue">XGBoost v2.0</span>
        <span class="badge badge-purple">Telco Dataset</span>
    </div>
</div>

<!-- Model Metrics -->
<div class="grid">
    <div class="metric-card">
        <div class="metric-value">{metrics['roc_auc']}</div>
        <div class="metric-label">ROC-AUC</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{metrics['recall']}</div>
        <div class="metric-label">Recall</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{metrics['precision']}</div>
        <div class="metric-label">Precision</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{metrics['accuracy']}</div>
        <div class="metric-label">Accuracy</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{metrics['f1']}</div>
        <div class="metric-label">F1 Score</div>
    </div>
</div>

<!-- Reports -->
<div class="section">
    <h2>📊 Monitoring Reports</h2>
    <a class="report-link" href="data_drift_report.html">
        <div class="title">📈 Data Drift Report</div>
        <div class="desc">Detects distribution shift between training and production data</div>
    </a>
    <a class="report-link" href="model_performance_report.html">
        <div class="title">🎯 Model Performance Report</div>
        <div class="desc">Classification metrics, confusion matrix, prediction distribution</div>
    </a>
</div>

<!-- Model Comparison -->
<div class="section">
    <h2>🏆 Model Comparison (A/B Test)</h2>
    <table class="model-table">
        <thead>
            <tr>
                <th>Model</th>
                <th>Role</th>
                <th>AUC</th>
                <th>Recall</th>
                <th>Accuracy</th>
                <th>Traffic</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Logistic Regression</td>
                <td>Baseline</td>
                <td>0.8450</td>
                <td>0.7995</td>
                <td>0.7374</td>
                <td>Reference</td>
            </tr>
            <tr>
                <td class="champion">⭐ XGBoost</td>
                <td class="champion">Champion</td>
                <td class="champion">0.8416</td>
                <td class="champion">0.8209</td>
                <td>0.7438</td>
                <td class="champion">80%</td>
            </tr>
            <tr>
                <td>LightGBM</td>
                <td>Challenger</td>
                <td>0.8429</td>
                <td>0.6070</td>
                <td>0.7857</td>
                <td>20%</td>
            </tr>
        </tbody>
    </table>
</div>

<!-- Top Features -->
<div class="section">
    <h2>🔍 Top Churn Drivers (SHAP)</h2>
    {feature_html}
</div>

<div class="footer">
    Built by Yuvraaj M N |
    <a href="https://github.com/Yuvraaj14/churn-predictor"
       style="color: #38bdf8;">GitHub</a> |
    <a href="https://dagshub.com/Yuvraaj14/churn-predictor"
       style="color: #38bdf8;">DagsHub MLflow</a>
</div>

</body>
</html>"""

    index_path = REPORTS_PATH / "index.html"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Dashboard generated: {index_path}")
    return str(index_path)


if __name__ == "__main__":
    generate_index_html()
    print("Open reports/index.html in browser to preview!")