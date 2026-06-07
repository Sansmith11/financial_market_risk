"""
app.py
Flask web server wrapping the NIFTY 500 Distribution Analysis Pipeline.
Designed for deployment on Render.com.
"""

import json
import logging
import os
import traceback
from pathlib import Path

from flask import Flask, jsonify, request, Response

# Pipeline imports
from config import CONFIG
from data import load_data
from features import build_features, get_return_series, feature_summary
from distributions import fit_all
from scoring import score_distributions, ranking_table
from regime import classify_regime
from confidence import compute_confidence
from reports import generate_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── embedded CSS/JS dashboard (single-file, no templates folder needed) ──
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>NIFTY 500 Distribution Pipeline</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',sans-serif;background:#0f1117;color:#e0e0e0;min-height:100vh}
  header{background:#1a1d2e;padding:20px 32px;border-bottom:2px solid #3a7bd5;
         display:flex;align-items:center;gap:16px}
  header h1{font-size:1.4rem;color:#fff}
  header span{font-size:.85rem;color:#888;margin-left:auto}
  .container{max-width:1200px;margin:0 auto;padding:24px 16px}
  .card{background:#1a1d2e;border:1px solid #2a2d3e;border-radius:12px;
        padding:20px 24px;margin-bottom:20px}
  .card h2{font-size:1rem;color:#3a7bd5;margin-bottom:14px;text-transform:uppercase;
            letter-spacing:.06em;border-bottom:1px solid #2a2d3e;padding-bottom:8px}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  .grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
  .kv{display:flex;justify-content:space-between;padding:5px 0;
      border-bottom:1px solid #23263a;font-size:.9rem}
  .kv:last-child{border-bottom:none}
  .kv .label{color:#888}
  .kv .value{color:#e0e0e0;font-weight:500}
  .badge{display:inline-block;padding:4px 14px;border-radius:20px;font-size:.85rem;
         font-weight:600;letter-spacing:.04em}
  .badge-bull{background:#0d4a1f;color:#4caf50}
  .badge-bear{background:#4a0d0d;color:#f44336}
  .badge-neutral{background:#3a3a0d;color:#ffc107}
  .badge-strong-bull{background:#0a3a15;color:#00e676}
  .badge-strong-bear{background:#3a0a0a;color:#ff1744}
  .badge-buy{background:#0d4a1f;color:#4caf50}
  .badge-sell{background:#4a0d0d;color:#f44336}
  .badge-hold{background:#1a2a3a;color:#42a5f5}
  table{width:100%;border-collapse:collapse;font-size:.88rem}
  th{text-align:left;padding:8px 10px;color:#3a7bd5;
     border-bottom:2px solid #2a2d3e;white-space:nowrap}
  td{padding:8px 10px;border-bottom:1px solid #1e2132}
  tr:first-child td{color:#fff;font-weight:600}
  tr:hover td{background:#1e2132}
  .form-row{display:flex;gap:12px;flex-wrap:wrap;align-items:flex-end}
  .form-group{display:flex;flex-direction:column;gap:6px;flex:1;min-width:160px}
  label{font-size:.8rem;color:#888;text-transform:uppercase;letter-spacing:.05em}
  input,select{background:#0f1117;border:1px solid #2a2d3e;color:#e0e0e0;
               padding:8px 12px;border-radius:8px;font-size:.9rem;outline:none}
  input:focus,select:focus{border-color:#3a7bd5}
  button{background:#3a7bd5;color:#fff;border:none;padding:10px 28px;
         border-radius:8px;font-size:.95rem;font-weight:600;cursor:pointer;
         transition:background .2s}
  button:hover{background:#2a6bc5}
  button:disabled{background:#333;color:#666;cursor:not-allowed}
  .error{background:#2a1010;border:1px solid #7a2020;border-radius:8px;
         padding:14px;color:#f48;font-size:.9rem}
  .spinner{display:none;border:3px solid #2a2d3e;border-top:3px solid #3a7bd5;
            border-radius:50%;width:24px;height:24px;animation:spin .8s linear infinite;
            margin:0 auto}
  @keyframes spin{to{transform:rotate(360deg)}}
  .conf-bar{background:#2a2d3e;border-radius:4px;height:10px;margin-top:6px}
  .conf-fill{height:10px;border-radius:4px;background:linear-gradient(90deg,#3a7bd5,#00e5ff);
             transition:width .5s}
  .metric-big{font-size:2rem;font-weight:700;color:#3a7bd5;text-align:center;
               padding:10px 0}
  .metric-label{font-size:.8rem;color:#888;text-align:center;text-transform:uppercase}
  @media(max-width:640px){.grid2,.grid3{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
    <rect width="32" height="32" rx="8" fill="#3a7bd5"/>
    <polyline points="4,24 10,14 16,18 22,8 28,12" stroke="#fff" stroke-width="2.5"
              fill="none" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>
  <h1>NIFTY 500 — Distribution Analysis Pipeline</h1>
  <span id="ts"></span>
</header>

<div class="container">

  <!-- Input card -->
  <div class="card">
    <h2>Run Analysis</h2>
    <div class="form-row">
      <div class="form-group">
        <label>Ticker (Yahoo Finance)</label>
        <input id="ticker" value="^CRSLDX" placeholder="e.g. ^CRSLDX or RELIANCE.NS"/>
      </div>
      <div class="form-group">
        <label>Period</label>
        <select id="period">
          <option value="6mo">6 Months</option>
          <option value="1y" selected>1 Year</option>
          <option value="2y">2 Years</option>
          <option value="3y">3 Years</option>
        </select>
      </div>
      <div class="form-group" style="flex:0">
        <label>&nbsp;</label>
        <button id="runBtn" onclick="runAnalysis()">▶ Run</button>
      </div>
      <div class="spinner" id="spinner"></div>
    </div>
    <div id="errorBox" class="error" style="display:none;margin-top:14px"></div>
  </div>

  <!-- Results (hidden until run) -->
  <div id="results" style="display:none">

    <!-- Top metrics row -->
    <div class="grid3">
      <div class="card" style="text-align:center">
        <h2>Trading Signal</h2>
        <div id="signalBadge" class="metric-big"></div>
        <div id="signalRationale" style="font-size:.82rem;color:#888;margin-top:8px"></div>
      </div>
      <div class="card" style="text-align:center">
        <h2>Market Regime</h2>
        <div id="regimeBadge" class="metric-big"></div>
        <div id="regimeScore" style="font-size:.82rem;color:#888;margin-top:8px"></div>
      </div>
      <div class="card">
        <h2>Confidence</h2>
        <div id="confPct" class="metric-big"></div>
        <div class="conf-bar"><div class="conf-fill" id="confFill" style="width:0%"></div></div>
        <div id="confLabel" class="metric-label" style="margin-top:6px"></div>
      </div>
    </div>

    <div class="grid2">
      <!-- Statistical Summary -->
      <div class="card">
        <h2>Statistical Summary</h2>
        <div id="statSummary"></div>
      </div>
      <!-- Best Distribution -->
      <div class="card">
        <h2>Best-Fit Distribution</h2>
        <div id="bestDist"></div>
      </div>
    </div>

    <!-- Ranking table -->
    <div class="card">
      <h2>Distribution Ranking</h2>
      <div style="overflow-x:auto">
        <table id="rankTable">
          <thead>
            <tr>
              <th>#</th><th>Distribution</th><th>Score</th>
              <th>KS Stat</th><th>KS p-val</th>
              <th>Log-Lik</th><th>AIC</th><th>BIC</th>
            </tr>
          </thead>
          <tbody id="rankBody"></tbody>
        </table>
      </div>
    </div>

    <div class="grid2">
      <!-- Regime details -->
      <div class="card">
        <h2>Regime Details</h2>
        <div id="regimeDetails"></div>
      </div>
      <!-- Tail Risk -->
      <div class="card">
        <h2>Tail Risk</h2>
        <div id="tailRisk"></div>
      </div>
    </div>

  </div><!-- /results -->
</div><!-- /container -->

<script>
document.getElementById('ts').textContent = new Date().toLocaleString();

function badgeClass(signal) {
  var s = signal.toLowerCase();
  if (s.includes('strong bull')) return 'badge-strong-bull';
  if (s.includes('strong bear')) return 'badge-strong-bear';
  if (s.includes('bull')) return 'badge-bull';
  if (s.includes('bear')) return 'badge-bear';
  if (s.includes('buy')) return 'badge-buy';
  if (s.includes('sell')) return 'badge-sell';
  if (s.includes('hold')) return 'badge-hold';
  return 'badge-neutral';
}

function kv(label, value) {
  return '<div class="kv"><span class="label">' + label +
         '</span><span class="value">' + value + '</span></div>';
}

function runAnalysis() {
  var ticker = document.getElementById('ticker').value.trim();
  var period = document.getElementById('period').value;
  if (!ticker) { alert('Enter a ticker symbol'); return; }

  document.getElementById('runBtn').disabled = true;
  document.getElementById('spinner').style.display = 'block';
  document.getElementById('errorBox').style.display = 'none';
  document.getElementById('results').style.display = 'none';

  fetch('/api/analyze?ticker=' + encodeURIComponent(ticker) + '&period=' + period)
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (data.error) throw new Error(data.error);
      renderResults(data);
    })
    .catch(function(err) {
      document.getElementById('errorBox').textContent = 'Error: ' + err.message;
      document.getElementById('errorBox').style.display = 'block';
    })
    .finally(function() {
      document.getElementById('runBtn').disabled = false;
      document.getElementById('spinner').style.display = 'none';
    });
}

function renderResults(d) {
  var sig = d.trading_signal;
  var reg = d.regime;
  var conf = d.confidence;
  var best = d.best_distribution;
  var stat = d.statistical_summary;
  var tail = d.tail_risk;

  // Signal
  var el = document.getElementById('signalBadge');
  el.innerHTML = '<span class="badge ' + badgeClass(sig.signal) + '">' + sig.signal + '</span>';
  document.getElementById('signalRationale').textContent = sig.risk_note;

  // Regime
  var rel = document.getElementById('regimeBadge');
  rel.innerHTML = '<span class="badge ' + badgeClass(reg.classification) + '">' +
                  reg.classification + '</span>';
  document.getElementById('regimeScore').textContent = 'Score: ' + reg.raw_score + '/100';

  // Confidence
  document.getElementById('confPct').textContent = conf.score_pct + '%';
  document.getElementById('confFill').style.width = conf.score_pct + '%';
  document.getElementById('confLabel').textContent = conf.interpretation;

  // Stat summary
  var ss = '';
  var statFields = [
    ['Observations', stat.n],
    ['Annualised Return', (stat.annualised_return * 100).toFixed(2) + '%'],
    ['Annualised Volatility', (stat.annualised_volatility * 100).toFixed(2) + '%'],
    ['Skewness', stat.skewness],
    ['Excess Kurtosis', stat.excess_kurtosis],
    ['VaR 5%', (stat.VaR_5pct * 100).toFixed(3) + '%'],
    ['CVaR 5%', (stat.CVaR_5pct * 100).toFixed(3) + '%'],
  ];
  statFields.forEach(function(p) { ss += kv(p[0], p[1]); });
  document.getElementById('statSummary').innerHTML = ss;

  // Best dist
  var bd = '';
  bd += kv('Distribution', best.name);
  bd += kv('Final Score', best.final_score);
  bd += kv('Log-Likelihood', best.log_likelihood);
  bd += kv('AIC', best.aic);
  bd += kv('BIC', best.bic);
  bd += kv('KS Statistic', best.ks_statistic);
  bd += kv('KS p-value', best.ks_pvalue);
  var params = best.params;
  Object.keys(params).forEach(function(k) {
    bd += kv('param: ' + k, Number(params[k]).toFixed(6));
  });
  document.getElementById('bestDist').innerHTML = bd;

  // Ranking table
  var tbody = '';
  d.ranking_table.forEach(function(r) {
    var cls = r.rank === 1 ? ' style="color:#3a7bd5"' : '';
    tbody += '<tr' + cls + '><td>' + r.rank + '</td>' +
             '<td>' + r.distribution + '</td>' +
             '<td>' + r.final_score + '</td>' +
             '<td>' + r.ks_stat + '</td>' +
             '<td>' + r.ks_pvalue + '</td>' +
             '<td>' + r.log_likelihood + '</td>' +
             '<td>' + r.aic + '</td>' +
             '<td>' + r.bic + '</td></tr>';
  });
  document.getElementById('rankBody').innerHTML = tbody;

  // Regime details
  var rd = '';
  rd += kv('Classification', reg.classification);
  rd += kv('Return Signal', reg.return_signal);
  rd += kv('Volatility', reg.volatility_signal);
  rd += kv('Trend', reg.trend_signal);
  rd += kv('Tail Behaviour', reg.tail_signal);
  rd += kv('Distribution', reg.distribution_signal);
  var sub = reg.sub_scores;
  Object.keys(sub).forEach(function(k) {
    rd += kv(k + ' score', sub[k]);
  });
  document.getElementById('regimeDetails').innerHTML = rd;

  // Tail risk
  var tr = '';
  tr += kv('VaR 5%', (tail.VaR_5pct * 100).toFixed(3) + '%');
  tr += kv('CVaR 5%', (tail.CVaR_5pct * 100).toFixed(3) + '%');
  tr += kv('Skewness', tail.skewness);
  tr += kv('Excess Kurtosis', tail.excess_kurtosis);
  tr += kv('Best Dist Tail Score', tail.best_dist_tail_score);
  tr += kv('Interpretation', tail.interpretation);
  document.getElementById('tailRisk').innerHTML = tr;

  document.getElementById('results').style.display = 'block';
  document.getElementById('ts').textContent = 'Last run: ' + new Date().toLocaleString();
}
</script>
</body>
</html>"""


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return Response(DASHBOARD_HTML, mimetype="text/html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/analyze")
def analyze():
    ticker = request.args.get("ticker", "^CRSLDX").strip()
    period = request.args.get("period", "1y").strip()

    try:
        cfg = CONFIG

        df_raw = load_data(ticker=ticker, period=period, cfg=cfg.data)
        df_feat = build_features(df_raw, cfg=cfg.features)
        returns = get_return_series(df_feat)

        dist_results = fit_all(returns, names=cfg.distributions)
        scored = score_distributions(dist_results, weights=cfg.scoring)
        regime = classify_regime(df_feat, scored, cfg=cfg.regime)
        confidence = compute_confidence(scored, regime, n_samples=len(returns))

        report = generate_report(
            df_feat, scored, regime, confidence,
            ticker=ticker,
            output_dir=None,        # no file I/O on Render free tier
        )

        return jsonify(report)

    except Exception as exc:
        logger.error("Pipeline error: %s\n%s", exc, traceback.format_exc())
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
