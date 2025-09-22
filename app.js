/* ------------------------------
   Utilities
------------------------------ */
function isNumericArray(arr) {
  return arr.every(v => v === null || v === '' || (typeof v === 'number' && isFinite(v)));
}
function toNumberOrNull(v) {
  if (v === null || v === undefined) return null;
  if (typeof v === 'number') return isFinite(v) ? v : null;
  const t = ('' + v).trim();
  if (!t) return null;
  const num = Number(t.replace(/,/g, '')); // support "1,234.5"
  return isFinite(num) ? num : null;
}
function linearFit(x, y) {
  // returns {a,b} for y ≈ a*x + b
  const n = x.length;
  const mean = arr => arr.reduce((s,v)=>s+v,0)/arr.length;
  const mx = mean(x), my = mean(y);
  let num=0, den=0;
  for (let i=0;i<n;i++){ num += (x[i]-mx)*(y[i]-my); den += (x[i]-mx)**2; }
  const a = den === 0 ? 0 : num/den;
  const b = my - a*mx;
  return {a,b};
}
function r2Score(yTrue, yPred) {
  const mean = yTrue.reduce((s,v)=>s+v,0)/yTrue.length;
  let ssRes = 0, ssTot = 0;
  for (let i=0;i<yTrue.length;i++){
    ssRes += (yTrue[i]-yPred[i])**2;
    ssTot += (yTrue[i]-mean)**2;
  }
  return ssTot === 0 ? 0 : 1 - (ssRes/ssTot);
}
function rmse(yTrue, yPred) {
  let s=0; for (let i=0;i<yTrue.length;i++) s += (yTrue[i]-yPred[i])**2;
  return Math.sqrt(s/yTrue.length);
}
function mae(yTrue, yPred) {
  let s=0; for (let i=0;i<yTrue.length;i++) s += Math.abs(yTrue[i]-yPred[i]);
  return s/yTrue.length;
}
function mape(yTrue, yPred) {
  // robust to zeros: only rows where yTrue != 0
  const idx = yTrue.map((v,i)=>[v,i]).filter(([v]) => v !== 0).map(([_,i])=>i);
  if (idx.length === 0) return NaN;
  let s=0;
  for (const i of idx) s += Math.abs((yTrue[i]-yPred[i]) / Math.abs(yTrue[i]));
  return (s/idx.length)*100;
}
function makeTable(container, data, maxRows=50) {
  // data: array of rows (objects)
  const el = document.getElementById(container);
  if (!data || data.length===0){ el.innerHTML = '<p class="hint">No rows.</p>'; return; }
  const cols = Object.keys(data[0]);
  const rows = data.slice(0, maxRows);
  let html = '<table><thead><tr>';
  cols.forEach(c=> html += `<th>${c}</th>`);
  html += '</tr></thead><tbody>';
  rows.forEach(r=>{
    html += '<tr>';
    cols.forEach(c=> html += `<td>${r[c] ?? ''}</td>`);
    html += '</tr>';
  });
  html += '</tbody></table>';
  el.innerHTML = html;
}
function downloadCSV(filename, rows) {
  const cols = Object.keys(rows[0]);
  const csv = [
    cols.join(','),
    ...rows.map(r => cols.map(c => {
      const v = r[c] ?? '';
      if (typeof v === 'string' && (v.includes(',') || v.includes('"'))) {
        return `"${v.replace(/"/g,'""')}"`;
      }
      return v;
    }).join(','))
  ].join('\n');
  const blob = new Blob([csv], {type:'text/csv;charset=utf-8;'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click();
  a.remove(); URL.revokeObjectURL(url);
}

/* ------------------------------
   State
------------------------------ */
let RAW = { rows: [], columns: [] };     // from selected sheet / file
let NUMERIC_COLS = [];                   // numeric columns
let CURRENT = { x: null, actual: null, pred: null }; // selected columns

/* ------------------------------
   File handling
------------------------------ */
const fileInput = document.getElementById('file-input');
const sheetRow = document.getElementById('sheet-row');
const sheetSelect = document.getElementById('sheet-select');

fileInput.addEventListener('change', async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;

  const name = file.name.toLowerCase();
  if (name.endsWith('.csv')) {
    sheetRow.style.display = 'none';
    const text = await file.text();
    const parsed = Papa.parse(text, { header: true, skipEmptyLines: true });
    RAW = {
      rows: parsed.data,
      columns: parsed.meta.fields || Object.keys(parsed.data[0] || {})
    };
    afterDataLoad();
  } else {
    // Excel
    const data = new Uint8Array(await file.arrayBuffer());
    const wb = XLSX.read(data, { type: 'array' });
    const sheets = wb.SheetNames || [];
    if (sheets.length === 0) { alert('No sheets found.'); return; }
    // populate dropdown
    sheetSelect.innerHTML = sheets.map(s => `<option value="${s}">${s}</option>`).join('');
    sheetRow.style.display = 'flex';
    const useSheet = sheets[0];

    function pickSheet(name) {
      const ws = wb.Sheets[name];
      const json = XLSX.utils.sheet_to_json(ws, { defval: null }); // array of objects
      RAW = {
        rows: json,
        columns: json.length ? Object.keys(json[0]) : []
      };
      afterDataLoad();
    }

    pickSheet(useSheet);
    sheetSelect.onchange = (ev) => pickSheet(ev.target.value);
  }
});

/* ------------------------------
   After data is loaded
------------------------------ */
function afterDataLoad() {
  // show preview (first 50 rows)
  document.getElementById('preview-card').style.display = 'block';
  makeTable('preview-table', RAW.rows, 50);

  // build selectors
  const xSel = document.getElementById('x-select');
  const actSel = document.getElementById('actual-select');
  const predSel = document.getElementById('pred-select');

  xSel.innerHTML = `<option value="">&lt;row index&gt;</option>` +
    RAW.columns.map(c => `<option value="${c}">${c}</option>`).join('');

  // infer numeric columns
  NUMERIC_COLS = RAW.columns.filter(c => {
    const arr = RAW.rows.map(r => toNumberOrNull(r[c])).filter(v => v !== null);
    if (arr.length === 0) return false;
    return isNumericArray(arr.map(Number));
  });

  const opts = NUMERIC_COLS.map(c => `<option value="${c}">${c}</option>`).join('');
  actSel.innerHTML = opts;
  predSel.innerHTML = opts;

  document.getElementById('selectors-card').style.display = 'block';
}

/* ------------------------------
   Analyze click
------------------------------ */
document.getElementById('analyze-btn').addEventListener('click', () => {
  const xSel = document.getElementById('x-select');
  const actSel = document.getElementById('actual-select');
  const predSel = document.getElementById('pred-select');

  const actual = actSel.value;
  const pred = predSel.value;
  if (!actual || !pred) { alert('Pick Actual and Predicted columns.'); return; }
  if (actual === pred) { alert('Actual and Predicted must be different.'); return; }

  CURRENT.x = xSel.value || null;
  CURRENT.actual = actual;
  CURRENT.pred = pred;

  // Build a tidy frame with chosen columns
  const tidy = RAW.rows.map((r, idx) => {
    const out = {
      Actual: toNumberOrNull(r[actual]),
      Predicted: toNumberOrNull(r[pred])
    };
    if (CURRENT.x) out[CURRENT.x] = r[CURRENT.x];
    else out.Index = idx;
    return out;
  }).filter(row => row.Actual !== null && row.Predicted !== null);

  if (tidy.length === 0) { alert('No numeric rows after cleaning.'); return; }

  renderMetricsAndCharts(tidy);
});

/* ------------------------------
   Metrics + Charts + Downloads
------------------------------ */
function renderMetricsAndCharts(tidy) {
  const yTrue = tidy.map(r => r.Actual);
  const yPred = tidy.map(r => r.Predicted);

  const R2 = r2Score(yTrue, yPred);
  const RMSE = rmse(yTrue, yPred);
  const MAE = mae(yTrue, yPred);
  const MAPE = mape(yTrue, yPred);

  document.getElementById('metrics-card').style.display = 'block';
  document.getElementById('m-r2').textContent   = R2.toFixed(4);
  document.getElementById('m-rmse').textContent = RMSE.toLocaleString(undefined, {maximumFractionDigits:4});
  document.getElementById('m-mae').textContent  = MAE.toLocaleString(undefined, {maximumFractionDigits:4});
  document.getElementById('m-mape').textContent = isNaN(MAPE) ? '—' : MAPE.toFixed(2);

  // Metrics small table
  const metricsRows = [
    { metric: 'R2', value: R2 },
    { metric: 'RMSE', value: RMSE },
    { metric: 'MAE', value: MAE },
    { metric: 'MAPE', value: isNaN(MAPE) ? '' : MAPE }
  ];
  makeTable('metrics-table', metricsRows, 10);

  // Overlay (Actual vs Predicted)
  const xKey = CURRENT.x ? CURRENT.x : 'Index';
  const xVals = tidy.map(r => r[xKey]);

  const overlayData = [
    { x: xVals, y: tidy.map(r => r.Actual), name: 'Actual', mode: 'lines', type: 'scatter' },
    { x: xVals, y: tidy.map(r => r.Predicted), name: 'Predicted', mode: 'lines', type: 'scatter' }
  ];
  Plotly.newPlot('overlay-chart', overlayData, {
    title: 'Actual vs Predicted (Overlay)',
    legend: { orientation: 'h' },
    margin: { t: 40, r: 20, l: 50, b: 40 }
  }, {responsive:true});

  // Scatter with y=x and fitted line
  const scatterTrace = { x: yTrue, y: yPred, mode: 'markers', type: 'scatter', name: 'Points' };
  const minX = Math.min(...yTrue), maxX = Math.max(...yTrue);
  const refLine = { x: [minX, maxX], y: [minX, maxX], mode: 'lines', name: 'Perfect Fit (y = x)', line: { dash: 'dash' } };
  const {a, b} = linearFit(yTrue, yPred);
  const fitX = [minX, maxX];
  const fitY = fitX.map(x => a*x + b);
  const fitLine = { x: fitX, y: fitY, mode: 'lines', name: 'Fitted Line' };

  Plotly.newPlot('scatter-chart', [scatterTrace, refLine, fitLine], {
    title: 'Scatter: Actual vs Predicted',
    xaxis: { title: 'Actual' },
    yaxis: { title: 'Predicted' },
    legend: { orientation: 'h' },
    margin: { t: 40, r: 20, l: 50, b: 40 }
  }, {responsive:true});

  // Residuals histogram
  const resid = yTrue.map((v,i)=> v - yPred[i]);
  Plotly.newPlot('resid-chart', [{
    x: resid, type:'histogram', nbinsx: 40, name: 'Residual'
  }], {
    title: 'Residual Distribution',
    margin: { t: 40, r: 20, l: 50, b: 40 }
  }, {responsive:true});

  // Downloads
  document.getElementById('downloads-card').style.display = 'block';
  document.getElementById('download-metrics').onclick = () => downloadCSV('metrics.csv', metricsRows);
  document.getElementById('download-tidy').onclick    = () => {
    // keep only [X, Actual, Predicted]
    const xk = CURRENT.x ? CURRENT.x : 'Index';
    const tidyRows = tidy.map(r => ({ [xk]: r[xk], Actual: r.Actual, Predicted: r.Predicted }));
    downloadCSV('tidy_actual_vs_predicted.csv', tidyRows);
  };
}
