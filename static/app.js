// static/app.js — auto-refresh & vẽ Plotly

(() => {
  const state = {
    n: 50, smooth: 5, stride: 1,
    autoRefreshMs: 60000, showProb: true, showMarkers: true, initialized: false
  };

  const el = {
    chart: document.getElementById('chart'),
    latest: document.getElementById('latest'),
    n: document.getElementById('n'),
    smooth: document.getElementById('smooth'),
    stride: document.getElementById('stride'),
    autoRefresh: document.getElementById('autoRefresh'),
    showProb: document.getElementById('showProb'),
    showMarkers: document.getElementById('showMarkers'),
    refreshBtn: document.getElementById('refreshBtn'),
    cfg: {
      symbol: document.getElementById('cfgSymbol'),
      tf: document.getElementById('cfgTF'),
      from: document.getElementById('cfgFrom'),
      to: document.getElementById('cfgTo'),
      auto: document.getElementById('cfgAutoJob'),
      saveBtn: document.getElementById('btnSave'),
    }
  };

  if (el.n) el.n.value = state.n;
  if (el.smooth) el.smooth.value = state.smooth;
  if (el.stride) el.stride.value = state.stride;
  if (el.autoRefresh) el.autoRefresh.checked = true;
  if (el.showProb) el.showProb.checked = state.showProb;
  if (el.showMarkers) el.showMarkers.checked = state.showMarkers;

  if (el.n) el.n.addEventListener('change', () => { state.n = +el.n.value || 50; refresh(); });
  if (el.smooth) el.smooth.addEventListener('change', () => { state.smooth = +el.smooth.value || 5; refresh(); });
  if (el.stride) el.stride.addEventListener('change', () => { state.stride = +el.stride.value || 1; refresh(); });
  if (el.autoRefresh) el.autoRefresh.addEventListener('change', tickAutoRefresh);
  if (el.showProb) el.showProb.addEventListener('change', () => { state.showProb = el.showProb.checked; refresh(); });
  if (el.showMarkers) el.showMarkers.addEventListener('change', () => { state.showMarkers = el.showMarkers.checked; refresh(); });
  if (el.refreshBtn) el.refreshBtn.addEventListener('click', refresh);
  if (el.cfg.saveBtn) el.cfg.saveBtn.addEventListener('click', saveSettings);

  async function loadSettings() {
    const s = await fetch('/api/settings').then(r=>r.json());
    if (el.cfg.symbol) el.cfg.symbol.value = s.symbol || 'XAUUSDm';
    if (el.cfg.tf) el.cfg.tf.value = s.timeframe || 'M15';
    if (el.cfg.from) el.cfg.from.value = (s.from || '2024-01-01').slice(0,10);
    if (el.cfg.to) el.cfg.to.value = s.to ? s.to.slice(0,10) : '';
    if (el.cfg.auto) el.cfg.auto.checked = !!s.auto_job;
  }

  async function saveSettings() {
    const body = {
      symbol: el.cfg.symbol.value.trim(),
      timeframe: el.cfg.tf.value,
      from: el.cfg.from.value || '2024-01-01',
      to: el.cfg.to.value || null,
      auto_job: !!el.cfg.auto.checked
    };
    await fetch('/api/settings', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
    await fetch('/api/train-now', {method:'POST'});
    refresh();
  }

  let timer = null;
  function tickAutoRefresh() {
    if (timer) clearInterval(timer);
    const enabled = el.autoRefresh ? el.autoRefresh.checked : true;
    if (enabled) timer = setInterval(refresh, state.autoRefreshMs);
  }

  async function fetchCandles() {
    const url = `/api/candles?n=${state.n}&smooth=${state.smooth}&stride=${state.stride}`;
    const r = await fetch(url); if (!r.ok) throw new Error('candles fetch failed');
    return r.json();
  }

  async function fetchLatest() {
    const r = await fetch('/api/latest'); if (!r.ok) return null;
    return r.json();
  }

  function traces(d) {
    const candle = {
      x: d.time, open: d.open, high: d.high, low: d.low, close: d.close,
      type: 'candlestick', name: 'XAUUSDm', increasing:{line:{width:1}}, decreasing:{line:{width:1}}
    };
    const probLine = {
      x: d.time, y: d.prob, mode:'lines', name:'prob_up (smoothed)',
      yaxis:'y2', opacity:0.6, visible: state.showProb ? true : 'legendonly'
    };
    const longM = {
      x: d.long_x, y: d.long_y, mode:'markers', name:'BUY',
      marker:{symbol:'triangle-up', size:12}, yaxis:'y',
      visible: state.showMarkers ? true : 'legendonly'
    };
    const shortM = {
      x: d.short_x, y: d.short_y, mode:'markers', name:'SELL',
      marker:{symbol:'triangle-down', size:12}, yaxis:'y',
      visible: state.showMarkers ? true : 'legendonly'
    };
    return [candle, longM, shortM, probLine];
  }

  function layout(d) {
    return {
      margin:{l:50,r:60,t:30,b:30},
      hovermode:'x unified',
      xaxis:{
        type:'date',
        rangeselector:{buttons:[
          {count:1,label:'1D',step:'day',stepmode:'backward'},
          {count:3,label:'3D',step:'day',stepmode:'backward'},
          {count:7,label:'1W',step:'day',stepmode:'backward'},
          {step:'all',label:'ALL'}
        ]},
        rangeslider:{visible:true, thickness:0.08}
      },
      yaxis:{title:'Price'},
      yaxis2:{title:'prob', overlaying:'y', side:'right', range:[0,1], showgrid:false},
      legend:{orientation:'h'},
      shapes:[
        {type:'line', xref:'paper', x0:0, x1:1, yref:'y2', y0:0.5 + d.dz, y1:0.5 + d.dz, line:{dash:'dot'}},
        {type:'line', xref:'paper', x0:0, x1:1, yref:'y2', y0:0.5 - d.dz, y1:0.5 - d.dz, line:{dash:'dot'}}
      ]
    };
  }

  async function refresh() {
    try {
      const d = await fetchCandles();
      const tr = traces(d), ly = layout(d);
      if (!state.initialized) {
        Plotly.newPlot(el.chart, tr, ly, {responsive:true});
        state.initialized = true;
      } else {
        Plotly.react(el.chart, tr, ly, {responsive:true});
      }
      const latest = await fetchLatest();
      if (latest && el.latest) {
        const sigLabel = latest.signal === 1 ? 'BUY'
                        : latest.signal === -1 ? 'SELL' : 'NO-TRADE';

        el.latest.innerHTML = `
  Date Time: <b>${latest.time}</b> | Symbol: <b>${latest.symbol}</b><br>
  
  TimeFrame: <b>${latest.tf}</b><br>
  Nến Đóng Gần Nhất: <b>${(+latest.close).toFixed(2)}</b><br>
  Ngưỡng xác suất để vào lệnh BUY=${(+latest.thrL).toFixed(2)} <br>
  Ngưỡng xác suất để vào lệnh SELL=${(+latest.thrS).toFixed(2)} <br>
  TRUNG LẬP=${(+latest.dz).toFixed(2)}
  <br>
  <span style="color:green"><b>Dự Báo: </b></span>
  <span style="color:${latest.signal === 1 ? 'green' : latest.signal === -1 ? 'red' : 'gray'}">
  <b>${sigLabel}</b>
  </span> <br>
  <span style="color:green"><b>Xác suất tăng giá: </b></span>
  <span style="color:${latest.prob_up > 0.5 ? 'green' : 'red'}">
    <b>${latest.prob_up.toFixed(4)}</b>
  </span>
`;

      }
    } catch (e) {
      console.error(e);
      if (el.latest) el.latest.textContent = '⚠️ Không tải được dữ liệu.';
    }
  }

  // init
  loadSettings().then(() => { refresh(); tickAutoRefresh(); });
  window._refresh = refresh; // tiện debug
})();
