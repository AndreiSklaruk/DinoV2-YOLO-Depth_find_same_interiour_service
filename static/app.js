/* global state */
let selectedFile   = null;
let isSearching    = false;
let _vizDepthB64   = null;
let _vizYoloB64    = null;

/* ── DOM refs ────────────────────────────────────────────────────────── */
const dropZone       = document.getElementById('drop-zone');
const fileInput      = document.getElementById('file-input');
const dropIdle       = document.getElementById('drop-idle');
const dropThumb      = document.getElementById('drop-thumb');
const thumbImg       = document.getElementById('thumb-img');
const previewImg     = document.getElementById('preview-img');
const heroState      = document.getElementById('hero-state');
const queryPreview   = document.getElementById('query-preview');
const searchBtn      = document.getElementById('search-btn');
const btnLabel       = document.getElementById('btn-label');
const btnSpinner     = document.getElementById('btn-spinner');
const resultsBar     = document.getElementById('results-bar');
const resultsGrid    = document.getElementById('results-grid');
const resultsMeta    = document.getElementById('results-meta');
const modeBadge      = document.getElementById('mode-badge');
const emptyState     = document.getElementById('empty-state');
const statusBadge    = document.getElementById('status-badge');
const statusText     = document.getElementById('status-text');
const indexedCount   = document.getElementById('indexed-count');
const rebuildBtn     = document.getElementById('rebuild-btn');
const notification   = document.getElementById('notification');
const depthToggle    = document.getElementById('depth-toggle');
const depthBadge     = document.getElementById('depth-badge');
const depthWrap      = document.getElementById('depth-wrap');
const dinoToggle     = document.getElementById('dino-toggle');
const yoloToggle     = document.getElementById('yolo-toggle');
const yoloBadge      = document.getElementById('yolo-badge');
const yoloWrap       = document.getElementById('yolo-wrap');
const mirrorToggle   = document.getElementById('mirror-toggle');

/* Viz refs */
const vizSpinner = document.getElementById('viz-spinner');
const vizBar     = document.getElementById('viz-bar');
const vizLegend  = document.getElementById('viz-legend');
const vizStat    = document.getElementById('viz-stat');
const vtDepth    = document.getElementById('vt-depth');
const vtYolo     = document.getElementById('vt-yolo');

/* ── Viz tab switcher ────────────────────────────────────────────────── */
let _origSrc    = '';   // оригинальный src после загрузки

function setVizTab(mode) {
  document.querySelectorAll('.viz-tab').forEach(t => t.classList.remove('active'));
  const activeTab = document.querySelector(`.viz-tab[data-viz="${mode}"]`);
  if (activeTab) activeTab.classList.add('active');

  if (mode === 'original') {
    if (previewImg) previewImg.src = _origSrc;
    if (vizLegend)  vizLegend.classList.add('hidden');
  } else if (mode === 'depth' && _vizDepthB64) {
    if (previewImg) previewImg.src = _vizDepthB64;
    if (vizLegend)  vizLegend.classList.add('hidden');
  } else if (mode === 'yolo' && _vizYoloB64) {
    if (previewImg) previewImg.src = _vizYoloB64;
    if (vizLegend)  vizLegend.classList.add('hidden');
  }
}

document.querySelectorAll('.viz-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    if (tab.disabled) return;
    setVizTab(tab.dataset.viz);
  });
});

/* ── Fetch visualizations (async after search) ───────────────────────── */
async function fetchVisualizations(filename) {
  if (!filename) return;
  if (vizSpinner) vizSpinner.classList.remove('hidden');

  try {
    const form = new FormData();
    form.append('filename', filename);
    const res  = await fetch('/visualize', { method: 'POST', body: form });
    if (!res.ok) return;
    const data = await res.json();

    _vizDepthB64 = data.depth_img;
    _vizYoloB64  = data.yolo_img;

    if (vtDepth) vtDepth.disabled = false;
    if (vtYolo)  vtYolo.disabled  = false;
  } catch (e) {
    console.warn('Viz error:', e);
  } finally {
    if (vizSpinner) vizSpinner.classList.add('hidden');
  }
}


/* ── Status polling ──────────────────────────────────────────────────── */
async function fetchStatus() {
  try {
    const res  = await fetch('/status');
    const data = await res.json();

    statusBadge.className = 'status-badge status-ready';
    statusText.textContent = 'Готово';
    indexedCount.textContent = `${data.indexed_rooms} комнат в базе`;

    if (data.depth_index_ready) {
      depthBadge.textContent = 'Готов';
      depthBadge.className = 'depth-badge depth-badge-ready';
      if (depthWrap) depthWrap.classList.remove('depth-unavail');
    } else {
      depthBadge.textContent = 'Недоступен';
      depthBadge.className = 'depth-badge depth-badge-unavail';
      if (depthWrap) depthWrap.classList.add('depth-unavail');
      if (depthToggle) depthToggle.checked = false;
    }

    if (data.yolo_index_ready) {
      if (yoloBadge) {
        yoloBadge.textContent = 'Готов';
        yoloBadge.className = 'depth-badge depth-badge-ready';
      }
      if (yoloWrap) yoloWrap.classList.remove('depth-unavail');
    } else {
      if (yoloBadge) {
        yoloBadge.textContent = 'Недоступен';
        yoloBadge.className = 'depth-badge depth-badge-unavail';
      }
      if (yoloWrap) yoloWrap.classList.add('depth-unavail');
      if (yoloToggle) yoloToggle.checked = false;
    }
  } catch {
    statusBadge.className = 'status-badge status-error';
    statusText.textContent = 'Нет соединения';
    if (depthBadge) { depthBadge.textContent = 'Недоступен'; depthBadge.className = 'depth-badge depth-badge-unavail'; }
    if (yoloBadge)  { yoloBadge.textContent = 'Недоступен'; yoloBadge.className = 'depth-badge depth-badge-unavail'; }
  }
}

fetchStatus();

/* ── Drag & Drop ─────────────────────────────────────────────────────── */
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

/* ── Handle selected file ────────────────────────────────────────────── */
function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    showNotification('Пожалуйста, загрузите изображение (JPEG или PNG)', 'error');
    return;
  }

  selectedFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    const src = e.target.result;

    /* Sidebar thumbnail */
    thumbImg.src = src;
    dropIdle.classList.add('hidden');
    dropThumb.classList.remove('hidden');

    /* Large preview in main panel */
    _origSrc = src;                          /* сохраняем оригинал */
    previewImg.src = src;
    heroState.classList.add('hidden');
    queryPreview.classList.remove('hidden');

    /* Сбросить viz при новом фото */
    _vizDepthB64 = null;
    _vizYoloB64  = null;
    if (vizBar) vizBar.classList.add('hidden');
    setVizTab('original');
    if (vtDepth) vtDepth.disabled = true;
    if (vtYolo)  vtYolo.disabled  = true;
  };
  reader.readAsDataURL(file);

  searchBtn.disabled = false;
  btnLabel.textContent = 'Найти похожие комнаты';
}

/* ── Search ──────────────────────────────────────────────────────────── */
searchBtn.addEventListener('click', async () => {
  if (!selectedFile || isSearching) return;
  if (dinoToggle && !dinoToggle.checked && depthToggle && !depthToggle.checked) {
    showNotification('Включите хотя бы один метод поиска (DINOv2 или Depth)', 'error');
    return;
  }
  await runSearch();
});

async function runSearch() {
  isSearching = true;
  searchBtn.disabled = true;
  btnLabel.textContent = 'Поиск...';
  btnSpinner.classList.remove('hidden');

  /* Hide old results */
  resultsBar.classList.add('hidden');
  resultsGrid.classList.add('hidden');
  emptyState.classList.remove('hidden');

  const formData = new FormData();
  formData.append('file', selectedFile);
  if (dinoToggle && !dinoToggle.checked) formData.append('use_dinov2', 'false');
  else formData.append('use_dinov2', 'true');
  if (depthToggle && depthToggle.checked) formData.append('use_depth', 'true');
  if (yoloToggle && yoloToggle.checked) formData.append('use_yolo', 'true');
  if (mirrorToggle && mirrorToggle.checked) formData.append('use_mirror', 'true');

  try {
    const t0      = performance.now();
    const res     = await fetch('/search', { method: 'POST', body: formData });
    const elapsed = ((performance.now() - t0) / 1000).toFixed(2);

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Неизвестная ошибка' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    renderResults(data.results, elapsed, data.mode, data.is_fallback);

    /* Сбросить viz-state и запустить анализ в фоне */
    _vizDepthB64 = null;
    _vizYoloB64  = null;
    if (vtDepth) vtDepth.disabled = true;
    if (vtYolo)  vtYolo.disabled  = true;
    setVizTab('original');
    if (vizBar) vizBar.classList.remove('hidden');
    if (vizLegend) vizLegend.classList.add('hidden');
    fetchVisualizations(data.query_filename);



  } catch (err) {
    showNotification(`Ошибка: ${err.message}`, 'error');
    emptyState.classList.remove('hidden');
  } finally {
    isSearching = false;
    searchBtn.disabled = false;
    btnLabel.textContent = 'Найти ещё раз';
    btnSpinner.classList.add('hidden');
  }
}

/* ── Render results ──────────────────────────────────────────────────── */
function renderResults(results, elapsed, mode, is_fallback) {
  if (!results || results.length === 0) {
    emptyState.classList.remove('hidden');
    showNotification('Результатов не найдено', 'info');
    return;
  }

  if (is_fallback) {
    showNotification('Точных совпадений нет, но вот наиболее похожие варианты', 'info');
  }

  resultsGrid.innerHTML = '';
  resultsMeta.textContent = `за ${elapsed} сек`;

  /* Mode badge */
  if (modeBadge) {
    const modeMap = {
      'dinov2': { text: 'DINOv2',             cls: 'mode-badge-dinov2' },
      'hybrid': { text: '⧆ Depth V2',          cls: 'mode-badge-hybrid' },
      'dinov2 + YOLO': { text: 'DINOv2 + YOLO', cls: 'mode-badge-yolo' },
      'hybrid + YOLO': { text: '★ Full (D+Y)',   cls: 'mode-badge-yolo' },
      'depth-only': { text: 'Depth ONLY',       cls: 'mode-badge-depth' },
      'depth-only + YOLO': { text: 'Depth+YOLO', cls: 'mode-badge-depth' },
    };
    const m = modeMap[mode] || modeMap['dinov2'];
    modeBadge.textContent = m.text;
    modeBadge.className = `mode-badge ${m.cls}`;
    modeBadge.classList.remove('hidden');
  }

  results.forEach((item, i) => {
    const card = buildCard(item, i);
    resultsGrid.appendChild(card);
  });

  emptyState.classList.add('hidden');
  resultsBar.classList.remove('hidden');
  resultsGrid.classList.remove('hidden');
}

/* ── Build single card ───────────────────────────────────────────────── */
function buildCard(item, index) {
  const card = document.createElement('div');
  card.className = 'result-card';
  card.style.animationDelay = `${index * 60}ms`;

  const scoreClass = item.score_pct >= 80 ? 'score-high'
                   : item.score_pct >= 60 ? 'score-mid'
                   : 'score-low';

  const rankClass = `rank-${item.rank}`;
  const displayName = item.filename.replace(/\.(jpg|jpeg|png)$/i, '');

  const afterUrl = item.after_filename
    ? `/images/${item.after_filename}`
    : null;

  const scoreDetail = item.mode === 'hybrid' && item.score_dino !== undefined
    ? `<span title="DINOv2: ${item.score_dino}% | Depth: ${item.score_depth}%" style="cursor:help">${item.score_pct}%</span>`
    : `${item.score_pct}%`;

  card.innerHTML = `
    <div class="card-image-wrap">
      <img class="card-img" src="/images/before/${item.filename}" alt="Комната ${item.rank}" loading="lazy" />
      ${afterUrl ? `
        <img class="card-after-img" src="${afterUrl}" alt="С мебелью" loading="lazy" />
        <div class="card-hover-label">с мебелью</div>
      ` : ''}
      <div class="card-rank ${rankClass}">#${item.rank}</div>
    </div>
    <div class="card-footer">
      <span class="card-name">${displayName}</span>
      <span class="score-pill ${scoreClass}">${scoreDetail}</span>
    </div>
  `;

  if (afterUrl) {
    const preload = new Image();
    preload.src = afterUrl;
  }

  return card;
}

/* ── Rebuild index ───────────────────────────────────────────────────── */
rebuildBtn.addEventListener('click', async () => {
  if (!confirm('Пересобрать индекс? Это займёт несколько минут.')) return;

  rebuildBtn.textContent = 'Пересборка...';
  rebuildBtn.disabled = true;

  try {
    const res  = await fetch('/rebuild-index', { method: 'POST' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Ошибка');

    showNotification(`Индекс пересобран: ${data.count} комнат за ${data.elapsed_sec}с`, 'success');
    await fetchStatus();
  } catch (err) {
    showNotification(`Ошибка пересборки: ${err.message}`, 'error');
  } finally {
    rebuildBtn.textContent = 'Пересобрать индекс';
    rebuildBtn.disabled = false;
  }
});

/* ── Notification helper ─────────────────────────────────────────────── */
let notifTimer = null;

function showNotification(message, type = 'info', duration = 4000) {
  notification.textContent = message;
  notification.className = `notification ${type} show`;

  if (notifTimer) clearTimeout(notifTimer);
  notifTimer = setTimeout(() => {
    notification.classList.remove('show');
  }, duration);
}
