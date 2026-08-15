/* ── State ──────────────────────────────────────── */
const state = {
  sessionId: null,
  totalPages: 0,
  currentPage: 1,
  scanning: false,
  pageResults: {},       // page_num → { detections, html, translated }
  pageImageUrls: {},     // page_num → image URL
  targetLang: '',
  zoom: 1,
  sourceName: '',
  scanStartedAt: null,
  timerId: null,
  resultMode: 'markdown',
  markdownRenderTimer: null,
  scanPaused: false,
  scanStopped: false,
  scanAbortController: null,
  typingCurrent: '',
  typingTarget: '',
  typingTimer: null,
  typingFinal: false,
  typingPage: null,
  followScanningPage: true,
  scanningPage: null,
  markdownPageSize: null,
};

/* ── DOM refs ───────────────────────────────────── */
const $ = id => document.getElementById(id);
const fileInput    = $('file-input');
const scanBtn      = $('scan-btn');
const prevBtn      = $('prev-page');
const nextBtn      = $('next-page');
const pageInd      = $('page-indicator');
const transLang    = $('translate-lang');
const transBtn     = $('translate-btn');
const exportMenuBtn = $('export-menu-btn');
const exportBtn    = exportMenuBtn;
const srcImg       = $('source-image');
const srcPlaceholder = $('source-placeholder');
const ocrContent   = $('ocr-content');
const progressBar  = $('progress-bar');
const progressFill = $('progress-fill');
const progressText = $('progress-text');
const uploadProgress = $('upload-progress');
const uploadProgressFill = $('upload-progress-fill');
const uploadProgressText = $('upload-progress-text');
const uploadProgressDetail = $('upload-progress-detail');
const statusText   = $('status-text');
const modelInfo    = $('model-info');
const divider      = $('divider');
const toast        = $('toast');
const pageList     = $('page-list');
const pageCount    = $('page-count');
const fileCard     = $('file-card');
const fileName     = $('file-name');
const fileDetails  = $('file-details');
const modelPill    = $('model-pill');
const resultStats  = $('result-stats');
const copyBtn      = $('copy-btn');
const sourceView   = $('source-view');
const zoomLevel    = $('zoom-level');
const elapsedTime  = $('elapsed-time');
const progressDetail = $('progress-detail');

/* ── Init ───────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  fileInput.addEventListener('change', handleUpload);
  scanBtn.addEventListener('click', startScan);
  prevBtn.addEventListener('click', () => goToPage(state.currentPage - 1));
  nextBtn.addEventListener('click', () => goToPage(state.currentPage + 1));
  transBtn.addEventListener('click', translatePage);
  exportMenuBtn.addEventListener('click', toggleExportMenu);
  $('export-md-option').addEventListener('click', () => { closeExportMenu(); exportMarkdown(); });
  $('export-word-option').addEventListener('click', () => { closeExportMenu(); showExportDialog(); });
  document.addEventListener('click', e => {
    if (!exportMenuBtn.parentElement.contains(e.target)) closeExportMenu();
  });
  $('markdown-view-btn').addEventListener('click', () => setResultMode('markdown'));
  $('text-view-btn').addEventListener('click', () => setResultMode('text'));
  transLang.addEventListener('change', () => {
    state.targetLang = transLang.value;
    transBtn.disabled = !state.targetLang || !state.sessionId;
  });
  $('theme-btn').addEventListener('click', toggleTheme);
  $('zoom-in').addEventListener('click', () => setZoom(state.zoom + 0.15));
  $('zoom-out').addEventListener('click', () => setZoom(state.zoom - 0.15));
  $('fit-btn').addEventListener('click', fitImage);
  copyBtn.addEventListener('click', copyResult);
  $('pause-btn').addEventListener('click', toggleScanPause);
  $('stop-btn').addEventListener('click', stopScan);
  $('follow-btn').addEventListener('click', toggleFollowScanning);
  setupDropZone();
  setupDivider();
  setupKeyboard();
  const savedTheme = localStorage.getItem('ocr-theme');
  if (savedTheme) document.documentElement.dataset.theme = savedTheme;
  updateThemeButton();
  checkHealth();
  if (window.mermaid) mermaid.initialize({ startOnLoad: false, securityLevel: 'strict', theme: document.documentElement.dataset.theme === 'dark' ? 'dark' : 'default' });
  new ResizeObserver(() => scheduleMarkdownFit()).observe(ocrContent);
});

/* ── Health check ───────────────────────────────── */
async function checkHealth() {
  try {
    const r = await fetch('/api/health');
    const d = await r.json();
    const ready = Boolean(d.loaded ?? d.model_loaded);
    statusText.textContent = ready ? '模型已就绪' : '模型加载中…';
    modelInfo.textContent = ready ? `模型就绪 · ${(d.device || '本机').toUpperCase()}` : '正在加载模型';
    modelPill.classList.toggle('ready', ready);
    modelPill.classList.toggle('loading', !ready);
    if (!ready) setTimeout(checkHealth, 2500);
  } catch {
    statusText.textContent = '无法连接本地服务';
    modelInfo.textContent = '服务未连接';
    modelPill.classList.add('error');
  }
}

/* ── Upload ─────────────────────────────────────── */
function showUploadProgress(totalPages) {
  uploadProgress.classList.remove('hidden');
  uploadProgressFill.style.width = '0%';
  uploadProgressText.textContent = '正在读取文档';
  uploadProgressDetail.textContent = `解析页面 0 / ${totalPages} 页…`;
  statusText.textContent = `正在读取 0 / ${totalPages} 页…`;
}

function updateUploadProgress(done, totalPages) {
  const pct = totalPages > 0 ? Math.min(100, Math.round((done / totalPages) * 100)) : 0;
  uploadProgressFill.style.width = `${pct}%`;
  uploadProgressDetail.textContent = `解析页面 ${done} / ${totalPages} 页`;
  uploadProgressText.textContent = pct >= 100 ? '读取完成' : '正在读取文档';
  statusText.textContent = `正在读取 ${done} / ${totalPages} 页…`;
}

function hideUploadProgress() {
  uploadProgress.classList.add('hidden');
}

async function waitForUpload(sessionId, totalPages) {
  while (true) {
    const r = await fetch(`/api/upload-progress/${sessionId}`);
    const d = await r.json();
    if (!r.ok || d.error) throw new Error(d.error || `服务器返回异常 (${r.status})`);
    updateUploadProgress(d.processed_pages || 0, totalPages);
    if (!d.processing) {
      if (d.error) throw new Error(d.error);
      return;
    }
    await new Promise(resolve => setTimeout(resolve, 300));
  }
}

async function handleUpload(e) {
  const file = e instanceof File ? e : e.target.files[0];
  if (!file) return;

  scanBtn.disabled = true;
  statusText.textContent = `正在读取 ${file.name}…`;
  $('source-title').textContent = '正在读取文档';
  ocrContent.innerHTML = emptyResult('正在打开文档…', '文件已接收，正在解析页面');
  srcPlaceholder.style.display = 'none';
  sourceView.classList.add('uploading');
  sourceView.dataset.uploadName = file.name;
  showToast('文件已接收，正在打开…');

  statusText.textContent = '正在打开文档…';
  const form = new FormData();
  form.append('file', file);

  try {
    const r = await fetch('/api/upload', { method: 'POST', body: form });
    const d = await r.json();
    if (!r.ok || d.error || !d.session_id) {
      const message = d.error || `服务器返回异常 (${r.status})`;
      throw new Error(message);
    }

    if (d.processing) {
      showUploadProgress(d.total_pages);
      await waitForUpload(d.session_id, d.total_pages);
      hideUploadProgress();
    }

    state.sessionId = d.session_id;
    state.totalPages = d.total_pages;
    state.currentPage = 1;
    state.followScanningPage = true;
    state.pageResults = {};
    state.pageImageUrls = {};
    state.sourceName = d.source_name;
    state.zoom = 1;

    for (let i = 1; i <= d.total_pages; i++) {
      state.pageImageUrls[i] = `/api/page-image/${d.session_id}/${i}`;
    }

    if (d.total_pages >= 1) loadPageImage(1);

    scanBtn.disabled = false;
    pageInd.textContent = `1 / ${d.total_pages}`;
    prevBtn.disabled = true;
    nextBtn.disabled = d.total_pages <= 1;
    transBtn.disabled = !state.targetLang;
    exportBtn.disabled = true;
    exportMenuBtn.disabled = true;
    statusText.textContent = `已打开 ${d.source_name}`;
    fileName.textContent = d.source_name;
    fileDetails.textContent = `${d.total_pages} 页 · ${formatBytes(file.size)}`;
    fileCard.classList.remove('hidden');
    pageCount.textContent = d.total_pages;
    $('source-title').textContent = d.source_name;
    ocrContent.innerHTML = emptyResult('文档已准备好', '点击右上角「开始识别」解析内容');
    if (resultStats) resultStats.textContent = '等待识别';
    copyBtn.disabled = true;
    ['zoom-in','zoom-out','fit-btn'].forEach(id => $(id).disabled = false);
    renderPageList();
    showToast('文档已打开');
  } catch (err) {
    hideUploadProgress();
    showToast('上传失败: ' + err.message);
    sourceView.classList.remove('uploading');
    srcPlaceholder.style.display = '';
    ocrContent.innerHTML = emptyResult('打开失败', '请重新拖入文件或点击「打开文档」');
  } finally {
    hideUploadProgress();
    sourceView.classList.remove('uploading');
    delete sourceView.dataset.uploadName;
  }
}

/* ── Scan (SSE) — real-time line-by-line ────────── */
async function startScan() {
  if (state.scanning) return;
  state.scanning = true;
  // Every new run, including a restart after Stop, begins in follow mode.
  state.followScanningPage = true;
  const followButton = $('follow-btn');
  followButton.classList.add('active');
  followButton.setAttribute('aria-pressed', 'true');
  followButton.querySelector('span').textContent = '自动跟随';
  state.scanPaused = false;
  state.scanStopped = false;
  state.scanAbortController = new AbortController();
  resetTypewriter();
  $('pause-btn').textContent = '暂停';
  $('pause-btn').disabled = false;
  $('stop-btn').disabled = false;
  document.querySelector('.spinner')?.classList.remove('paused');
  scanBtn.disabled = true;
  scanBtn.querySelector('span').textContent = '识别中';
  progressBar.classList.remove('hidden');
  requestAnimationFrame(() => { fitImage(); scheduleMarkdownFit(); });
  progressFill.style.width = '0%';
  progressText.textContent = '准备中...';
  progressDetail.textContent = '正在分析文档结构';
  startTimer();

  try {
    const r = await fetch('/api/scan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      // Ask for the full configured cap; the backend clamps to OCR_MAX_TOKENS
      // and reports truncation per page when the cap is hit.
      body: JSON.stringify({ session_id: state.sessionId, max_length: 8192 }),
      signal: state.scanAbortController.signal,
    });

    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      while (true) {
        const boundary = buf.includes('\r\n\r\n') ? '\r\n\r\n' : (buf.includes('\n\n') ? '\n\n' : null);
        if (!boundary) break;
        const idx = buf.indexOf(boundary);
        if (idx === -1) break;
        const chunk = buf.substring(0, idx);
        buf = buf.substring(idx + boundary.length);
        if (chunk.trim()) handleSSEChunk(chunk);
      }
    }
    if (buf.trim()) handleSSEChunk(buf);
  } catch (err) {
    if (err.name !== 'AbortError') showToast('扫描出错: ' + err.message);
  }

  state.scanning = false;
  scanBtn.disabled = false;
  scanBtn.querySelector('span').textContent = state.scanStopped ? '重新识别' : '重新识别';
  progressBar.classList.add('hidden');
  requestAnimationFrame(() => {
    fitImage();
    scheduleMarkdownFit();
    // Flex layout settles after the progress strip disappears.
    requestAnimationFrame(scheduleMarkdownFit);
    setTimeout(scheduleMarkdownFit, 220);
  });
  stopTimer();
  exportBtn.disabled = false;
  copyBtn.disabled = false;
  statusText.textContent = state.scanStopped ? '识别已停止' : `扫描完成 (${state.totalPages} 页)`;
  updateResultStats();
}

function handleSSEChunk(chunk) {
  const lines = chunk.split(/\r?\n/);
  let eventType = '';
  let dataStr = '';

  for (const line of lines) {
    if (line.startsWith('event:')) eventType = line.slice(6).trim();
    else if (line.startsWith('data:')) dataStr = line.slice(5).trim();
    else if (line.startsWith(':')) continue;
  }

  if (!eventType || !dataStr) return;
  let data;
  try { data = JSON.parse(dataStr); } catch { return; }

  switch (eventType) {
    case 'page_start':
      state.scanningPage = data.page_num;
      progressText.textContent = `扫描第 ${data.page_num} / ${data.total_pages} 页...`;
      progressFill.style.width = `${(data.page_num - 1) / data.total_pages * 100}%`;
      setPageStatus(data.page_num, 'scanning');
      progressDetail.textContent = '视觉模型正在读取页面内容';
      if (state.followScanningPage) {
        resetTypewriter();
        state.currentPage = data.page_num;
        pageInd.textContent = `${data.page_num} / ${data.total_pages}`;
        prevBtn.disabled = data.page_num <= 1;
        nextBtn.disabled = data.page_num >= data.total_pages;
        document.querySelectorAll('.page-item').forEach((el, i) => el.classList.toggle('active', i + 1 === data.page_num));
        loadPageImage(data.page_num);
        if (state.resultMode === 'markdown') {
          state.typingPage = data.page_num;
          renderMarkdown('', true);
        } else {
          ocrContent.innerHTML = `<div class="ocr-page" data-page="${data.page_num}"></div>`;
        }
      }
      break;

    case 'page_progress':
      const statusMap = { scanning: 'OCR识别中', parsing: '解析结果中', converting: '转换中' };
      progressText.textContent = `第 ${data.page_num} 页 - ${statusMap[data.status] || data.status}`;
      progressDetail.textContent = data.status === 'parsing' ? '正在整理版面与文本块' : '请保持此页面开启';
      break;

    case 'det_result':
      // Append this single detection line to the current page container
      appendDetection(data);
      break;

    case 'token':
      if (!state.pageResults[data.page_num]) state.pageResults[data.page_num] = { detections: [] };
      state.pageResults[data.page_num].raw = data.text;
      state.pageResults[data.page_num].markdown = data.markdown;
      if (data.truncated) state.pageResults[data.page_num].truncated = true;
      if (!state.scanStopped && state.resultMode === 'markdown' && data.page_num === state.currentPage) {
        updateTypewriterTarget(data.markdown, Boolean(data.done), data.page_num);
      }
      progressDetail.textContent = `已生成 ${data.tokens || 0} tokens · 正在实时排版`;
      break;

    case 'page_done':
      // Store full results
      state.pageResults[data.page_num] = {
        ...(state.pageResults[data.page_num] || {}),
        detections: state.pageResults[data.page_num]?.detections || [],
        html: data.html,
        markdown: data.markdown,
        truncated: Boolean(data.truncated),
      };
      if (!state.scanStopped && data.page_num === state.currentPage) {
        if (state.resultMode === 'markdown') updateTypewriterTarget(data.markdown, true, data.page_num);
        else renderOCRContent(data.html);
        updatePageWarning();
      }
      setPageStatus(data.page_num, data.truncated ? 'warning' : 'done');
      updateResultStats(data.page_num);
      break;

    case 'page_image':
      state.pageImageUrls[data.page_num] = data.image_url;
      if (data.page_num === state.currentPage) loadPageImage(data.page_num);
      break;

    case 'scan_complete':
      progressFill.style.width = '100%';
      progressText.textContent = '扫描完成!';
      exportBtn.disabled = false;
      exportMenuBtn.disabled = false;
      copyBtn.disabled = false;
      progressDetail.textContent = '所有页面均已处理完成';
      break;

    case 'scan_stopped':
      state.scanStopped = true;
      progressText.textContent = '识别已停止';
      progressDetail.textContent = '已保留当前生成的内容';
      break;

    case 'error':
      showToast(`第 ${data.page_num} 页出错: ${data.message}`);
      break;
  }
}

/* ── Append a single detection to the right panel ── */
function appendDetection(data) {
  const page = data.page_num;
  const det = data.detection;
  const detHtml = data.html;

  // Track detections for this page
  if (!state.pageResults[page]) state.pageResults[page] = { detections: [] };
  state.pageResults[page].detections.push(det);

  if (state.resultMode === 'markdown') {
    updateResultStats(page);
    return;
  }

  // A page may continue arriving in the background after the user navigates.
  // Keep storing it, but never let it replace the page currently on screen.
  if (page !== state.currentPage) {
    updateResultStats(page);
    return;
  }

  // Find or create the page container
  let container = ocrContent.querySelector(`.ocr-page[data-page="${page}"]`);
  if (!container) {
    ocrContent.innerHTML = `<div class="ocr-page" data-page="${page}"></div>`;
    container = ocrContent.querySelector(`.ocr-page[data-page="${page}"]`);
  }

  // Append the detection HTML
  container.insertAdjacentHTML('beforeend', detHtml);

  // Auto-scroll right panel to bottom
  ocrContent.scrollTop = ocrContent.scrollHeight;
  updateResultStats(page);
}

/* ── Render ─────────────────────────────────────── */
function renderOCRContent(html) {
  ocrContent.classList.remove('markdown-preview');
  ocrContent.innerHTML = html;
  attachEditListeners();
}

function setResultMode(mode) {
  resetTypewriter();
  state.resultMode = mode;
  $('markdown-view-btn').classList.toggle('active', mode === 'markdown');
  $('text-view-btn').classList.toggle('active', mode === 'text');
  $('result-title').textContent = mode === 'markdown' ? '实时 Markdown' : '可编辑文本';
  const result = state.pageResults[state.currentPage];
  if (mode === 'markdown') {
    const markdown = result?.markdown || '';
    primeTypewriter(state.currentPage, markdown, Boolean(result?.html));
    renderMarkdown(markdown, state.scanning && !result?.html);
  }
  else if (result?.html) renderOCRContent(result.html);
  else if (result?.detections?.length) goToPage(state.currentPage);
  else ocrContent.innerHTML = emptyResult('此页尚未识别', '点击「开始识别」后结果会显示在这里');
}

function scheduleMarkdownRender(markdown, live) {
  clearTimeout(state.markdownRenderTimer);
  state.markdownRenderTimer = setTimeout(() => renderMarkdown(markdown, live), 90);
}

function resetTypewriter() {
  clearTimeout(state.typingTimer);
  state.typingTimer = null;
  state.typingCurrent = '';
  state.typingTarget = '';
  state.typingFinal = false;
  state.typingPage = null;
}

function primeTypewriter(pageNum, text, isFinal = false) {
  clearTimeout(state.typingTimer);
  state.typingTimer = null;
  state.typingPage = pageNum;
  state.typingCurrent = text || '';
  state.typingTarget = state.typingCurrent;
  state.typingFinal = isFinal;
}

function updateTypewriterTarget(text, isFinal = false, pageNum = state.currentPage) {
  // Streaming events for non-visible pages are retained in pageResults only.
  if (pageNum !== state.currentPage) return;
  if (state.typingPage !== pageNum) primeTypewriter(pageNum, '', false);
  const target = text || '';
  if (!target.startsWith(state.typingCurrent)) {
    let common = 0;
    while (common < target.length && common < state.typingCurrent.length && target[common] === state.typingCurrent[common]) common++;
    state.typingCurrent = target.slice(0, common);
  }
  state.typingTarget = target;
  state.typingFinal = state.typingFinal || isFinal;
  if (!state.typingTimer) typeNextCharacter();
}

function typeNextCharacter() {
  // Navigation invalidates the old page's animation immediately. Without this
  // guard, its pending timer would keep repainting over the newly selected page.
  if (state.typingPage !== state.currentPage) {
    clearTimeout(state.typingTimer);
    state.typingTimer = null;
    return;
  }
  if (state.scanPaused) {
    state.typingTimer = setTimeout(typeNextCharacter, 60);
    return;
  }
  if (state.typingCurrent.length < state.typingTarget.length) {
    const codePoint = state.typingTarget.codePointAt(state.typingCurrent.length);
    state.typingCurrent += String.fromCodePoint(codePoint);
    renderMarkdown(state.typingCurrent, true);
    state.typingTimer = setTimeout(typeNextCharacter, 9);
    return;
  }
  state.typingTimer = null;
  renderMarkdown(state.typingCurrent, !state.typingFinal && state.scanning);
}

async function sendScanControl(action) {
  if (!state.sessionId || !state.scanning) return null;
  const r = await fetch('/api/scan-control', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: state.sessionId, action }),
  });
  return r.json();
}

async function toggleScanPause() {
  if (!state.scanning) return;
  const nextPaused = !state.scanPaused;
  const result = await sendScanControl(nextPaused ? 'pause' : 'resume');
  if (!result?.ok) return;
  state.scanPaused = nextPaused;
  $('pause-btn').textContent = nextPaused ? '继续' : '暂停';
  progressText.textContent = nextPaused ? '已暂停' : `继续识别第 ${state.currentPage} 页`;
  progressDetail.textContent = nextPaused ? '模型已停在当前位置' : '正在继续生成';
  document.querySelector('.spinner')?.classList.toggle('paused', nextPaused);
}

async function stopScan() {
  if (!state.scanning) return;
  state.scanStopped = true;
  state.scanPaused = false;
  clearTimeout(state.typingTimer);
  state.typingTimer = null;
  state.typingTarget = state.typingCurrent;
  state.typingFinal = true;
  if (state.resultMode === 'markdown') {
    renderMarkdown(state.typingCurrent, false);
    requestAnimationFrame(scheduleMarkdownFit);
  }
  $('stop-btn').disabled = true;
  progressText.textContent = '正在停止…';
  const controller = state.scanAbortController;
  await sendScanControl('stop');
  setTimeout(() => controller?.abort(), 500);
}

function renderMarkdown(markdown, live = false) {
  const source = markdown || (live ? '_正在读取页面…_' : '');
  if (!source) {
    ocrContent.innerHTML = emptyResult('没有 Markdown 内容', '可切换到「编辑」查看文字块');
    return;
  }
  try {
    const parsed = window.marked ? marked.parse(source, { gfm: true, breaks: true }) : source;
    const safe = window.DOMPurify ? DOMPurify.sanitize(parsed, { USE_PROFILES: { html: true, svg: true, svgFilters: true } }) : parsed;
    ocrContent.classList.add('markdown-preview');
    ocrContent.innerHTML = `<article class="markdown-body${live ? ' is-live' : ''}"><div class="markdown-page-content">${safe}${live ? '<span class="stream-cursor"></span>' : ''}</div></article>`;
    // The typewriter replaces the article on every character. Size the new
    // element immediately so there is never a one-frame auto-sized flash.
    fitMarkdownPreview();
    scheduleMarkdownFit();
    if (!live) enhanceMarkdown(ocrContent.querySelector('.markdown-body'));
  } catch (err) {
    ocrContent.textContent = source;
  }
}

async function enhanceMarkdown(root) {
  if (!root) return;
  const diagrams = [...root.querySelectorAll('pre code.language-mermaid')];
  diagrams.forEach(code => {
    const box = document.createElement('div');
    box.className = 'mermaid';
    box.textContent = code.textContent;
    code.closest('pre').replaceWith(box);
  });
  if (diagrams.length && window.mermaid) {
    try { await mermaid.run({ nodes: [...root.querySelectorAll('.mermaid')] }); } catch { /* incomplete live diagram */ }
  }
  if (window.MathJax?.typesetPromise) {
    try { await MathJax.typesetPromise([root]); } catch { /* incomplete live formula */ }
  }
  scheduleMarkdownFit();
}

let markdownFitFrame = null;
function scheduleMarkdownFit() {
  if (markdownFitFrame !== null) return;
  markdownFitFrame = requestAnimationFrame(() => {
    markdownFitFrame = null;
    fitMarkdownPreview();
  });
}

function fitMarkdownPreview() {
  const page = ocrContent.querySelector('.markdown-body');
  const content = page?.querySelector('.markdown-page-content');
  if (!page || !content || state.resultMode !== 'markdown') return;

  const fittedPage = getFittedPageSize();
  const availableWidth = Math.max(120, ocrContent.clientWidth - 4);
  const topGap = Math.max(0, (ocrContent.clientHeight - fittedPage.height) / 2);
  const availableHeight = Math.max(120, ocrContent.clientHeight - topGap * 2);
  const sourceAspect = srcImg.naturalWidth && srcImg.naturalHeight
    ? srcImg.naturalWidth / srcImg.naturalHeight
    : 0.707;

  // Use the same deterministic A4 box as fitImage(). Reading getBoundingClientRect()
  // here used to capture an intermediate value from the image's CSS transition
  // during page 1 streaming and permanently lock the Markdown canvas too small.
  let pageWidth = state.markdownPageSize?.width || fittedPage.width;
  let pageHeight = state.markdownPageSize?.height || fittedPage.height;
  if (!state.markdownPageSize && (pageWidth > availableWidth || pageHeight > availableHeight)) {
    const sharedScale = Math.min(availableWidth / pageWidth, availableHeight / pageHeight);
    pageWidth *= sharedScale;
    pageHeight *= sharedScale;
  }
  page.style.width = `${Math.floor(pageWidth)}px`;
  page.style.height = `${Math.floor(pageHeight)}px`;
  page.style.marginTop = `${Math.round(topGap)}px`;
  if (!state.markdownPageSize) {
    state.markdownPageSize = { width: pageWidth, height: pageHeight };
  }

  // 小四号 = 12 pt. Convert that physical size onto the displayed A4 sheet:
  // 1 pt = 25.4 / 72 mm; portrait A4 width = 210 mm (landscape = 297 mm).
  // The result therefore follows the paper scale instead of a hard-coded px.
  const a4WidthMm = sourceAspect > 1 ? 297 : 210;
  const smallFourMm = 12 * 25.4 / 72;
  const baseFontSize = pageWidth * smallFourMm / a4WidthMm;
  content.style.transform = '';
  content.style.fontSize = `${baseFontSize}px`;
  content.style.lineHeight = '1.5';
  // Render at physical 小四号 first, then shrink only when the generated
  // Markdown exceeds the same A4 canvas used by the source preview.
  const innerWidth = Math.max(1, pageWidth * 0.89);
  const innerHeight = Math.max(1, pageHeight * 0.91);
  const contentWidth = Math.max(content.scrollWidth, content.offsetWidth);
  const contentHeight = Math.max(content.scrollHeight, content.offsetHeight);
  const renderScale = Math.min(1, innerWidth / contentWidth, innerHeight / contentHeight);
  const fontSize = baseFontSize * Math.max(0.35, renderScale);
  content.style.fontSize = `${fontSize}px`;
  page.dataset.fontSize = fontSize.toFixed(1);
  page.dataset.renderScale = renderScale.toFixed(3);
}

function loadPageImage(pageNum) {
  const url = state.pageImageUrls[pageNum];
  if (url) {
    state.markdownPageSize = null;
    srcImg.classList.remove('loaded');
    srcImg.onload = () => {
      srcImg.classList.add('loaded');
      srcPlaceholder.style.display = 'none';
      fitImage();
      const fittedPage = getFittedPageSize();
      state.markdownPageSize = { width: fittedPage.width, height: fittedPage.height };
      scheduleMarkdownFit();
      setTimeout(scheduleMarkdownFit, 220);
    };
    srcImg.onerror = () => {
      srcImg.classList.remove('loaded');
      srcPlaceholder.style.display = '';
    };
    srcImg.src = url;
  }
}

function goToPage(num, manual = true) {
  if (num < 1 || num > state.totalPages) return;
  if (manual) {
    state.followScanningPage = false;
    const followButton = $('follow-btn');
    followButton.classList.remove('active');
    followButton.setAttribute('aria-pressed', 'false');
    followButton.querySelector('span').textContent = '开启跟随';
  }
  resetTypewriter();
  state.markdownPageSize = null;
  state.currentPage = num;
  pageInd.textContent = `${num} / ${state.totalPages}`;
  prevBtn.disabled = num <= 1;
  nextBtn.disabled = num >= state.totalPages;
  document.querySelectorAll('.page-item').forEach((el, i) => el.classList.toggle('active', i + 1 === num));

  loadPageImage(num);

  const result = state.pageResults[num];
  if (state.resultMode === 'markdown' && result?.markdown !== undefined) {
    primeTypewriter(num, result.markdown, Boolean(result.html));
    renderMarkdown(result.markdown, state.scanning && !result.html);
  } else if (result?.html) {
    renderOCRContent(result.html);
    if (result.translated) applyTranslations(result.translated);
  } else if (result?.detections?.length) {
    // Page was being scanned (partial), show what we have
    ocrContent.innerHTML = `<div class="ocr-page" data-page="${num}"></div>`;
    const container = ocrContent.querySelector(`.ocr-page[data-page="${num}"]`);
    result.detections.forEach((det, i) => {
      const detHtml = buildDetHtml(det, i);
      container.insertAdjacentHTML('beforeend', detHtml);
    });
  } else {
    ocrContent.innerHTML = emptyResult('此页尚未识别', '点击「开始识别」后结果会显示在这里');
  }
  updatePageWarning();
  updateResultStats(num);
}

function toggleFollowScanning() {
  state.followScanningPage = !state.followScanningPage;
  const button = $('follow-btn');
  button.classList.toggle('active', state.followScanningPage);
  button.setAttribute('aria-pressed', String(state.followScanningPage));
  button.querySelector('span').textContent = state.followScanningPage ? '自动跟随' : '开启跟随';
  if (state.followScanningPage && state.scanningPage && state.scanningPage !== state.currentPage) {
    goToPage(state.scanningPage, false);
  }
}

function closeExportMenu() {
  exportMenuBtn?.setAttribute('aria-expanded', 'false');
}

function toggleExportMenu(e) {
  e.stopPropagation();
  const open = exportMenuBtn.getAttribute('aria-expanded') === 'true';
  exportMenuBtn.setAttribute('aria-expanded', String(!open));
}

function buildDetHtml(det, index) {
  const t = det.type;
  const text = det.text || '';
  const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  if (t === 'title') {
    const bbox = det.bbox || [];
    const h = (bbox[3] - bbox[1]) || 0;
    const tag = h > 60 ? 'h1' : 'h2';
    return `<${tag} class="ocr-heading" contenteditable="true" data-detection-index="${index}">${esc(text)}</${tag}>`;
  }
  if (t === 'image') return `<div class="ocr-image" data-detection-index="${index}"><span class="image-placeholder">🖼 图片区域</span></div>`;
  if (t === 'table') return `<div class="ocr-table" contenteditable="true" data-detection-index="${index}">${esc(text)}</div>`;
  if (t === 'page_number') return `<span class="ocr-page-number" data-detection-index="${index}">${esc(text)}</span>`;
  return `<p class="ocr-text" contenteditable="true" data-detection-index="${index}">${esc(text)}</p>`;
}

/* ── Inline Edit ────────────────────────────────── */
function attachEditListeners() {
  ocrContent.querySelectorAll('[contenteditable="true"]').forEach(el => {
    el.addEventListener('blur', () => {
      const idx = parseInt(el.dataset.detectionIndex);
      const page = state.currentPage;
      const result = state.pageResults[page];
      if (!result?.detections?.[idx]) return;
      const original = result.detections[idx].text || '';
      const newText = el.innerText.trim();
      if (newText !== original && newText !== '') saveEdit(page, idx, newText);
    });
    el.addEventListener('keydown', e => {
      if (e.key === 'Enter' && el.tagName === 'P') { e.preventDefault(); el.blur(); }
    });
  });
}

async function saveEdit(pageNum, detectionIndex, newText) {
  try {
    await fetch('/api/edit', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId, page_num: pageNum, detection_index: detectionIndex, new_text: newText }),
    });
    const result = state.pageResults[pageNum];
    if (result?.detections?.[detectionIndex]) result.detections[detectionIndex].text = newText;
    showToast('已保存');
  } catch { showToast('保存失败'); }
}

/* ── Translate (bilingual) ──────────────────────── */
async function translatePage() {
  const lang = state.targetLang;
  if (!lang || !state.sessionId) return;
  const page = state.currentPage;
  const result = state.pageResults[page];
  if (!result?.detections?.length) { showToast('此页未扫描'); return; }

  // Toggle off if already translated
  if (result.translated) {
    delete result.translated;
    if (result.html) renderOCRContent(result.html);
    else ocrContent.innerHTML = '<p class="placeholder">无结果</p>';
    transBtn.textContent = '🌐 翻译';
    return;
  }

  transBtn.disabled = true;
  transBtn.textContent = '⏳ 翻译中...';
  statusText.textContent = `翻译第 ${page} 页...`;

  try {
    const r = await fetch('/api/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: state.sessionId,
        page_num: page,
        source_lang: 'auto',
        target_lang: lang,
        detections: result.detections,
      }),
    });
    const d = await r.json();
    if (d.error) { showToast(d.error); return; }

    state.pageResults[page].translated = d.translated_detections;
    applyTranslations(d.translated_detections);
    transBtn.textContent = '🌐 取消翻译';
    showToast('翻译完成');
  } catch (err) {
    showToast('翻译失败: ' + err.message);
  } finally {
    transBtn.disabled = false;
    statusText.textContent = '就绪';
  }
}

function applyTranslations(translations) {
  if (!translations) return;
  ocrContent.querySelectorAll('.translated-text').forEach(el => el.remove());

  for (const t of translations) {
    const el = ocrContent.querySelector(`[data-detection-index="${t.index}"]`);
    if (el && t.translated && t.translated !== t.original) {
      // Wrap original + translation in a pair container for tight layout
      const pair = document.createElement('div');
      pair.className = 'ocr-pair';
      el.parentNode.insertBefore(pair, el);
      pair.appendChild(el);
      const div = document.createElement('div');
      div.className = 'translated-text';
      div.textContent = t.translated;
      pair.appendChild(div);
    }
  }
}

/* ── Export with mode selection ──────────────────── */
function showExportDialog() {
  const hasTranslation = Object.values(state.pageResults).some(r => r.translated);

  if (!hasTranslation) {
    doExport('original');
    return;
  }

  const existing = document.getElementById('export-dialog');
  if (existing) existing.remove();

  const dialog = document.createElement('div');
  dialog.id = 'export-dialog';
  dialog.innerHTML = `
    <div class="export-dialog-content">
      <h3>选择导出模式</h3>
      <label><input type="radio" name="export-mode" value="original" checked> 仅原文</label>
      <label><input type="radio" name="export-mode" value="translated"> 仅译文</label>
      <label><input type="radio" name="export-mode" value="bilingual"> 双语对照（一行原文一行译文）</label>
      <div class="export-dialog-actions">
        <button id="export-confirm" class="btn primary">确认导出</button>
        <button id="export-cancel" class="btn">取消</button>
      </div>
    </div>
  `;
  document.body.appendChild(dialog);

  document.getElementById('export-confirm').onclick = () => {
    const mode = document.querySelector('input[name="export-mode"]:checked').value;
    dialog.remove();
    doExport(mode);
  };
  document.getElementById('export-cancel').onclick = () => dialog.remove();
}

async function doExport(mode) {
  if (!state.sessionId) return;
  exportBtn.disabled = true;
  const exportLabel = exportBtn.querySelector('span');
  exportLabel.textContent = '导出中';
  statusText.textContent = '生成 Word 文档...';

  try {
    const r = await fetch('/api/export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId, export_mode: mode }),
    });
    if (!r.ok) throw new Error('导出失败');
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ocr_result_${mode}.docx`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('导出成功!');
  } catch (err) {
    showToast('导出失败: ' + err.message);
  }

  exportBtn.disabled = false;
  exportLabel.textContent = '导出 Word';
  statusText.textContent = '就绪';
}

async function exportMarkdown() {
  if (!state.sessionId) return;
  exportMenuBtn.disabled = true;
  statusText.textContent = '生成 Markdown...';
  try {
    const r = await fetch('/api/export-markdown', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId }),
    });
    if (!r.ok) throw new Error('导出失败');
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${state.sourceName.replace(/\.[^.]+$/, '')}_ocr_markdown.zip`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('Markdown 与图片包已下载');
  } catch (err) { showToast('Markdown 导出失败: ' + err.message); }
  exportMenuBtn.disabled = false;
  statusText.textContent = '就绪';
}

/* ── Divider drag ───────────────────────────────── */
function setupDivider() {
  let isDragging = false;
  const left = $('left-panel');
  const right = $('right-panel');

  divider.addEventListener('pointerdown', e => {
    isDragging = true;
    divider.classList.add('active');
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    divider.setPointerCapture?.(e.pointerId);
    e.preventDefault();
  });

  document.addEventListener('pointermove', e => {
    if (!isDragging) return;
    const leftRect = left.getBoundingClientRect();
    const rightRect = right.getBoundingClientRect();
    // Use the actual pointer-to-panel distance in CSS pixels. Percentage flex
    // bases are relative to the whole workspace (including the sidebar), so
    // they introduced a persistent offset while dragging.
    const width = e.clientX - leftRect.left;
    const minWidth = 220;
    const maxWidth = Math.max(minWidth, rightRect.right - leftRect.left - 220);
    left.style.flex = `0 0 ${Math.max(minWidth, Math.min(maxWidth, width))}px`;
    right.style.flex = '1';
  });

  document.addEventListener('pointerup', () => {
    if (!isDragging) return;
    isDragging = false;
    divider.classList.remove('active');
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  });
}

/* ── Keyboard shortcuts ─────────────────────────── */
function setupKeyboard() {
  document.addEventListener('keydown', e => {
    if (e.target.isContentEditable) return;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { e.preventDefault(); goToPage(state.currentPage - 1); }
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { e.preventDefault(); goToPage(state.currentPage + 1); }
  });
}

/* ── Modern workspace helpers ───────────────────── */
function renderPageList() {
  if (!state.totalPages) return;
  pageList.innerHTML = '';
  for (let page = 1; page <= state.totalPages; page++) {
    const button = document.createElement('button');
    button.className = `page-item${page === state.currentPage ? ' active' : ''}`;
    button.dataset.page = page;
    button.innerHTML = `<img class="page-thumb" src="${state.pageImageUrls[page]}" alt="第 ${page} 页缩略图"><span class="page-item-meta"><strong>第 ${page} 页</strong><span>等待识别</span></span>`;
    button.addEventListener('click', () => goToPage(page));
    pageList.appendChild(button);
  }
}

function setPageStatus(page, status) {
  const item = pageList.querySelector(`[data-page="${page}"]`);
  if (!item) return;
  item.classList.toggle('done', status === 'done');
  item.classList.toggle('warning', status === 'warning');
  item.classList.toggle('failed', status === 'failed');
  const label = item.querySelector('.page-item-meta span');
  label.textContent =
    status === 'done' ? '识别完成'
    : status === 'warning' ? '结果可能不完整'
    : status === 'failed' ? '识别失败'
    : status === 'scanning' ? '正在识别…' : '等待识别';
}

// Persistent truncation warning banner that follows the page result.
function updatePageWarning() {
  const warning = $('page-warning');
  if (!warning) return;
  const result = state.pageResults[state.currentPage];
  const show = Boolean(result && result.truncated);
  warning.classList.toggle('hidden', !show);
  if (show) $('page-warning-text').textContent = '⚠ 本页达到 OCR 最大生成长度，结果可能不完整';
}

function setupDropZone() {
  let dragDepth = 0;
  sourceView.addEventListener('dragenter', e => { e.preventDefault(); dragDepth++; sourceView.classList.add('dragging'); });
  sourceView.addEventListener('dragover', e => e.preventDefault());
  sourceView.addEventListener('dragleave', () => { dragDepth--; if (dragDepth <= 0) { dragDepth = 0; sourceView.classList.remove('dragging'); } });
  sourceView.addEventListener('drop', e => {
    e.preventDefault(); dragDepth = 0; sourceView.classList.remove('dragging');
    const file = e.dataTransfer.files[0];
    if (file) handleUpload(file);
  });
}

function setZoom(value) {
  // High-DPI PDF pages often need 15–30% scale to fit vertically.
  state.zoom = Math.max(.1, Math.min(2.5, value));
  if (srcImg.naturalWidth) {
    const width = Math.floor(srcImg.naturalWidth * state.zoom);
    const height = Math.floor(srcImg.naturalHeight * state.zoom);
    srcImg.style.width = `${width}px`;
    srcImg.style.height = `${height}px`;
    const overflow = width > sourceView.clientWidth - 24 || height > sourceView.clientHeight - 24;
    sourceView.classList.toggle('zoomed', overflow);
  }
  zoomLevel.textContent = `${Math.round(state.zoom * 100)}%`;
}

function fitImage() {
  if (!srcImg.naturalWidth || !sourceView.clientWidth) return;
  // CSS pixels are used on both sides, so Retina DPI does not affect this.
  // A generous inset guarantees that both the top and bottom page edges remain visible.
  // Both panels display the same page at the same pixel dimensions. This is
  // slightly smaller than fitting each panel independently, but aligns all
  // four paper edges even when the panels have different widths.
  const { width: pageWidth, height: pageHeight } = getFittedPageSize();
  srcImg.style.width = `${pageWidth}px`;
  srcImg.style.height = `${pageHeight}px`;
  state.zoom = pageWidth / srcImg.naturalWidth;
  zoomLevel.textContent = '适应';
  sourceView.classList.remove('zoomed');
  sourceView.scrollTop = 0;
  sourceView.scrollLeft = 0;
}

function getFittedPageSize() {
  const sharedRightWidth = ocrContent?.clientWidth ? ocrContent.clientWidth - 4 : Infinity;
  const availableW = Math.max(1, Math.min(sourceView.clientWidth - 40, sharedRightWidth));
  const availableH = Math.max(1, sourceView.clientHeight - 40);
  // PDF pages may be rasterized at different pixel densities; their physical
  // preview remains A4 so every page gets an identical CSS-pixel canvas.
  const pageAspect = 595.32 / 842.04;
  const width = Math.floor(Math.min(availableW, availableH * pageAspect) * 0.98);
  return { width, height: Math.floor(width / pageAspect) };
}

function toggleTheme() {
  const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  localStorage.setItem('ocr-theme', next);
  updateThemeButton();
  if (window.mermaid) mermaid.initialize({ startOnLoad: false, securityLevel: 'strict', theme: next === 'dark' ? 'dark' : 'default' });
  if (state.resultMode === 'markdown') renderMarkdown(state.pageResults[state.currentPage]?.markdown || '', state.scanning);
}

function updateThemeButton() {
  const dark = document.documentElement.dataset.theme === 'dark';
  const button = $('theme-btn');
  button.title = dark ? '切换浅色模式' : '切换深色模式';
  button.setAttribute('aria-label', button.title);
  button.innerHTML = dark
    ? '<svg class="moon-icon" viewBox="0 0 24 24"><path d="M20 15.5A8.5 8.5 0 0 1 8.5 4 8.5 8.5 0 1 0 20 15.5z"/></svg>'
    : '<svg class="sun-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.42 1.42M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.42-1.42M17.66 6.34l1.41-1.41"/></svg>';
}

async function copyResult() {
  const text = state.resultMode === 'markdown'
    ? (state.pageResults[state.currentPage]?.markdown || '').trim()
    : ocrContent.innerText.trim();
  if (!text) return;
  try { await navigator.clipboard.writeText(text); showToast('识别结果已复制'); }
  catch { showToast('复制失败，请手动选择文字'); }
}

function updateResultStats(page = state.currentPage) {
  const count = state.pageResults[page]?.detections?.length || 0;
  const text = (state.pageResults[page]?.detections || []).map(d => d.text || '').join('');
  if (resultStats) resultStats.textContent = count ? `${count} 个文本块 · ${text.length} 字符` : '0 个文本块';
}

function startTimer() {
  state.scanStartedAt = Date.now();
  elapsedTime.textContent = '00:00';
  clearInterval(state.timerId);
  state.timerId = setInterval(() => {
    const seconds = Math.floor((Date.now() - state.scanStartedAt) / 1000);
    elapsedTime.textContent = `${String(Math.floor(seconds / 60)).padStart(2,'0')}:${String(seconds % 60).padStart(2,'0')}`;
  }, 1000);
}

function stopTimer() { clearInterval(state.timerId); state.timerId = null; }
function formatBytes(bytes) { return bytes < 1024 * 1024 ? `${Math.round(bytes / 1024)} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`; }
function emptyResult(title, copy) { return `<div class="result-empty"><div class="result-empty-icon"><svg viewBox="0 0 24 24"><path d="M4 6h16M4 12h10M4 18h13"/></svg></div><h2>${title}</h2><p>${copy}</p></div>`; }

/* ── Toast ──────────────────────────────────────── */
let toastTimer;
function showToast(msg) {
  toast.textContent = msg;
  toast.classList.remove('hidden');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.add('hidden'), 2500);
}
