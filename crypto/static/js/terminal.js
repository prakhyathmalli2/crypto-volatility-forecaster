/**
 * ALPHAVOL PRO v3.5 - Institutional Multi-Asset Market Intelligence & Quantitative Platform
 * Architecture: Clean Multi-Page Trading Workspaces (One Screen = One Primary Job)
 * Pages:
 * 1. Market Overview (Global Indices, Heatmap, Movers, Headlines)
 * 2. Trading Terminal (TV Lightweight Charts, Watchlist, Oscillators, Trade Intelligence, AI Drawer)
 * 3. Asset Research (Overview, Fundamentals, Technical Readout, News, Events, AI Report)
 * 4. News & Events (Prioritized News Feed + Macro Economic Calendar)
 * 5. AI Research Assistant (Dedicated Quantitative Chat Console + Live Context Inspector)
 * 6. Paper Trading (Simulated Execution Desk, ATR Sizing, Open Positions Ledger)
 * 7. Backtest Lab (Strategy Simulation, Equity Curves, Drawdowns, Trade Log, Model Matrix)
 * 8. Trading Principles (Quantitative Discipline Handbook & Risk Axioms)
 */

(function () {
  'use strict';

  // --- Constants & Config ---
  const API_BASE = '';
  const TIMEFRAME_SECONDS = {
    '1m': 60,
    '5m': 300,
    '15m': 900,
    '30m': 1800,
    '1h': 3600,
    '4h': 14400,
    '1d': 86400,
    '1w': 604800,
  };

  // --- Authoritative State Store ---
  const state = {
    activeSymbol: 'BTC/USDT',
    activeTimeframe: '1h',
    chartType: 'candles', // 'candles' | 'line'
    subChartType: 'rsi',  // 'rsi' | 'macd' | 'stoch' | 'none'
    currentPage: 'overview',
    watchlistCategory: 'ALL',
    favorites: new Set(['BTC/USDT', 'RELIANCE.NS', 'AAPL', '^NSEI']),
    
    // Static Instrument Catalog
    instruments: [
      { symbol: 'BTC/USDT', name: 'Bitcoin', type: 'crypto', exchange: 'Binance', currency: 'USDT' },
      { symbol: 'ETH/USDT', name: 'Ethereum', type: 'crypto', exchange: 'Binance', currency: 'USDT' },
      { symbol: 'SOL/USDT', name: 'Solana', type: 'crypto', exchange: 'Binance', currency: 'USDT' },
      { symbol: 'BNB/USDT', name: 'BNB', type: 'crypto', exchange: 'Binance', currency: 'USDT' },
      { symbol: 'XRP/USDT', name: 'XRP', type: 'crypto', exchange: 'Binance', currency: 'USDT' },
      { symbol: 'RELIANCE.NS', name: 'Reliance Industries', type: 'indian_stock', exchange: 'NSE', currency: 'INR' },
      { symbol: 'TCS.NS', name: 'Tata Consultancy Services', type: 'indian_stock', exchange: 'NSE', currency: 'INR' },
      { symbol: 'HDFCBANK.NS', name: 'HDFC Bank', type: 'indian_stock', exchange: 'NSE', currency: 'INR' },
      { symbol: 'INFY.NS', name: 'Infosys Ltd', type: 'indian_stock', exchange: 'NSE', currency: 'INR' },
      { symbol: 'AAPL', name: 'Apple Inc.', type: 'global_stock', exchange: 'NASDAQ', currency: 'USD' },
      { symbol: 'NVDA', name: 'NVIDIA Corporation', type: 'global_stock', exchange: 'NASDAQ', currency: 'USD' },
      { symbol: 'TSLA', name: 'Tesla Inc.', type: 'global_stock', exchange: 'NASDAQ', currency: 'USD' },
      { symbol: 'MSFT', name: 'Microsoft Corporation', type: 'global_stock', exchange: 'NASDAQ', currency: 'USD' },
      { symbol: '^NSEI', name: 'NIFTY 50', type: 'index', exchange: 'NSE', currency: 'INR' },
      { symbol: '^BSESN', name: 'S&P BSE SENSEX', type: 'index', exchange: 'BSE', currency: 'INR' },
      { symbol: '^GSPC', name: 'S&P 500', type: 'index', exchange: 'INDEX', currency: 'USD' },
      { symbol: '^IXIC', name: 'NASDAQ Composite', type: 'index', exchange: 'INDEX', currency: 'USD' },
      { symbol: 'USDINR=X', name: 'USD / INR', type: 'forex', exchange: 'FX', currency: 'INR' },
      { symbol: 'EURUSD=X', name: 'EUR / USD', type: 'forex', exchange: 'FX', currency: 'USD' },
      { symbol: 'GBPUSD=X', name: 'GBP / USD', type: 'forex', exchange: 'FX', currency: 'USD' },
    ],

    tickers: {},
    candles: [],
    indicators: null,
    signal: null,
    fundamentals: null,
    news: [],
    events: [],
    paperPortfolio: null,
    paperApiKey: '', // Volatile in-memory credentials for protected mutations (never persisted)
    backtestData: null,

    // Overlay Indicator Toggles for Terminal Chart
    indicatorToggles: {
      ema20: true,
      ema50: true,
      ema200: false,
      sma50: false,
      bb: false,
      vwap: false,
    },
    markerToggles: {
      news: true,
      events: true,
      signals: true,
    },

    searchSelectedIdx: -1,
  };

  // --- Chart Instance References ---
  let mainChart = null;
  let candleSeries = null;
  let lineSeries = null;
  let volumeSeries = null;
  let overlaySeries = {};

  let subChart = null;
  let subSeries = {};
  let isSyncingRange = false;

  // --- WebSocket References ---
  let marketWs = null;
  let signalsWs = null;
  let wsReconnectTimer = null;
  let isWsConnected = false;

  // --- Utility Functions ---
  function formatCurrency(val, currency = 'USD') {
    if (val === null || val === undefined || isNaN(val)) return '--';
    const num = Number(val);
    const prefix = currency === 'INR' ? '₹' : (currency === 'EUR' ? '€' : (currency === 'GBP' ? '£' : '$'));
    if (Math.abs(num) >= 1000) {
      return prefix + num.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    } else if (Math.abs(num) >= 1) {
      return prefix + num.toFixed(2);
    } else {
      return prefix + num.toFixed(4);
    }
  }

  function formatChange(pct) {
    if (pct === null || pct === undefined || isNaN(pct)) return '--%';
    const sign = pct >= 0 ? '+' : '';
    return `${sign}${Number(pct).toFixed(2)}%`;
  }

  function formatVolume(vol) {
    if (!vol || isNaN(vol)) return '--';
    const num = Number(vol);
    if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B';
    if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
    if (num >= 1e3) return (num / 1e3).toFixed(2) + 'K';
    return num.toFixed(0);
  }

  function sanitize(str) {
    if (!str) return '';
    return String(str).replace(/[&<>"']/g, function (m) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m];
    });
  }

  function formatIST(dateObj) {
    if (!dateObj) return '--';
    try {
      const d = typeof dateObj === 'string' ? new Date(dateObj) : dateObj;
      if (isNaN(d.getTime())) return String(dateObj);
      return d.toLocaleString('en-IN', {
        timeZone: 'Asia/Kolkata',
        day: '2-digit',
        month: 'short',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false,
      }) + ' IST';
    } catch (e) {
      return String(dateObj);
    }
  }

  /**
   * Convert UTC timestamp (seconds or ms) to intended display timezone representation for Lightweight Charts.
   * Lightweight Charts renders time internally using UTC accessors (getUTCHours, etc.).
   * By projecting UTC epoch time into target display timezone components (IST / Asia/Kolkata) and creating
   * a UTC representation of those local components, Lightweight Charts displays the exact local/IST hours,
   * minutes, and calendar dates without hard-coding static offsets.
   */
  function toChartTime(utcSecOrMs, timeZone = 'Asia/Kolkata') {
    if (utcSecOrMs === undefined || utcSecOrMs === null) return utcSecOrMs;
    const sec = typeof utcSecOrMs === 'number' && utcSecOrMs > 1e11
      ? Math.floor(utcSecOrMs / 1000)
      : Number(utcSecOrMs);
    if (isNaN(sec)) return utcSecOrMs;

    try {
      const d = new Date(sec * 1000);
      const targetTz = timeZone || 'Asia/Kolkata';
      const formatter = new Intl.DateTimeFormat('en-US', {
        timeZone: targetTz,
        year: 'numeric',
        month: 'numeric',
        day: 'numeric',
        hour: 'numeric',
        minute: 'numeric',
        second: 'numeric',
        hour12: false,
      });
      const parts = formatter.formatToParts(d);
      const map = {};
      for (const p of parts) map[p.type] = p.value;
      let hr = parseInt(map.hour, 10);
      if (hr === 24) hr = 0;
      return Date.UTC(
        parseInt(map.year, 10),
        parseInt(map.month, 10) - 1,
        parseInt(map.day, 10),
        hr,
        parseInt(map.minute, 10),
        parseInt(map.second, 10)
      ) / 1000;
    } catch (_) {
      const d = new Date(sec * 1000);
      return Date.UTC(
        d.getFullYear(),
        d.getMonth(),
        d.getDate(),
        d.getHours(),
        d.getMinutes(),
        d.getSeconds()
      ) / 1000;
    }
  }

  function formatChartCrosshairTime(time) {
    if (typeof time === 'number') {
      const d = new Date(time * 1000);
      const day = String(d.getUTCDate()).padStart(2, '0');
      const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
      const month = months[d.getUTCMonth()];
      const year = String(d.getUTCFullYear()).slice(2);
      const hours = String(d.getUTCHours()).padStart(2, '0');
      const minutes = String(d.getUTCMinutes()).padStart(2, '0');
      const isIntraday = !['1d', '1w'].includes(state.activeTimeframe);
      return isIntraday ? `${day} ${month} '${year} ${hours}:${minutes} IST` : `${day} ${month} '${year}`;
    }
    return String(time);
  }

  function formatChartTickMark(time, tickMarkType, locale) {
    if (typeof time !== 'number') return null;
    const d = new Date(time * 1000);
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    if (tickMarkType === 0) return String(d.getUTCFullYear());
    if (tickMarkType === 1) return months[d.getUTCMonth()];
    if (tickMarkType === 2) return `${d.getUTCDate()} ${months[d.getUTCMonth()]}`;
    if (tickMarkType === 3 || tickMarkType === 4) {
      const hours = String(d.getUTCHours()).padStart(2, '0');
      const minutes = String(d.getUTCMinutes()).padStart(2, '0');
      return `${hours}:${minutes}`;
    }
    return null;
  }

  // --- Authoritative Data Fetching Layer (Route-based & Lazy) ---
  const API = {
    async get(endpoint) {
      try {
        const res = await fetch(API_BASE + endpoint);
        if (!res.ok) {
          let errDetail = `HTTP ${res.status}`;
          try {
            const errBody = await res.json();
            if (errBody && errBody.detail) errDetail = errBody.detail;
          } catch (e) {
            console.warn(`[API JSON parse error] ${endpoint}:`, e);
          }
          console.warn(`[API GET Warning] ${endpoint} returned ${res.status}: ${errDetail}`);
          return { ok: false, data: null, error: errDetail };
        }
        const data = await res.json();
        return { ok: true, data, error: null };
      } catch (err) {
        console.warn(`[API GET Error] ${endpoint}:`, err);
        return { ok: false, data: null, error: err.message || 'Network request failed' };
      }
    },
    async post(endpoint, data, customHeaders = {}) {
      try {
        const headers = { 'Content-Type': 'application/json', ...customHeaders };
        const res = await fetch(API_BASE + endpoint, {
          method: 'POST',
          headers,
          body: JSON.stringify(data),
        });
        if (!res.ok) {
          let errDetail = `HTTP ${res.status}`;
          try {
            const errBody = await res.json();
            if (errBody && errBody.detail) errDetail = errBody.detail;
          } catch (e) {
            console.warn(`[API POST JSON parse error] ${endpoint}:`, e);
          }
          const error = new Error(errDetail);
          error.status = res.status;
          throw error;
        }
        return await res.json();
      } catch (err) {
        console.warn(`[API POST Failed] ${endpoint}:`, err);
        throw err;
      }
    },
  };

  // ========================================================================
  // ROUTING & WORKSPACE NAVIGATION
  // ========================================================================
  function initRouting() {
    window.addEventListener('hashchange', handleRoute);
    
    // Main navigation clicks
    document.querySelectorAll('#mainNav .nav-item').forEach(item => {
      item.addEventListener('click', (e) => {
        const page = item.getAttribute('data-page');
        if (page) {
          switchPage(page);
        }
      });
    });

    // Initial page resolution
    const initialHash = window.location.hash.replace('#', '') || 'overview';
    switchPage(initialHash);
  }

  function handleRoute() {
    const hash = window.location.hash.replace('#', '') || 'overview';
    // Support query params e.g. #terminal?symbol=NVDA
    const [page, queryString] = hash.split('?');
    if (queryString) {
      const params = new URLSearchParams(queryString);
      const sym = params.get('symbol');
      if (sym && sym !== state.activeSymbol) {
        setActiveSymbol(sym, false);
      }
    }
    switchPage(page || 'overview');
  }

  function switchPage(pageName) {
    const validPages = ['overview', 'terminal', 'research', 'news', 'ai', 'paper', 'backtest', 'principles'];
    const targetPage = validPages.includes(pageName) ? pageName : 'overview';
    state.currentPage = targetPage;

    // Update Top Navigation
    document.querySelectorAll('#mainNav .nav-item').forEach(link => {
      if (link.getAttribute('data-page') === targetPage) {
        link.classList.add('active');
      } else {
        link.classList.remove('active');
      }
    });

    // Update Page View Visibility
    document.querySelectorAll('.page-view').forEach(view => {
      view.classList.remove('active');
    });
    const targetEl = document.getElementById(`page-${targetPage}`);
    if (targetEl) {
      targetEl.classList.add('active');
    }

    // Update URL hash without causing loop
    if (window.location.hash.replace('#', '').split('?')[0] !== targetPage) {
      window.history.replaceState(null, '', `#${targetPage}`);
    }

    // Lazy load data for the active workspace
    triggerWorkspaceLoader(targetPage);
  }

  function triggerWorkspaceLoader(page) {
    if (page !== 'terminal') {
      closeMarketWs(true);
    }
    switch (page) {
      case 'overview':
        loadOverviewData();
        break;
      case 'terminal':
        initOrResizeTerminalChart();
        loadTerminalCandles();
        loadTerminalSignal();
        ensureWebSockets();
        break;
      case 'research':
        loadResearchData();
        break;
      case 'news':
        loadNewsEventsPage();
        break;
      case 'ai':
        updateAiContextPanel();
        break;
      case 'paper':
        loadPaperData();
        break;
      case 'backtest':
        loadBacktestData();
        break;
      case 'principles':
        loadPrinciplesData();
        break;
    }
  }

  function setActiveSymbol(symbol, switchWorkspace = false) {
    if (!symbol) return;
    state.activeSymbol = symbol;
    updateHeaderPill();

    // Invalidate dependent caches
    state.candles = [];
    state.indicators = null;
    state.signal = null;
    state.fundamentals = null;

    // Immediately hydrate spot ticker for active symbol across all components
    API.get(`/api/v1/market/${encodeURIComponent(symbol)}/ticker`).then(res => {
      if (res && res.ok && res.data) {
        handleLiveTick(res.data);
      }
    });

    // If currently on Terminal, reload chart & signals
    if (state.currentPage === 'terminal') {
      loadTerminalCandles();
      loadTerminalSignal();
      ensureWebSockets();
    } else if (state.currentPage === 'research') {
      loadResearchData();
    } else if (state.currentPage === 'ai') {
      updateAiContextPanel();
    } else if (state.currentPage === 'paper') {
      document.getElementById('paperOrderSymbol').value = symbol;
      updatePaperFormMarkPrice();
    }

    if (switchWorkspace) {
      switchPage('terminal');
    }
  }

  // ========================================================================
  // HEADER TELEMETRY & CLOCK
  // ========================================================================
  function initHeader() {
    updateHeaderPill();
    startClock();

    // Click header pill -> open Terminal
    const pill = document.getElementById('headerActivePill');
    if (pill) {
      pill.addEventListener('click', () => {
        switchPage('terminal');
      });
    }

    // Status modal toggle
    const statusBtn = document.getElementById('openStatusBtn');
    const statusModal = document.getElementById('statusModal');
    const closeStatusModal = document.getElementById('closeStatusModal');
    if (statusBtn && statusModal) {
      statusBtn.addEventListener('click', async () => {
        statusModal.style.display = 'flex';
        const body = document.getElementById('statusModalBody');
        body.innerHTML = '<p>Querying cluster telemetry...</p>';
        const res = await API.get('/api/v1/system/status');
        if (res && res.ok && res.data) {
          const data = res.data;
          body.innerHTML = `
            <div style="font-family: var(--font-mono); font-size: 11px; display: flex; flex-direction: column; gap: 8px;">
              <div>Status: <strong class="text-success">${data.status}</strong></div>
              <div>Timestamp: ${data.timestamp_utc}</div>
              <hr style="border-color: var(--border-subtle);" />
              <div>Market API: <strong>${data.services?.market_api?.status || 'CONNECTED'}</strong> (${data.services?.market_api?.exchange || 'binance'})</div>
              <div>ML Models: <strong>${data.services?.ml_models?.status || 'READY'}</strong> [${(data.services?.ml_models?.models_loaded || []).join(', ')}]</div>
              <div>Signal Engine: <strong>${data.services?.signal_engine?.status || 'READY'}</strong> (${data.services?.signal_engine?.version || 'v2.5'})</div>
              <div>AI Assistant: <strong>${data.services?.ai_assistant?.status || 'READY'}</strong></div>
              <div>WebSockets: <strong>${data.services?.websockets?.status || 'ACTIVE'}</strong></div>
              <div>Paper Broker: <strong>${data.services?.paper_trading?.status || 'ACTIVE'}</strong></div>
            </div>
          `;
        } else {
          body.innerHTML = `<div class="error-state" style="padding: 20px; text-align: center; color: var(--text-muted);">
            Failed to query cluster telemetry (${sanitize(res?.error || 'Unavailable')}).
          </div>`;
        }
      });
    }
    if (closeStatusModal && statusModal) {
      closeStatusModal.addEventListener('click', () => {
        statusModal.style.display = 'none';
      });
      statusModal.addEventListener('click', (e) => {
        if (e.target === statusModal) statusModal.style.display = 'none';
      });
    }
  }

  function updateHeaderPill() {
    const symEl = document.getElementById('headerPillSymbol');
    const priceEl = document.getElementById('headerPillPrice');
    const chgEl = document.getElementById('headerPillChange');
    const statusEl = document.getElementById('headerPillStatus');

    if (symEl) symEl.textContent = state.activeSymbol;

    const tick = state.tickers[state.activeSymbol];
    if (tick && priceEl && chgEl) {
      priceEl.textContent = formatCurrency(tick.price, tick.currency);
      chgEl.textContent = formatChange(tick.change_24h_pct);
      chgEl.className = 'pill-change ' + ((tick.change_24h_pct || 0) >= 0 ? 'text-success' : 'text-danger');
      if (statusEl) {
        statusEl.textContent = tick.market_status || 'OPEN';
        statusEl.className = 'pill-status ' + (tick.market_status === 'CLOSED' ? 'status-closed' : 'status-open');
      }
    }
  }

  function startClock() {
    const clockEl = document.getElementById('headerClock');
    function tick() {
      const now = new Date();
      if (clockEl) {
        const timeStr = now.toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour12: false });
        const dateStr = now.toLocaleDateString('en-IN', { timeZone: 'Asia/Kolkata', day: '2-digit', month: 'short' });
        clockEl.textContent = `${dateStr} • ${timeStr} IST`;
        clockEl.title = `UTC: ${now.toUTCString()} • Presentation in IST (Asia/Kolkata)`;
      }
    }
    tick();
    setInterval(tick, 1000);
  }

  // ========================================================================
  // GLOBAL SEARCH WITH KEYBOARD NAVIGATION
  // ========================================================================
  function initGlobalSearch() {
    const input = document.getElementById('globalSearchInput');
    const dropdown = document.getElementById('searchResultsDropdown');
    const clearBtn = document.getElementById('searchClearBtn');

    if (!input || !dropdown) return;

    let debounceTimer = null;

    input.addEventListener('input', (e) => {
      const q = e.target.value.trim();
      clearBtn.style.display = q ? 'block' : 'none';
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        performSearch(q);
      }, 150);
    });

    clearBtn.addEventListener('click', () => {
      input.value = '';
      clearBtn.style.display = 'none';
      dropdown.style.display = 'none';
      input.focus();
    });

    // Keyboard navigation
    input.addEventListener('keydown', (e) => {
      const items = dropdown.querySelectorAll('.search-item');
      if (items.length === 0) return;

      if (e.key === 'ArrowDown') {
        e.preventDefault();
        state.searchSelectedIdx = (state.searchSelectedIdx + 1) % items.length;
        highlightSearchItem(items);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        state.searchSelectedIdx = (state.searchSelectedIdx - 1 + items.length) % items.length;
        highlightSearchItem(items);
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (state.searchSelectedIdx >= 0 && items[state.searchSelectedIdx]) {
          items[state.searchSelectedIdx].click();
        } else if (items[0]) {
          items[0].click();
        }
      } else if (e.key === 'Escape') {
        dropdown.style.display = 'none';
      }
    });

    // Close on click outside
    document.addEventListener('click', (e) => {
      if (!input.contains(e.target) && !dropdown.contains(e.target)) {
        dropdown.style.display = 'none';
      }
    });

    async function performSearch(q) {
      if (!q) {
        dropdown.style.display = 'none';
        return;
      }
      const res = await API.get(`/api/v1/market/search?q=${encodeURIComponent(q)}&limit=10`);
      const results = (res && res.ok && Array.isArray(res.data)) ? res.data : [];
      if (results && results.length > 0) {
        state.searchSelectedIdx = -1;
        dropdown.innerHTML = results.map((item, idx) => `
          <div class="search-item" data-sym="${sanitize(item.symbol)}" data-idx="${idx}">
            <div class="search-item-left">
              <span class="search-item-sym">${sanitize(item.symbol)}</span>
              <span class="search-item-name">${sanitize(item.name)}</span>
            </div>
            <span class="search-item-type">${sanitize(item.type || item.exchange || 'ASSET')}</span>
          </div>
        `).join('');
        dropdown.style.display = 'block';

        dropdown.querySelectorAll('.search-item').forEach(el => {
          el.addEventListener('click', () => {
            const sym = el.getAttribute('data-sym');
            setActiveSymbol(sym, true);
            dropdown.style.display = 'none';
            input.value = '';
            clearBtn.style.display = 'none';
          });
        });
      } else {
        dropdown.innerHTML = '<div style="padding: 12px; color: var(--text-dim); text-align: center;">No instruments found</div>';
        dropdown.style.display = 'block';
      }
    }

    function highlightSearchItem(items) {
      items.forEach((item, idx) => {
        if (idx === state.searchSelectedIdx) {
          item.classList.add('selected');
          item.scrollIntoView({ block: 'nearest' });
        } else {
          item.classList.remove('selected');
        }
      });
    }
  }

  // ========================================================================
  // PAGE 1: MARKET OVERVIEW
  // ========================================================================
  async function loadOverviewData() {
    // 1. Fear & Greed sentiment
    API.get('/api/v1/sentiment').then(res => {
      const s = (res && res.ok) ? res.data : null;
      const valEl = document.getElementById('overviewFgVal');
      const lblEl = document.getElementById('overviewFgLabel');
      const barEl = document.getElementById('overviewFgBar');
      const ctxSentiment = document.getElementById('ctxSentiment');
      if (s) {
        const score = s.fear_and_greed_value ?? s.fear_and_greed_score ?? 50;
        const label = s.classification ?? s.sentiment_label ?? 'Neutral';
        if (valEl) valEl.textContent = score;
        if (lblEl) {
          lblEl.textContent = label;
          lblEl.className = 'fg-label-tag ' + (score >= 60 ? 'bullish' : score <= 40 ? 'bearish' : 'neutral');
        }
        if (barEl) barEl.style.width = `${Math.min(100, Math.max(0, score))}%`;
        if (ctxSentiment) ctxSentiment.textContent = `${score} (${label})`;
      } else {
        if (valEl) valEl.textContent = '--';
        if (lblEl) lblEl.textContent = 'Unavailable';
        if (barEl) barEl.style.width = '50%';
        if (ctxSentiment) ctxSentiment.textContent = 'Unavailable';
      }
    });

    // 2. Market Overview Cards & Market Hours
    const overviewRes = await API.get('/api/v1/market/overview');
    if (overviewRes && overviewRes.ok && overviewRes.data) {
      const overview = overviewRes.data;
      state.marketOverview = overview;
      if (overview.market_hours) {
        const us = document.getElementById('valHoursUS');
        const nse = document.getElementById('valHoursNSE');
        const fx = document.getElementById('valHoursFX');
        if (us) us.textContent = overview.market_hours.us_equities || 'OPEN';
        if (nse) nse.textContent = overview.market_hours.indian_nse || 'OPEN';
        if (fx) fx.textContent = overview.market_hours.forex || 'OPEN';
      }
      renderOverviewCards(state.watchlistCategory);
      renderHeatmap(overview);
    } else {
      const grid = document.getElementById('overviewCardsGrid');
      if (grid) {
        grid.innerHTML = `<div class="error-state" style="padding: 24px; text-align: center; color: var(--text-muted);">
          <span>Failed to load market overview (${sanitize(overviewRes?.error || 'timeout')}).</span>
          <button class="btn btn-xs btn-outline" onclick="window.AlphaVol.retryOverview()" style="margin-left: 10px; cursor: pointer;">Retry</button>
        </div>`;
      }
      const heatmap = document.getElementById('overviewHeatmap');
      if (heatmap) {
        heatmap.innerHTML = `<div class="empty-state" style="padding: 16px; text-align: center; color: var(--text-muted); font-size: 12px;">Heatmap snapshot unavailable</div>`;
      }
    }

    // 3. Market Movers
    const moversRes = await API.get('/api/v1/market/movers');
    if (moversRes && moversRes.ok && moversRes.data) {
      state.marketMovers = moversRes.data;
      renderMovers(moversRes.data);
    } else {
      ['moversGainersList', 'moversLosersList', 'moversVolumeList'].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
          el.innerHTML = `<div class="error-state" style="padding: 14px; text-align: center; color: var(--text-muted); font-size: 11px;">
            Failed to load movers. <button class="btn btn-xs btn-link" onclick="window.AlphaVol.retryOverview()" style="cursor:pointer; color: var(--accent-cyan);">Retry</button>
          </div>`;
        }
      });
    }

    // 4. Market Headlines
    const newsRes = await API.get('/api/v1/news?limit=4');
    if (newsRes && newsRes.ok && newsRes.data) {
      renderOverviewNews(newsRes.data);
    } else {
      const newsList = document.getElementById('overviewNewsList');
      if (newsList) {
        newsList.innerHTML = `<div class="empty-state" style="padding: 16px; color: var(--text-muted); font-size: 12px;">Headlines temporarily unavailable</div>`;
      }
    }
  }

  function initOverviewCategoryFilters() {
    const filters = document.getElementById('overviewCategoryFilters');
    if (!filters) return;
    filters.querySelectorAll('.cat-pill').forEach(btn => {
      btn.addEventListener('click', () => {
        filters.querySelectorAll('.cat-pill').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const cat = btn.getAttribute('data-cat');
        renderOverviewCards(cat);
      });
    });
  }

  function renderOverviewCards(category = 'ALL') {
    const grid = document.getElementById('overviewCardsGrid');
    if (!grid || !state.marketOverview) return;

    let items = [];
    const ov = state.marketOverview;
    if (category === 'ALL') {
      items = [...(ov.indices || []), ...(ov.crypto || []), ...(ov.indian_stocks || []), ...(ov.us_stocks || []), ...(ov.forex || [])];
    } else if (category === 'INDEX') {
      items = ov.indices || [];
    } else if (category === 'CRYPTO') {
      items = ov.crypto || [];
    } else if (category === 'INDIA') {
      items = ov.indian_stocks || [];
    } else if (category === 'US') {
      items = ov.us_stocks || [];
    } else if (category === 'FOREX') {
      items = ov.forex || [];
    }

    // Store in global ticker cache
    items.forEach(t => {
      state.tickers[t.symbol] = t;
    });

    if (items.length === 0) {
      grid.innerHTML = '<div class="empty-state">No instruments in this category</div>';
      return;
    }

    grid.innerHTML = items.map(t => {
      const isPos = (t.change_24h_pct || 0) >= 0;
      const chgClass = isPos ? 'text-success' : 'text-danger';
      const statusClass = t.market_status === 'CLOSED' ? 'status-closed' : 'status-open';

      return `
        <div class="overview-card" data-sym="${sanitize(t.symbol)}">
          <div class="ov-card-top">
            <span class="ov-sym">${sanitize(t.symbol)}</span>
            <span class="market-status-pill ${statusClass}">${sanitize(t.market_status || 'OPEN')}</span>
          </div>
          <div class="ov-name">${sanitize(t.name || t.symbol)}</div>
          <div class="ov-card-price-row">
            <span class="ov-price">${formatCurrency(t.price, t.currency)}</span>
            <span class="ov-change ${chgClass}">${formatChange(t.change_24h_pct)}</span>
          </div>
          <div class="ov-meta-row">
            <span>24h H: ${formatCurrency(t.high_24h, t.currency)}</span>
            <span>24h L: ${formatCurrency(t.low_24h, t.currency)}</span>
          </div>
        </div>
      `;
    }).join('');

    grid.querySelectorAll('.overview-card').forEach(card => {
      card.addEventListener('click', () => {
        const sym = card.getAttribute('data-sym');
        setActiveSymbol(sym, true);
      });
    });
  }

  function renderHeatmap(overview) {
    const container = document.getElementById('overviewHeatmap');
    if (!container) return;

    // Pick top benchmarks & instruments
    const targets = ['^NSEI', '^BSESN', '^GSPC', '^IXIC', 'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'RELIANCE.NS', 'TCS.NS', 'AAPL', 'NVDA', 'TSLA'];
    const all = [...(overview.indices || []), ...(overview.crypto || []), ...(overview.indian_stocks || []), ...(overview.us_stocks || [])];
    const mapItems = all.filter(t => targets.includes(t.symbol));

    container.innerHTML = mapItems.map(t => {
      const chg = t.change_24h_pct || 0.0;
      let colorClass = 'tile-neutral';
      if (chg >= 2.0) colorClass = 'tile-deep-green';
      else if (chg > 0.3) colorClass = 'tile-green';
      else if (chg <= -2.0) colorClass = 'tile-deep-red';
      else if (chg < -0.3) colorClass = 'tile-red';

      return `
        <div class="heatmap-tile ${colorClass}" data-sym="${sanitize(t.symbol)}" title="${sanitize(t.name)}: ${formatChange(chg)}">
          <span class="tile-sym">${sanitize(t.symbol)}</span>
          <span class="tile-chg">${formatChange(chg)}</span>
        </div>
      `;
    }).join('');

    container.querySelectorAll('.heatmap-tile').forEach(tile => {
      tile.addEventListener('click', () => {
        setActiveSymbol(tile.getAttribute('data-sym'), true);
      });
    });
  }

  function renderMovers(movers) {
    renderMoverList('moversGainersList', movers.top_gainers || [], true);
    renderMoverList('moversLosersList', movers.top_losers || [], false);
    renderMoverList('moversVolumeList', movers.highest_volume || [], null);
  }

  function renderMoverList(containerId, items, isGainers) {
    const container = document.getElementById(containerId);
    if (!container) return;
    if (items.length === 0) {
      container.innerHTML = '<div class="empty-state">No data available</div>';
      return;
    }

    container.innerHTML = items.map(t => {
      const chgClass = (t.change_24h_pct || 0) >= 0 ? 'text-success' : 'text-danger';
      const rightContent = isGainers === null
        ? `<span class="mover-price">${formatCurrency(t.price, t.currency)}</span><span class="mover-chg text-cyan">Vol $${formatVolume((t.volume_24h || 0) * t.price)}</span>`
        : `<span class="mover-price">${formatCurrency(t.price, t.currency)}</span><span class="mover-chg ${chgClass}">${formatChange(t.change_24h_pct)}</span>`;

      return `
        <div class="mover-row" data-sym="${sanitize(t.symbol)}">
          <div class="mover-left">
            <span class="mover-sym">${sanitize(t.symbol)}</span>
            <span class="mover-name">${sanitize(t.name || t.symbol)}</span>
          </div>
          <div class="mover-right">
            ${rightContent}
          </div>
        </div>
      `;
    }).join('');

    container.querySelectorAll('.mover-row').forEach(row => {
      row.addEventListener('click', () => {
        setActiveSymbol(row.getAttribute('data-sym'), true);
      });
    });
  }

  function renderOverviewNews(newsList) {
    const grid = document.getElementById('overviewNewsGrid');
    if (!grid) return;
    if (newsList.length === 0) {
      grid.innerHTML = '<div class="empty-state">No news headlines available</div>';
      return;
    }

    grid.innerHTML = newsList.map(n => {
      const sentClass = (n.sentiment || 'neutral').toLowerCase();
      return `
        <a href="${sanitize(n.url || '#')}" target="_blank" rel="noopener noreferrer" class="overview-news-item">
          <div class="ov-news-top">
            <span>${sanitize(n.source || 'Market Feed')}</span>
            <span class="sentiment-badge ${sentClass}">${sanitize(n.sentiment || 'NEUTRAL')}</span>
          </div>
          <h4 class="ov-news-headline">${sanitize(n.headline || n.title)}</h4>
        </a>
      `;
    }).join('');
  }

  // ========================================================================
  // PAGE 2: TRADING TERMINAL
  // ========================================================================
  function initTerminal() {
    initTerminalWatchlist();
    initTerminalControls();
    initTerminalSubChartTabs();
    initTerminalAiDrawer();
  }

  function initTerminalWatchlist() {
    const filterBar = document.getElementById('terminalWlFilters');
    if (filterBar) {
      filterBar.querySelectorAll('.wl-chip').forEach(btn => {
        btn.addEventListener('click', () => {
          filterBar.querySelectorAll('.wl-chip').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          state.watchlistCategory = btn.getAttribute('data-cat');
          renderTerminalWatchlist();
        });
      });
    }

    const searchInput = document.getElementById('wlSearchInput');
    if (searchInput) {
      searchInput.addEventListener('input', () => {
        renderTerminalWatchlist(searchInput.value.trim().toLowerCase());
      });
    }
  }

  function renderTerminalWatchlist(query = '') {
    const container = document.getElementById('terminalWatchlist');
    if (!container) return;

    let list = state.instruments;
    if (state.watchlistCategory === 'CRYPTO') list = list.filter(i => i.type === 'crypto');
    else if (state.watchlistCategory === 'INDIA') list = list.filter(i => i.type === 'indian_stock');
    else if (state.watchlistCategory === 'US') list = list.filter(i => i.type === 'global_stock');
    else if (state.watchlistCategory === 'INDEX') list = list.filter(i => i.type === 'index');
    else if (state.watchlistCategory === 'FOREX') list = list.filter(i => i.type === 'forex');
    else if (state.watchlistCategory === 'FAV') list = list.filter(i => state.favorites.has(i.symbol));

    if (query) {
      list = list.filter(i => i.symbol.toLowerCase().includes(query) || i.name.toLowerCase().includes(query));
    }

    container.innerHTML = list.map(item => {
      const tick = state.tickers[item.symbol];
      const isFav = state.favorites.has(item.symbol);
      const isActive = item.symbol === state.activeSymbol;
      const priceStr = tick ? formatCurrency(tick.price, item.currency) : '--';
      const chgStr = tick ? formatChange(tick.change_24h_pct) : '--%';
      const chgClass = tick && (tick.change_24h_pct || 0) >= 0 ? 'text-success' : 'text-danger';

      return `
        <div class="wl-row ${isActive ? 'active' : ''}" data-sym="${sanitize(item.symbol)}">
          <div class="wl-left">
            <span class="wl-star ${isFav ? 'starred' : ''}" data-fav-sym="${sanitize(item.symbol)}">★</span>
            <div class="wl-sym-box">
              <span class="wl-sym">${sanitize(item.symbol)}</span>
              <span class="wl-name">${sanitize(item.name)}</span>
            </div>
          </div>
          <div class="wl-right">
            <span class="wl-price">${priceStr}</span>
            <span class="wl-chg ${chgClass}">${chgStr}</span>
          </div>
        </div>
      `;
    }).join('');

    container.querySelectorAll('.wl-row').forEach(row => {
      row.addEventListener('click', (e) => {
        if (e.target.classList.contains('wl-star')) return;
        const sym = row.getAttribute('data-sym');
        setActiveSymbol(sym, false);
        container.querySelectorAll('.wl-row').forEach(r => r.classList.remove('active'));
        row.classList.add('active');
      });
    });

    container.querySelectorAll('.wl-star').forEach(star => {
      star.addEventListener('click', (e) => {
        e.stopPropagation();
        const sym = star.getAttribute('data-fav-sym');
        if (state.favorites.has(sym)) {
          state.favorites.delete(sym);
          star.classList.remove('starred');
        } else {
          state.favorites.add(sym);
          star.classList.add('starred');
        }
      });
    });
  }

  function initTerminalControls() {
    // Timeframe selector
    const tfSelector = document.getElementById('terminalTfSelector');
    if (tfSelector) {
      tfSelector.querySelectorAll('.tf-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          tfSelector.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          state.activeTimeframe = btn.getAttribute('data-tf');
          loadTerminalCandles();
          loadTerminalSignal();
          ensureWebSockets();
        });
      });
    }

    // Chart Type (Candles / Line)
    const btnCandles = document.getElementById('terminalBtnCandles');
    const btnLine = document.getElementById('terminalBtnLine');
    if (btnCandles && btnLine) {
      btnCandles.addEventListener('click', () => {
        btnCandles.classList.add('active');
        btnLine.classList.remove('active');
        state.chartType = 'candles';
        toggleChartSeriesType();
      });
      btnLine.addEventListener('click', () => {
        btnLine.classList.add('active');
        btnCandles.classList.remove('active');
        state.chartType = 'line';
        toggleChartSeriesType();
      });
    }

    // Overlay Indicators
    ['EMA20', 'EMA50', 'EMA200', 'SMA50', 'BB', 'VWAP'].forEach(ind => {
      const chk = document.getElementById(`terminalToggle${ind}`);
      if (chk) {
        chk.addEventListener('change', () => {
          state.indicatorToggles[ind.toLowerCase()] = chk.checked;
          renderTerminalOverlayIndicators();
        });
      }
    });

    // Fit View Button
    const fitBtn = document.getElementById('terminalFitBtn');
    if (fitBtn) {
      fitBtn.addEventListener('click', () => {
        if (mainChart) mainChart.timeScale().fitContent();
        if (subChart) subChart.timeScale().fitContent();
      });
    }
  }

  function initTerminalSubChartTabs() {
    const tabs = document.getElementById('terminalSubChartTabs');
    const wrapper = document.getElementById('terminalSubChartWrapper');
    if (!tabs || !wrapper) return;

    tabs.querySelectorAll('.sub-tab').forEach(btn => {
      btn.addEventListener('click', () => {
        tabs.querySelectorAll('.sub-tab').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const sub = btn.getAttribute('data-sub');
        state.subChartType = sub;

        if (sub === 'none') {
          wrapper.style.display = 'none';
        } else {
          wrapper.style.display = 'flex';
          initOrResizeTerminalChart();
          renderTerminalSubChart();
        }
      });
    });
  }

  function initTerminalAiDrawer() {
    const drawer = document.getElementById('terminalAiDrawer');
    const toggleBtn = document.getElementById('terminalToggleAiDrawer');
    const closeBtn = document.getElementById('terminalCloseAiDrawer');
    const form = document.getElementById('terminalDrawerForm');
    const input = document.getElementById('terminalDrawerInput');
    const symSpan = document.getElementById('terminalDrawerSym');

    if (toggleBtn && drawer) {
      toggleBtn.addEventListener('click', () => {
        drawer.classList.add('open');
        if (symSpan) symSpan.textContent = state.activeSymbol;
        if (input) input.focus();
      });
    }
    if (closeBtn && drawer) {
      closeBtn.addEventListener('click', () => {
        drawer.classList.remove('open');
      });
    }

    if (form) {
      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (isAiExecuting) return;
        const q = input.value.trim();
        if (!q) return;
        appendDrawerMessage('user', q);
        input.value = '';
        await executeAiQuery(q, appendDrawerMessage, 'terminalDrawerMessages');
      });
    }

    // Quick chips inside drawer
    drawer.querySelectorAll('.drawer-chip').forEach(btn => {
      btn.addEventListener('click', async () => {
        if (isAiExecuting) return;
        const q = btn.getAttribute('data-q');
        if (!q) return;
        appendDrawerMessage('user', q);
        await executeAiQuery(q, appendDrawerMessage, 'terminalDrawerMessages');
      });
    });
  }

  function appendDrawerMessage(role, text, meta = null, isLoading = false) {
    appendAiMessageToContainer('terminalDrawerMessages', role, text, meta, isLoading, appendDrawerMessage);
  }

  // --- Lightweight Charts Rendering Engine ---
  function initOrResizeTerminalChart() {
    const container = document.getElementById('terminalChartContainer');
    const subContainer = document.getElementById('terminalSubChartCanvas');
    if (!container || typeof LightweightCharts === 'undefined') return;

    if (!mainChart) {
      mainChart = LightweightCharts.createChart(container, {
        layout: {
          background: { color: '#0a0d14' },
          textColor: '#94a3b8',
          fontSize: 11,
          fontFamily: "'JetBrains Mono', monospace",
        },
        grid: {
          vertLines: { color: 'rgba(255, 255, 255, 0.04)' },
          horzLines: { color: 'rgba(255, 255, 255, 0.04)' },
        },
        crosshair: {
          mode: LightweightCharts.CrosshairMode.Normal,
        },
        timeScale: {
          borderColor: 'rgba(255, 255, 255, 0.08)',
          timeVisible: true,
          secondsVisible: false,
          tickMarkFormatter: (time, tickMarkType, locale) => formatChartTickMark(time, tickMarkType, locale),
        },
        localization: {
          locale: 'en-IN',
          dateFormat: "dd MMM 'yy",
          timeFormatter: (time) => formatChartCrosshairTime(time),
        },
        rightPriceScale: {
          borderColor: 'rgba(255, 255, 255, 0.08)',
        },
      });

      // Volume Series
      volumeSeries = mainChart.addHistogramSeries({
        color: '#26a69a',
        priceFormat: { type: 'volume' },
        priceScaleId: '',
        scaleMargins: { top: 0.85, bottom: 0 },
      });

      // Candlestick Series
      candleSeries = mainChart.addCandlestickSeries({
        upColor: '#0ecb81',
        downColor: '#f6465d',
        borderUpColor: '#0ecb81',
        borderDownColor: '#f6465d',
        wickUpColor: '#0ecb81',
        wickDownColor: '#f6465d',
      });

      // Line Series
      lineSeries = mainChart.addLineSeries({
        color: '#00e5ff',
        lineWidth: 2,
        visible: false,
      });

      // Overlay series
      overlaySeries.ema20 = mainChart.addLineSeries({ color: '#f59e0b', lineWidth: 1.5, title: 'EMA 20' });
      overlaySeries.ema50 = mainChart.addLineSeries({ color: '#00e5ff', lineWidth: 1.5, title: 'EMA 50' });
      overlaySeries.ema200 = mainChart.addLineSeries({ color: '#a855f7', lineWidth: 2, title: 'EMA 200' });
      overlaySeries.sma50 = mainChart.addLineSeries({ color: '#38bdf8', lineWidth: 1.5, title: 'SMA 50' });
      overlaySeries.bbUpper = mainChart.addLineSeries({ color: 'rgba(255, 255, 255, 0.3)', lineWidth: 1, lineStyle: 2, title: 'BB Upper' });
      overlaySeries.bbLower = mainChart.addLineSeries({ color: 'rgba(255, 255, 255, 0.3)', lineWidth: 1, lineStyle: 2, title: 'BB Lower' });
      overlaySeries.vwap = mainChart.addLineSeries({ color: '#ec4899', lineWidth: 1.5, title: 'VWAP' });

      // Resize observer
      const resizeObserver = new ResizeObserver(() => {
        if (mainChart) mainChart.resize(container.clientWidth, container.clientHeight);
        if (subChart && subContainer) subChart.resize(subContainer.clientWidth, subContainer.clientHeight);
      });
      resizeObserver.observe(container);
    }

    // Sub-Chart Initialization
    if (subContainer && !subChart && state.subChartType !== 'none') {
      subChart = LightweightCharts.createChart(subContainer, {
        layout: {
          background: { color: '#0a0d14' },
          textColor: '#94a3b8',
          fontSize: 10,
          fontFamily: "'JetBrains Mono', monospace",
        },
        grid: {
          vertLines: { color: 'rgba(255, 255, 255, 0.04)' },
          horzLines: { color: 'rgba(255, 255, 255, 0.04)' },
        },
        timeScale: {
          borderColor: 'rgba(255, 255, 255, 0.08)',
          timeVisible: true,
          secondsVisible: false,
          tickMarkFormatter: (time, tickMarkType, locale) => formatChartTickMark(time, tickMarkType, locale),
        },
        localization: {
          locale: 'en-IN',
          dateFormat: "dd MMM 'yy",
          timeFormatter: (time) => formatChartCrosshairTime(time),
        },
        rightPriceScale: {
          borderColor: 'rgba(255, 255, 255, 0.08)',
        },
      });

      // Synchronize range between main chart and sub-chart
      mainChart.timeScale().subscribeVisibleLogicalRangeChange(range => {
        if (isSyncingRange || !subChart || !range) return;
        isSyncingRange = true;
        subChart.timeScale().setVisibleLogicalRange(range);
        isSyncingRange = false;
      });

      subChart.timeScale().subscribeVisibleLogicalRangeChange(range => {
        if (isSyncingRange || !mainChart || !range) return;
        isSyncingRange = true;
        mainChart.timeScale().setVisibleLogicalRange(range);
        isSyncingRange = false;
      });
    }

    if (mainChart) mainChart.resize(container.clientWidth, container.clientHeight);
    if (subChart && subContainer) subChart.resize(subContainer.clientWidth, subContainer.clientHeight);
  }

  function toggleChartSeriesType() {
    if (!candleSeries || !lineSeries) return;
    if (state.chartType === 'line') {
      candleSeries.applyOptions({ visible: false });
      lineSeries.applyOptions({ visible: true });
    } else {
      candleSeries.applyOptions({ visible: true });
      lineSeries.applyOptions({ visible: false });
    }
  }

  async function loadTerminalCandles(showLoader = true) {
    const loader = document.getElementById('terminalChartLoader');
    if (loader && showLoader && (!state.candles || state.candles.length === 0)) loader.style.display = 'flex';

    const url = `/api/v1/market/${encodeURIComponent(state.activeSymbol)}/candles?timeframe=${state.activeTimeframe}&limit=160`;
    const res = await API.get(url);

    if (loader) loader.style.display = 'none';

    const candles = (res && res.ok && Array.isArray(res.data)) ? res.data : null;

    if (candles && candles.length > 0) {
      // Ensure each candle in state.candles has a canonical integer seconds .time and display-projected .chartTime
      candles.forEach(c => {
        if (c.time === undefined || c.time === null) {
          c.time = c.timestamp_ms ? Math.floor(c.timestamp_ms / 1000) : Math.floor(new Date(c.timestamp).getTime() / 1000);
        }
        c.chartTime = toChartTime(c.time);
      });
      state.candles = candles;

      // Transform for TV Lightweight Charts (projected into intended display timezone)
      const tvData = candles.map(c => ({
        time: c.chartTime,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      }));

      const lineData = candles.map(c => ({
        time: c.chartTime,
        value: c.close,
      }));

      const volData = candles.map(c => ({
        time: c.chartTime,
        value: c.volume,
        color: c.close >= c.open ? 'rgba(14, 203, 129, 0.4)' : 'rgba(246, 70, 93, 0.4)',
      }));

      if (candleSeries) candleSeries.setData(tvData);
      if (lineSeries) lineSeries.setData(lineData);
      if (volumeSeries) volumeSeries.setData(volData);

      // Adjust timeScale visibility based on timeframe (intraday shows HH:MM; 1d/1w shows date)
      const isIntraday = !['1d', '1w'].includes(state.activeTimeframe);
      if (mainChart) {
        mainChart.applyOptions({
          timeScale: {
            timeVisible: isIntraday,
            secondsVisible: false,
          },
        });
      }
      if (subChart) {
        subChart.applyOptions({
          timeScale: {
            timeVisible: isIntraday,
            secondsVisible: false,
          },
        });
      }

      // Update Header with authoritative spot ticker price if available, else latest bar price
      const tick = state.tickers[state.activeSymbol];
      const latest = candles[candles.length - 1];
      const prev = candles.length > 1 ? candles[candles.length - 2] : latest;
      const chgPct = prev.close > 0 ? ((latest.close - prev.close) / prev.close) * 100 : 0;
      
      const displayPrice = tick ? tick.price : latest.close;
      const displayChg = tick ? tick.change_24h_pct : chgPct;
      const displayHigh = tick ? (tick.high_24h || latest.high) : latest.high;
      const displayLow = tick ? (tick.low_24h || latest.low) : latest.low;
      const displayVol = tick ? (tick.volume_24h || latest.volume) : latest.volume;

      updateTerminalHeader(
        displayPrice,
        displayChg,
        displayHigh,
        displayLow,
        displayVol,
        latest.freshness_text || (tick && tick.freshness_text),
        latest.data_status || (tick && tick.data_status)
      );

      // Compute and render indicators
      loadTerminalIndicators();
    } else {
      console.warn(`[Candles Warning] Failed to load candles for ${state.activeSymbol}: ${res?.error || 'Empty'}`);
      const tick = state.tickers[state.activeSymbol];
      if (tick) {
        updateTerminalHeader(tick.price, tick.change_24h_pct, tick.high_24h || tick.price, tick.low_24h || tick.price, tick.volume_24h || 0, tick.freshness_text, tick.data_status);
      }
    }
  }

  function updateTerminalHeader(price, changePct, high, low, volume, freshnessText = null, dataStatus = null) {
    const symEl = document.getElementById('terminalSymbol');
    const nameEl = document.getElementById('terminalName');
    const exchEl = document.getElementById('terminalExchange');
    const priceEl = document.getElementById('terminalPrice');
    const chgEl = document.getElementById('terminalChange');
    const volEl = document.getElementById('terminalVolume');
    const highEl = document.getElementById('terminalHigh24');
    const lowEl = document.getElementById('terminalLow24');
    const rangeEl = document.getElementById('terminal52wRange');
    const statusEl = document.getElementById('terminalStatusBadge');

    const item = state.instruments.find(i => i.symbol === state.activeSymbol) || { name: state.activeSymbol, exchange: 'MARKET', currency: 'USD' };

    if (symEl) symEl.textContent = state.activeSymbol;
    if (nameEl) nameEl.textContent = item.name;
    if (exchEl) exchEl.textContent = item.exchange.toUpperCase();

    if (priceEl) priceEl.textContent = formatCurrency(price, item.currency);
    if (chgEl) {
      chgEl.textContent = formatChange(changePct);
      chgEl.className = 'change-display ' + (changePct >= 0 ? 'text-success' : 'text-danger');
    }
    if (volEl) volEl.textContent = formatVolume(volume);
    if (highEl) highEl.textContent = formatCurrency(high, item.currency);
    if (lowEl) lowEl.textContent = formatCurrency(low, item.currency);

    const tick = state.tickers[state.activeSymbol];
    if (rangeEl && tick && tick.low_52w && tick.high_52w) {
      rangeEl.textContent = `${formatCurrency(tick.low_52w, item.currency)} – ${formatCurrency(tick.high_52w, item.currency)}`;
    }
    if (statusEl) {
      const text = freshnessText || (tick && tick.freshness_text) || (tick && tick.market_status) || 'LIVE';
      statusEl.textContent = text;
      const dStat = dataStatus || (tick && tick.data_status) || (tick && tick.market_status === 'CLOSED' ? 'CLOSED' : 'LIVE');
      statusEl.className = 'market-status-pill ' + (dStat === 'LIVE' ? 'status-open' : (dStat === 'DELAYED' ? 'status-warning' : (dStat === 'STALE' ? 'status-warning' : 'status-closed')));
    }
  }

  async function loadTerminalIndicators() {
    const url = `/api/v1/market/${encodeURIComponent(state.activeSymbol)}/indicators?timeframe=${state.activeTimeframe}&limit=160`;
    const res = await API.get(url);
    if (res && res.ok && res.data) {
      state.indicators = res.data;
      renderTerminalOverlayIndicators();
      renderTerminalSubChart();
    } else {
      console.warn(`[Indicators Warning] Failed to load indicators for ${state.activeSymbol}:`, res?.error);
    }
  }

  function renderTerminalOverlayIndicators() {
    if (!state.indicators || !state.indicators.time_series) return;
    const ts = state.indicators.time_series;
    const timestamps = ts.timestamps || [];
    const n = timestamps.length;

    function buildSeriesData(arr) {
      if (!arr || arr.length === 0) return [];
      const out = [];
      for (let i = 0; i < n; i++) {
        if (arr[i] !== null && arr[i] !== undefined && !isNaN(arr[i])) {
          out.push({
            time: toChartTime(Math.floor(new Date(timestamps[i]).getTime() / 1000)),
            value: arr[i],
          });
        }
      }
      return out;
    }

    if (overlaySeries.ema20) {
      overlaySeries.ema20.applyOptions({ visible: state.indicatorToggles.ema20 });
      if (state.indicatorToggles.ema20) overlaySeries.ema20.setData(buildSeriesData(ts.ema20));
    }
    if (overlaySeries.ema50) {
      overlaySeries.ema50.applyOptions({ visible: state.indicatorToggles.ema50 });
      if (state.indicatorToggles.ema50) overlaySeries.ema50.setData(buildSeriesData(ts.ema50));
    }
    if (overlaySeries.ema200) {
      overlaySeries.ema200.applyOptions({ visible: state.indicatorToggles.ema200 });
      if (state.indicatorToggles.ema200) overlaySeries.ema200.setData(buildSeriesData(ts.ema200));
    }
    if (overlaySeries.sma50) {
      overlaySeries.sma50.applyOptions({ visible: state.indicatorToggles.sma50 });
      if (state.indicatorToggles.sma50) overlaySeries.sma50.setData(buildSeriesData(ts.sma50));
    }
    if (overlaySeries.bbUpper && overlaySeries.bbLower) {
      overlaySeries.bbUpper.applyOptions({ visible: state.indicatorToggles.bb });
      overlaySeries.bbLower.applyOptions({ visible: state.indicatorToggles.bb });
      if (state.indicatorToggles.bb) {
        overlaySeries.bbUpper.setData(buildSeriesData(ts.bb_upper));
        overlaySeries.bbLower.setData(buildSeriesData(ts.bb_lower));
      }
    }
    if (overlaySeries.vwap) {
      overlaySeries.vwap.applyOptions({ visible: state.indicatorToggles.vwap });
      if (state.indicatorToggles.vwap) overlaySeries.vwap.setData(buildSeriesData(ts.vwap));
    }
  }

  function renderTerminalSubChart() {
    if (!subChart || !state.indicators || !state.indicators.time_series) return;
    const ts = state.indicators.time_series;
    const timestamps = ts.timestamps || [];
    const n = timestamps.length;

    // Clear previous subSeries
    Object.values(subSeries).forEach(s => {
      try { subChart.removeSeries(s); } catch (e) {}
    });
    subSeries = {};

    const valEl = document.getElementById('terminalSubChartVal');

    if (state.subChartType === 'rsi') {
      const rsiLine = subChart.addLineSeries({ color: '#a855f7', lineWidth: 2, title: 'RSI (14)' });
      const data = [];
      for (let i = 0; i < n; i++) {
        if (ts.rsi[i] !== null && !isNaN(ts.rsi[i])) {
          data.push({ time: toChartTime(Math.floor(new Date(timestamps[i]).getTime() / 1000)), value: ts.rsi[i] });
        }
      }
      rsiLine.setData(data);
      subSeries.rsi = rsiLine;

      const curr = state.indicators.current?.rsi;
      if (valEl && curr !== undefined) {
        valEl.textContent = `RSI: ${curr.toFixed(1)} • ${curr > 70 ? 'Overbought' : (curr < 30 ? 'Oversold' : 'Neutral')}`;
      }
    } else if (state.subChartType === 'macd') {
      const macdLine = subChart.addLineSeries({ color: '#00e5ff', lineWidth: 1.5, title: 'MACD' });
      const signalLine = subChart.addLineSeries({ color: '#f59e0b', lineWidth: 1.5, title: 'Signal' });
      const hist = subChart.addHistogramSeries({ color: '#0ecb81', title: 'Hist' });

      const mData = [], sData = [], hData = [];
      for (let i = 0; i < n; i++) {
        const t = toChartTime(Math.floor(new Date(timestamps[i]).getTime() / 1000));
        if (ts.macd_line[i] !== null) mData.push({ time: t, value: ts.macd_line[i] });
        if (ts.macd_signal[i] !== null) sData.push({ time: t, value: ts.macd_signal[i] });
        if (ts.macd_hist[i] !== null) {
          hData.push({
            time: t,
            value: ts.macd_hist[i],
            color: ts.macd_hist[i] >= 0 ? 'rgba(14, 203, 129, 0.6)' : 'rgba(246, 70, 93, 0.6)',
          });
        }
      }
      macdLine.setData(mData);
      signalLine.setData(sData);
      hist.setData(hData);
      subSeries.macd = macdLine;
      subSeries.signal = signalLine;
      subSeries.hist = hist;

      const currH = state.indicators.current?.macd_hist;
      if (valEl && currH !== undefined) {
        valEl.textContent = `MACD Hist: ${currH.toFixed(2)}`;
      }
    } else if (state.subChartType === 'stoch') {
      const kLine = subChart.addLineSeries({ color: '#38bdf8', lineWidth: 1.5, title: '%K' });
      const dLine = subChart.addLineSeries({ color: '#ec4899', lineWidth: 1.5, title: '%D' });

      const kData = [], dData = [];
      for (let i = 0; i < n; i++) {
        const t = toChartTime(Math.floor(new Date(timestamps[i]).getTime() / 1000));
        if (ts.stoch_k && ts.stoch_k[i] !== null) kData.push({ time: t, value: ts.stoch_k[i] });
        if (ts.stoch_d && ts.stoch_d[i] !== null) dData.push({ time: t, value: ts.stoch_d[i] });
      }
      kLine.setData(kData);
      dLine.setData(dData);
      subSeries.k = kLine;
      subSeries.d = dLine;

      const currK = state.indicators.current?.stoch_k;
      if (valEl && currK !== undefined) {
        valEl.textContent = `Stoch %K: ${currK.toFixed(1)}`;
      }
    }
  }

  async function loadTerminalSignal() {
    const url = `/api/v1/signal/${encodeURIComponent(state.activeSymbol)}?timeframe=${state.activeTimeframe}`;
    const res = await API.get(url);
    if (res && res.ok && res.data) {
      state.signal = res.data;
      renderTerminalSignal(res.data);
    } else {
      console.warn(`[Signal Warning] Failed to load ML signal for ${state.activeSymbol}:`, res?.error);
      const badgeEl = document.getElementById('terminalSignalBadge');
      if (badgeEl) badgeEl.textContent = 'SIGNAL UNAVAILABLE';
      const reasonsList = document.getElementById('terminalReasonsList');
      if (reasonsList) {
        reasonsList.innerHTML = `<div style="color:var(--text-muted); padding:10px; font-size:12px;">
          Signal analysis unavailable (${sanitize(res?.error || 'error')}). <a href="javascript:void(0)" onclick="window.AlphaVol.retrySignal()" style="color:var(--accent-cyan); text-decoration:underline; cursor:pointer;">Retry</a>
        </div>`;
      }
    }
  }

  function renderTerminalSignal(sig) {
    const actionEl = document.getElementById('terminalSigAction');
    const confEl = document.getElementById('terminalSigConf');
    const setupEl = document.getElementById('terminalSigSetup');
    const volEl = document.getElementById('terminalSigVol');
    const regimeEl = document.getElementById('terminalSigRegime');
    const agreeEl = document.getElementById('terminalSigAgreement');
    const riskEl = document.getElementById('terminalSigRisk');
    const badgeEl = document.getElementById('terminalSignalBadge');

    const action = sig.signal || sig.action || 'HOLD';
    if (actionEl) {
      actionEl.textContent = action.replace('_', ' ');
      actionEl.className = 'signal-action-tag ' + `badge-${action.toLowerCase().replace(/[^a-z0-9]/g, '-')}`;
    }

    // Separate probabilities vs calibrated model confidence
    const confPct = Math.round((sig.confidence || 0.5) * 100);
    const buyProb = Math.round((sig.buy_probability !== undefined ? sig.buy_probability : 0.33) * 100);
    const sellProb = Math.round((sig.sell_probability !== undefined ? sig.sell_probability : 0.33) * 100);
    const holdProb = Math.round((sig.hold_probability !== undefined ? sig.hold_probability : 0.34) * 100);

    if (confEl) {
      confEl.innerHTML = `<strong>${confPct}%</strong> <span style="font-size:10px; color:var(--text-muted); display:block; margin-top:2px;">(BUY ${buyProb}% • SELL ${sellProb}% • HOLD ${holdProb}%)</span>`;
    }
    if (setupEl) setupEl.textContent = sig.setup_type || (sig.is_unpredictable_regime ? 'UNPREDICTABLE' : 'CALIBRATED');
    if (volEl) volEl.textContent = `${((sig.volatility_forecast || sig.volatility_forecast_1h || 0.01) * 100).toFixed(2)}%`;
    if (regimeEl) regimeEl.textContent = sig.volatility_regime || 'NORMAL';
    if (agreeEl) agreeEl.textContent = sig.model_agreement_fraction || sig.model_agreement || '3/5';
    if (riskEl) {
      riskEl.textContent = sig.event_risk_high ? 'HIGH EVENT RISK' : 'NORMAL';
      riskEl.className = 'val ' + (sig.event_risk_high ? 'text-danger' : 'text-success');
    }
    if (badgeEl) badgeEl.textContent = 'ALPHAVOL CALIBRATED';

    // Unpredictable / High-Volatility alert banner insertion
    let unpredBanner = document.getElementById('unpredictableRegimeBanner');
    if (sig.is_unpredictable_regime) {
      if (!unpredBanner && actionEl && actionEl.parentElement) {
        unpredBanner = document.createElement('div');
        unpredBanner.id = 'unpredictableRegimeBanner';
        unpredBanner.className = 'unpredictable-regime-alert';
        actionEl.parentElement.insertAdjacentElement('beforebegin', unpredBanner);
      }
      if (unpredBanner) {
        unpredBanner.style.display = 'block';
        unpredBanner.innerHTML = `⚠️ <strong>HIGH VOLATILITY DETECTED</strong>: ${sanitize(sig.unpredictable_warning || 'Prediction confidence is unstable due to abnormal volatility or conflicting indicators. Recommended action: HOLD / WAIT FOR CONFIRMATION.')}`;
      }
    } else if (unpredBanner) {
      unpredBanner.style.display = 'none';
    }

    // Trade Plan
    const currSym = state.instruments.find(i => i.symbol === state.activeSymbol);
    const curr = currSym ? currSym.currency : 'USD';

    const planPrice = document.getElementById('terminalPlanPrice');
    const planEntry = document.getElementById('terminalPlanEntry');
    const planStop = document.getElementById('terminalPlanStop');
    const planTarget = document.getElementById('terminalPlanTarget');
    const planRR = document.getElementById('terminalPlanRR');
    const planStatus = document.getElementById('terminalPlanStatus');

    const entryPrice = sig.entry || sig.current_price || sig.trade_plan?.entry_price;
    const stopLoss = sig.stop_loss || sig.trade_plan?.stop_loss;
    const target1 = sig.target_1 || sig.take_profit || sig.trade_plan?.take_profit_target;
    const target2 = sig.target_2;
    const rr = sig.risk_reward || sig.trade_plan?.reward_risk_ratio || 2.0;

    if (planPrice) planPrice.textContent = formatCurrency(sig.current_price, curr);
    if (planEntry) planEntry.textContent = formatCurrency(entryPrice, curr);
    if (planStop) planStop.textContent = `${formatCurrency(stopLoss, curr)} (ATR/Swing)`;
    if (planTarget) planTarget.textContent = `${formatCurrency(target1, curr)}${target2 ? ' / ' + formatCurrency(target2, curr) : ''}`;
    if (planRR) planRR.textContent = `1:${Number(rr).toFixed(1)}`;
    if (planStatus) planStatus.textContent = sig.is_unpredictable_regime ? 'NO TRADE' : (sig.execution_status || 'READY');

    // Evidence & Risks lists
    const reasonsList = document.getElementById('terminalReasonsList');
    if (reasonsList) {
      const evidence = sig.evidence || [];
      const risks = sig.risks || [];
      if (evidence.length > 0 || risks.length > 0) {
        const evHtml = evidence.map(e => `<li class="evidence-item-plus">✔ ${sanitize(e)}</li>`).join('');
        const rskHtml = risks.map(r => `<li class="evidence-item-minus">⚠ ${sanitize(r)}</li>`).join('');
        reasonsList.innerHTML = `<ul class="evidence-list">${evHtml}${rskHtml}</ul>`;
      } else if (sig.reasons) {
        reasonsList.innerHTML = sig.reasons.map(r => `<li>${sanitize(r)}</li>`).join('');
      }
    }
  }

  // ========================================================================
  // PAGE 3: ASSET RESEARCH
  // ========================================================================
  async function loadResearchData() {
    const sym = state.activeSymbol;
    const item = state.instruments.find(i => i.symbol === sym) || { name: sym, type: 'crypto', currency: 'USD' };

    // Header banner
    const symEl = document.getElementById('researchSymbol');
    const nameEl = document.getElementById('researchName');
    const typeEl = document.getElementById('researchTypeBadge');
    const statusEl = document.getElementById('researchStatusBadge');
    const priceEl = document.getElementById('researchPrice');
    const chgEl = document.getElementById('researchChange');
    const volEl = document.getElementById('researchVol');

    if (symEl) symEl.textContent = sym;
    if (nameEl) nameEl.textContent = item.name;
    if (typeEl) typeEl.textContent = (item.type || 'ASSET').toUpperCase();

    const tick = state.tickers[sym];
    if (tick) {
      if (statusEl) {
        statusEl.textContent = tick.market_status || 'OPEN';
        statusEl.className = 'market-status-pill ' + (tick.market_status === 'CLOSED' ? 'status-closed' : 'status-open');
      }
      if (priceEl) priceEl.textContent = formatCurrency(tick.price, item.currency);
      if (chgEl) {
        chgEl.textContent = formatChange(tick.change_24h_pct);
        chgEl.className = 'research-change ' + ((tick.change_24h_pct || 0) >= 0 ? 'text-success' : 'text-danger');
      }
      if (volEl) volEl.textContent = `24h Vol: ${formatVolume(tick.volume_24h)}`;
    }

    // Tabs setup
    initResearchTabs();

    // 1. Fundamentals
    const fundRes = await API.get(`/api/v1/fundamentals/${encodeURIComponent(sym)}`);
    if (fundRes && fundRes.ok && fundRes.data) {
      state.fundamentals = fundRes.data;
      renderResearchOverview(fundRes.data, item, tick);
      renderResearchFundamentals(fundRes.data, item);
    } else {
      const fundGrid = document.getElementById('researchFundamentalsGrid');
      if (fundGrid) {
        fundGrid.innerHTML = `<div class="empty-state" style="padding: 16px; color: var(--text-muted);">Fundamentals temporarily unavailable for ${sanitize(sym)}</div>`;
      }
    }

    // 2. Technicals
    const indRes = await API.get(`/api/v1/market/${encodeURIComponent(sym)}/indicators?timeframe=1h&limit=100`);
    if (indRes && indRes.ok && indRes.data) {
      renderResearchTechnicals(indRes.data);
    }

    // 3. News
    const newsRes = await API.get(`/api/v1/news?symbol=${encodeURIComponent(sym)}&limit=6`);
    if (newsRes && newsRes.ok && newsRes.data) {
      renderResearchNews(newsRes.data);
    } else {
      const newsEl = document.getElementById('researchNewsList');
      if (newsEl) newsEl.innerHTML = `<div class="empty-state">No recent news for this instrument</div>`;
    }

    // 4. Events
    const eventsRes = await API.get(`/api/v1/events?symbol=${encodeURIComponent(sym)}&max_hours=120`);
    if (eventsRes && eventsRes.ok && eventsRes.data) {
      renderResearchEvents(eventsRes.data);
    } else {
      const evEl = document.getElementById('researchEventsList');
      if (evEl) evEl.innerHTML = `<div class="empty-state">No scheduled events for this instrument</div>`;
    }

    // 5. AI Synthesis Report trigger
    initResearchAiReportButton();
  }

  function initResearchTabs() {
    const bar = document.getElementById('researchTabsBar');
    if (!bar) return;
    bar.querySelectorAll('.research-tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        bar.querySelectorAll('.research-tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const tab = btn.getAttribute('data-tab');

        document.querySelectorAll('.research-tab-pane').forEach(p => p.classList.remove('active'));
        const pane = document.getElementById(`pane-${tab}`);
        if (pane) pane.classList.add('active');
      });
    });
  }

  function renderResearchOverview(fund, item, tick) {
    const profText = document.getElementById('researchProfileText');
    const exchTag = document.getElementById('researchExchangeTag');
    const metaClass = document.getElementById('resMetaClass');
    const metaCurr = document.getElementById('resMetaCurr');
    const meta52w = document.getElementById('resMeta52w');
    const metaFresh = document.getElementById('resMetaFreshness');

    if (profText) profText.textContent = fund.description || `${item.name} (${item.symbol}) traded on ${item.exchange}.`;
    if (exchTag) exchTag.textContent = item.exchange.toUpperCase();
    if (metaClass) metaClass.textContent = (item.type || 'Asset').toUpperCase();
    if (metaCurr) metaCurr.textContent = item.currency || 'USD';
    if (metaFresh) metaFresh.textContent = tick?.is_stale ? 'STALE' : 'LIVE';

    if (meta52w) {
      if (fund.low_52w && fund.high_52w) {
        meta52w.textContent = `${formatCurrency(fund.low_52w, item.currency)} – ${formatCurrency(fund.high_52w, item.currency)}`;
      } else {
        meta52w.textContent = '--';
      }
    }

    // Key Stats List
    const statsList = document.getElementById('researchKeyStats');
    if (statsList) {
      statsList.innerHTML = `
        <div class="fund-cell"><span class="f-lbl">Market Capitalization</span><span class="f-val">${fund.market_cap ? '$' + formatVolume(fund.market_cap) : 'Data unavailable'}</span></div>
        <div class="fund-cell"><span class="f-lbl">24h Trading Volume</span><span class="f-val">${tick?.volume_24h ? '$' + formatVolume(tick.volume_24h) : 'Data unavailable'}</span></div>
        <div class="fund-cell"><span class="f-lbl">52-Week High</span><span class="f-val">${fund.high_52w ? formatCurrency(fund.high_52w, item.currency) : 'Data unavailable'}</span></div>
        <div class="fund-cell"><span class="f-lbl">52-Week Low</span><span class="f-val">${fund.low_52w ? formatCurrency(fund.low_52w, item.currency) : 'Data unavailable'}</span></div>
      `;
    }
  }

  function renderResearchFundamentals(fund, item) {
    const grid = document.getElementById('researchFundamentalsGrid');
    if (!grid) return;

    const isCrypto = item.type === 'crypto';

    if (isCrypto) {
      grid.innerHTML = `
        <div class="fund-cell"><span class="f-lbl">Circulating Supply</span><span class="f-val">${fund.circulating_supply ? formatVolume(fund.circulating_supply) + ' UNITS' : 'Data unavailable'}</span></div>
        <div class="fund-cell"><span class="f-lbl">Total Supply</span><span class="f-val">${fund.total_supply ? formatVolume(fund.total_supply) + ' UNITS' : 'Data unavailable'}</span></div>
        <div class="fund-cell"><span class="f-lbl">All-Time High</span><span class="f-val">${fund.all_time_high ? formatCurrency(fund.all_time_high) : 'Data unavailable'}</span></div>
        <div class="fund-cell"><span class="f-lbl">Network Hash Rate</span><span class="f-val">${fund.hash_rate || 'Data unavailable'}</span></div>
        <div class="fund-cell"><span class="f-lbl">Active Addresses</span><span class="f-val">${fund.active_addresses ? Number(fund.active_addresses).toLocaleString() : 'Data unavailable'}</span></div>
        <div class="fund-cell"><span class="f-lbl">Fear & Greed Index</span><span class="f-val">${fund.sentiment_score || 'Data unavailable'}</span></div>
        <div class="fund-cell"><span class="f-lbl">P/E Ratio</span><span class="f-val text-muted">Data unavailable (Crypto)</span></div>
        <div class="fund-cell"><span class="f-lbl">Dividend Yield</span><span class="f-val text-muted">Data unavailable (Crypto)</span></div>
      `;
    } else {
      grid.innerHTML = `
        <div class="fund-cell"><span class="f-lbl">Market Capitalization</span><span class="f-val">${fund.market_cap ? '$' + formatVolume(fund.market_cap) : 'Data unavailable'}</span></div>
        <div class="fund-cell"><span class="f-lbl">Trailing P/E Ratio</span><span class="f-val">${fund.trailing_pe ? fund.trailing_pe.toFixed(2) : 'Data unavailable'}</span></div>
        <div class="fund-cell"><span class="f-lbl">Forward P/E Ratio</span><span class="f-val">${fund.forward_pe ? fund.forward_pe.toFixed(2) : 'Data unavailable'}</span></div>
        <div class="fund-cell"><span class="f-lbl">Earnings Per Share (EPS)</span><span class="f-val">${fund.eps !== undefined ? formatCurrency(fund.eps, item.currency) : 'Data unavailable'}</span></div>
        <div class="fund-cell"><span class="f-lbl">Annual Revenue</span><span class="f-val">${fund.revenue ? '$' + formatVolume(fund.revenue) : 'Data unavailable'}</span></div>
        <div class="fund-cell"><span class="f-lbl">Profit Margin</span><span class="f-val">${fund.profit_margin ? (fund.profit_margin * 100).toFixed(2) + '%' : 'Data unavailable'}</span></div>
        <div class="fund-cell"><span class="f-lbl">Return on Equity (ROE)</span><span class="f-val">${fund.roe ? (fund.roe * 100).toFixed(2) + '%' : 'Data unavailable'}</span></div>
        <div class="fund-cell"><span class="f-lbl">Debt to Equity</span><span class="f-val">${fund.debt_to_equity ? fund.debt_to_equity.toFixed(2) : 'Data unavailable'}</span></div>
        <div class="fund-cell"><span class="f-lbl">Dividend Yield</span><span class="f-val">${fund.dividend_yield ? (fund.dividend_yield * 100).toFixed(2) + '%' : 'Data unavailable'}</span></div>
        <div class="fund-cell"><span class="f-lbl">Beta (vs Benchmark)</span><span class="f-val">${fund.beta ? fund.beta.toFixed(2) : 'Data unavailable'}</span></div>
      `;
    }
  }

  function renderResearchTechnicals(ind) {
    const grid = document.getElementById('researchTechnicalGrid');
    if (!grid || !ind.current) return;
    const c = ind.current;

    grid.innerHTML = `
      <div class="fund-cell"><span class="f-lbl">RSI (14)</span><span class="f-val">${c.rsi !== undefined ? c.rsi.toFixed(1) : '--'}</span><span class="f-lbl text-cyan">${c.rsi > 70 ? 'Overbought' : (c.rsi < 30 ? 'Oversold' : 'Neutral Zone')}</span></div>
      <div class="fund-cell"><span class="f-lbl">MACD Histogram</span><span class="f-val ${c.macd_hist >= 0 ? 'text-success' : 'text-danger'}">${c.macd_hist !== undefined ? c.macd_hist.toFixed(2) : '--'}</span><span class="f-lbl">${c.macd_hist >= 0 ? 'Bullish Drift' : 'Bearish Drift'}</span></div>
      <div class="fund-cell"><span class="f-lbl">Trend Regime</span><span class="f-val text-cyan">${c.trend_regime || 'NEUTRAL'}</span><span class="f-lbl">Moving Average Filter</span></div>
      <div class="fund-cell"><span class="f-lbl">EMA 20</span><span class="f-val">${c.ema20 !== undefined ? formatCurrency(c.ema20) : '--'}</span><span class="f-lbl">Short-Term Baseline</span></div>
      <div class="fund-cell"><span class="f-lbl">EMA 50</span><span class="f-val">${c.ema50 !== undefined ? formatCurrency(c.ema50) : '--'}</span><span class="f-lbl">Intermediate Baseline</span></div>
      <div class="fund-cell"><span class="f-lbl">EMA 200</span><span class="f-val">${c.ema200 !== undefined ? formatCurrency(c.ema200) : '--'}</span><span class="f-lbl">Macro Institutional Level</span></div>
      <div class="fund-cell"><span class="f-lbl">Bollinger Bandwidth</span><span class="f-val">${c.bb_bandwidth ? c.bb_bandwidth.toFixed(2) + '%' : '--'}</span><span class="f-lbl">Volatility Envelope</span></div>
      <div class="fund-cell"><span class="f-lbl">ATR (14) Volatility</span><span class="f-val">${c.atr !== undefined ? formatCurrency(c.atr) : '--'}</span><span class="f-lbl">Dynamic Stop Distance</span></div>
    `;
  }

  function renderResearchNews(newsList) {
    const container = document.getElementById('researchNewsList');
    if (!container) return;
    if (newsList.length === 0) {
      container.innerHTML = '<div class="empty-state">No recent headlines tagged for this symbol.</div>';
      return;
    }

    container.innerHTML = newsList.map(n => `
      <div class="news-article-card">
        <div class="article-meta-row">
          <span>${sanitize(n.source)} • ${sanitize(n.published_time || 'Recent')}</span>
          <span class="sentiment-badge ${(n.sentiment || 'neutral').toLowerCase()}">${sanitize(n.sentiment || 'NEUTRAL')}</span>
        </div>
        <h4 class="article-headline">${sanitize(n.headline || n.title)}</h4>
        <p class="article-snippet">${sanitize(n.summary || '')}</p>
      </div>
    `).join('');
  }

  function renderResearchEvents(eventsList) {
    const container = document.getElementById('researchEventsList');
    if (!container) return;
    if (eventsList.length === 0) {
      container.innerHTML = '<div class="empty-state">No high-impact macro dates or token milestones within the next 120 hours.</div>';
      return;
    }

    container.innerHTML = eventsList.map(ev => `
      <div class="event-feed-card">
        <div class="event-card-top">
          <span class="event-title">${sanitize(ev.title || ev.event)}</span>
          <span class="impact-badge ${(ev.importance || 'MED').toLowerCase()}">${sanitize(ev.importance || 'MED')}</span>
        </div>
        <div class="event-data-row">
          <span>Time: ${sanitize(ev.scheduled_time || ev.time || 'Upcoming')}</span>
          <span>Country: ${sanitize(ev.country || 'Global')}</span>
          <span>Impact Risk: <strong>${sanitize(ev.event_risk || 'Normal')}</strong></span>
        </div>
      </div>
    `).join('');
  }

  function initResearchAiReportButton() {
    const btn = document.getElementById('btnRefreshAiReport');
    const body = document.getElementById('researchAiReport');
    if (!btn || !body) return;

    btn.addEventListener('click', async () => {
      body.innerHTML = '<p class="loading-placeholder">Synthesizing quantitative metrics, valuation parameters, and recent news...</p>';
      const q = `Provide a comprehensive quantitative research note on ${state.activeSymbol}. Include technical momentum, valuation fundamentals, and risk regime.`;
      const res = await API.post('/api/v1/chat', { query: q, symbol: state.activeSymbol });
      if (res && res.response) {
        body.innerHTML = formatAiResponse(res.response);
      } else {
        body.innerHTML = '<p class="text-danger">Failed to generate AI report.</p>';
      }
    });
  }

  // ========================================================================
  // PAGE 4: NEWS & EVENTS WORKSPACE
  // ========================================================================
  async function loadNewsEventsPage() {
    initNewsEventsFilters();
    loadFullNewsFeed();
    loadFullEventsFeed();
  }

  function initNewsEventsFilters() {
    const catPills = document.getElementById('newsCategoryPills');
    if (catPills) {
      catPills.querySelectorAll('.cat-pill').forEach(btn => {
        btn.addEventListener('click', () => {
          catPills.querySelectorAll('.cat-pill').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          loadFullNewsFeed(btn.getAttribute('data-cat'));
        });
      });
    }

    const searchInput = document.getElementById('newsSearchInput');
    if (searchInput) {
      searchInput.addEventListener('input', () => {
        filterNewsCards(searchInput.value.trim().toLowerCase());
      });
    }

    // Event filters
    const timeTabs = document.getElementById('eventTimeTabs');
    if (timeTabs) {
      timeTabs.querySelectorAll('.event-time-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          timeTabs.querySelectorAll('.event-time-btn').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          loadFullEventsFeed(btn.getAttribute('data-time'));
        });
      });
    }
  }

  async function loadFullNewsFeed(category = 'ALL') {
    const container = document.getElementById('fullNewsFeed');
    const countEl = document.getElementById('newsTotalCount');
    if (!container) return;

    container.innerHTML = '<div class="loading-placeholder">Fetching news articles...</div>';
    const catParam = category === 'ALL' ? '' : `&category=${encodeURIComponent(category)}`;
    const res = await API.get(`/api/v1/news?limit=25${catParam}`);
    const articles = (res && res.ok && Array.isArray(res.data)) ? res.data : null;

    if (articles && articles.length > 0) {
      state.news = articles;
      if (countEl) countEl.textContent = `${articles.length} Articles`;
      renderNewsCards(articles);
    } else {
      container.innerHTML = `<div class="error-state" style="padding: 24px; text-align: center; color: var(--text-muted);">
        <span>${res?.ok ? 'No news articles found.' : 'Failed to fetch news feed.'}</span>
        ${!res?.ok ? `<button class="btn btn-xs btn-outline" onclick="window.AlphaVol.loadNewsEventsPage()" style="margin-left: 10px; cursor: pointer;">Retry</button>` : ''}
      </div>`;
    }
  }

  function renderNewsCards(articles) {
    const container = document.getElementById('fullNewsFeed');
    if (!container) return;

    container.innerHTML = articles.map(a => {
      const sentClass = (a.sentiment || 'neutral').toLowerCase();
      const impactClass = (a.importance || 'med').toLowerCase().includes('high') ? 'high' : 'med';

      return `
        <div class="news-article-card" data-headline="${sanitize(a.headline || a.title)}">
          <div class="article-meta-row">
            <span>${sanitize(a.source)} • ${sanitize(a.published_time || 'Today')}</span>
            <div style="display: flex; gap: 4px;">
              <span class="impact-badge ${impactClass}">${sanitize(a.importance || 'NORMAL')}</span>
              <span class="sentiment-badge ${sentClass}">${sanitize(a.sentiment || 'NEUTRAL')}</span>
            </div>
          </div>
          <h4 class="article-headline">${sanitize(a.headline || a.title)}</h4>
          <p class="article-snippet">${sanitize(a.summary || '')}</p>
          <div class="article-footer">
            <span class="badge-tag">${sanitize(a.category || 'MARKET')}</span>
            <a href="${sanitize(a.url || '#')}" target="_blank" rel="noopener noreferrer" class="btn-text-link">Read Full ↗</a>
          </div>
        </div>
      `;
    }).join('');
  }

  function filterNewsCards(query) {
    const cards = document.querySelectorAll('#fullNewsFeed .news-article-card');
    cards.forEach(card => {
      const headline = (card.getAttribute('data-headline') || '').toLowerCase();
      card.style.display = headline.includes(query) ? 'flex' : 'none';
    });
  }

  async function loadFullEventsFeed(timeWindow = 'today') {
    const container = document.getElementById('fullEventsFeed');
    const badge = document.getElementById('globalEventRiskBadge');
    if (!container) return;

    container.innerHTML = '<div class="loading-placeholder">Loading calendar...</div>';
    let maxHours = 24.0;
    if (timeWindow === 'tomorrow') maxHours = 48.0;
    else if (timeWindow === 'week') maxHours = 96.0;
    else if (timeWindow === 'all') maxHours = 168.0;

    const res = await API.get(`/api/v1/events?max_hours=${maxHours}`);
    const events = (res && res.ok && Array.isArray(res.data)) ? res.data : null;
    if (events && events.length > 0) {
      state.events = events;
      const hasHighRisk = events.some(e => e.importance === 'HIGH' || e.event_risk === 'HIGH');
      if (badge) {
        badge.textContent = hasHighRisk ? 'RISK: HIGH ⚠️' : 'RISK: NORMAL';
        badge.className = 'event-risk-pill ' + (hasHighRisk ? 'impact-high' : '');
      }

      container.innerHTML = events.map(ev => {
        const impClass = (ev.importance || 'MED').toLowerCase();
        return `
          <div class="event-feed-card">
            <div class="event-card-top">
              <span class="event-title">${sanitize(ev.title || ev.event)}</span>
              <span class="impact-badge ${impClass}">${sanitize(ev.importance || 'MED')}</span>
            </div>
            <div class="event-data-row">
              <span>Time: <strong>${sanitize(ev.scheduled_time || ev.time || 'Upcoming')}</strong></span>
              <span>Country: ${sanitize(ev.country || 'Global')}</span>
              <span>Risk: <strong>${sanitize(ev.event_risk || 'Normal')}</strong></span>
            </div>
          </div>
        `;
      }).join('');
    } else {
      container.innerHTML = `<div class="empty-state">${res?.ok ? 'No economic events scheduled in this time horizon.' : 'Macro calendar temporarily unavailable.'}</div>`;
    }
  }

  let isAiExecuting = false;

  function setAiPendingState(pending, activePrompt = '') {
    isAiExecuting = pending;

    const fullInput = document.getElementById('fullAiChatInput');
    const fullSendBtn = document.getElementById('btnFullAiSend');
    const fullChips = document.querySelectorAll('#fullAiQuickChips .action-chip');
    const presetBtns = document.querySelectorAll('.preset-btn');
    const engineTag = document.getElementById('aiEngineTag');

    const drawerInput = document.getElementById('terminalDrawerInput');
    const drawerSendBtn = document.querySelector('#terminalDrawerForm .drawer-send-btn');
    const drawerChips = document.querySelectorAll('#terminalAiDrawer .drawer-chip');

    if (pending) {
      if (fullInput) fullInput.disabled = true;
      if (fullSendBtn) {
        fullSendBtn.disabled = true;
        fullSendBtn.innerHTML = '<span class="spinner-sm"></span> Evaluating...';
      }
      fullChips.forEach(c => (c.disabled = true));
      presetBtns.forEach(p => (p.disabled = true));
      if (engineTag) {
        if (!engineTag.dataset.origText) engineTag.dataset.origText = engineTag.textContent;
        engineTag.textContent = 'Querying live telemetry & models...';
        engineTag.classList.add('evaluating');
      }

      if (drawerInput) drawerInput.disabled = true;
      if (drawerSendBtn) {
        drawerSendBtn.disabled = true;
        drawerSendBtn.textContent = 'Evaluating...';
      }
      drawerChips.forEach(c => (c.disabled = true));
    } else {
      if (fullInput) {
        fullInput.disabled = false;
        fullInput.focus();
      }
      if (fullSendBtn) {
        fullSendBtn.disabled = false;
        fullSendBtn.innerHTML = 'Send ↵';
      }
      fullChips.forEach(c => (c.disabled = false));
      presetBtns.forEach(p => (p.disabled = false));
      if (engineTag) {
        engineTag.textContent = engineTag.dataset.origText || 'Deterministic Multi-Asset Engine';
        engineTag.classList.remove('evaluating');
      }

      if (drawerInput) {
        drawerInput.disabled = false;
        drawerInput.focus();
      }
      if (drawerSendBtn) {
        drawerSendBtn.disabled = false;
        drawerSendBtn.textContent = 'Send';
      }
      drawerChips.forEach(c => (c.disabled = false));
    }
  }

  function appendAiMessageToContainer(containerId, role, text, meta = null, isLoading = false, retryFn = null) {
    const thread = document.getElementById(containerId);
    if (!thread) return;

    const msg = document.createElement('div');
    msg.className = `msg ${role}` + (meta && meta.is_error ? ' msg-error' : '') + (isLoading ? ' msg-loading' : '');

    if (role === 'user') {
      msg.innerHTML = `<div class="msg-bubble">${sanitize(text)}</div>`;
    } else if (isLoading) {
      msg.innerHTML = `<div class="msg-bubble msg-bubble-loading">${text}</div>`;
    } else {
      let headerHtml = '';
      let footerHtml = '';

      if (meta) {
        if (meta.is_error) {
          headerHtml = `
            <div class="msg-meta-bar error-meta">
              <span class="meta-badge badge-error">⚠️ REQUEST ERROR</span>
              ${meta.retry_prompt ? `<button type="button" class="btn-retry-query" data-prompt="${sanitize(meta.retry_prompt)}">Retry ↺</button>` : ''}
            </div>
          `;
        } else {
          let badgeClass = 'badge-deterministic';
          let badgeText = '⚡ Grounded Deterministic Analysis';
          let badgeTitle = 'Telemetry-grounded quantitative analysis generated without external LLM dependencies';

          if (meta.fallback_used) {
            badgeClass = 'badge-fallback';
            badgeText = '⚠️ FALLBACK (Deterministic Engine)';
            badgeTitle = meta.fallback_reason ? `Provider fallback: ${sanitize(meta.fallback_reason)}` : 'External provider unavailable, fell back to deterministic engine';
          } else if (meta.provider && !meta.provider.includes('deterministic') && meta.provider !== 'error_handler') {
            badgeClass = 'badge-external';
            badgeText = `🤖 AI Model: ${sanitize(meta.provider.toUpperCase())}`;
            badgeTitle = `Synthesized via ${sanitize(meta.provider)} grounded in live system telemetry`;
          }

          const statusClass = meta.data_status === 'UNAVAILABLE' ? 'status-unavail' : 'status-live';
          const statusText = meta.data_status || 'AVAILABLE';

          headerHtml = `
            <div class="msg-meta-bar">
              <span class="meta-badge ${badgeClass}" title="${badgeTitle}">${badgeText}</span>
              <span class="meta-status-badge ${statusClass}">${statusText}</span>
              ${meta.timestamp_ist ? `<span class="meta-time">${sanitize(meta.timestamp_ist)}</span>` : ''}
            </div>
          `;

          if (meta.fallback_used && meta.fallback_reason) {
            headerHtml += `
              <div class="msg-fallback-notice">
                <small>ℹ️ Note: External provider was unavailable (${sanitize(meta.fallback_reason)}). Deterministic reasoning engine provided verified telemetry.</small>
              </div>
            `;
          }

          if (meta.tools_called && meta.tools_called.length > 0) {
            footerHtml = `
              <div class="msg-tools-bar">
                <span class="tools-label">Telemetry Verified:</span>
                ${meta.tools_called.map(t => `<span class="tool-tag">${sanitize(t)}</span>`).join('')}
              </div>
            `;
          }
        }
      }

      msg.innerHTML = `
        <div class="msg-bubble">
          ${headerHtml}
          <div class="msg-body">${formatAiResponse(text)}</div>
          ${footerHtml}
        </div>
      `;

      const retryBtn = msg.querySelector('.btn-retry-query');
      if (retryBtn && retryFn) {
        retryBtn.addEventListener('click', () => {
          const p = retryBtn.getAttribute('data-prompt');
          if (p && !isAiExecuting) {
            retryFn('user', p);
            executeAiQuery(p, retryFn, containerId);
          }
        });
      }
    }

    thread.appendChild(msg);
    thread.scrollTop = thread.scrollHeight;
  }

  function appendMainAiMessage(role, text, meta = null, isLoading = false) {
    appendAiMessageToContainer('fullAiChatThread', role, text, meta, isLoading, appendMainAiMessage);
  }

  async function executeAiQuery(prompt, appendFn, containerId = 'fullAiChatThread') {
    if (isAiExecuting) return;

    const cleanPrompt = (prompt || '').trim();
    if (!cleanPrompt) return;

    setAiPendingState(true, cleanPrompt);

    const loadingId = 'ai-load-' + Date.now();
    const loadingHtml = `
      <div class="ai-loading-card" id="${loadingId}">
        <div class="ai-loading-spinner"></div>
        <div class="ai-loading-text">
          <strong>Evaluating live market telemetry...</strong>
          <span>Querying indicators, volatility models, and macro events for ${sanitize(state.activeSymbol)}</span>
        </div>
      </div>
    `;
    appendFn('assistant', loadingHtml, null, true);

    try {
      const res = await API.post('/api/v1/chat', { query: cleanPrompt, symbol: state.activeSymbol });

      const loadEl = document.getElementById(loadingId);
      if (loadEl && loadEl.closest('.msg')) {
        loadEl.closest('.msg').remove();
      }

      if (res && res.response) {
        appendFn('assistant', res.response, {
          provider: res.provider || 'unknown',
          provider_configured: res.provider_configured,
          fallback_used: Boolean(res.fallback_used),
          fallback_reason: res.fallback_reason,
          data_status: res.data_status || 'AVAILABLE',
          tools_called: res.tools_called || [],
          timestamp_ist: res.timestamp_ist,
        });
      } else {
        appendFn('assistant', 'Unable to retrieve quantitative analysis at this moment.', {
          is_error: true,
          error_msg: 'Empty or invalid response from server',
          retry_prompt: cleanPrompt,
        });
      }
    } catch (e) {
      const loadEl = document.getElementById(loadingId);
      if (loadEl && loadEl.closest('.msg')) {
        loadEl.closest('.msg').remove();
      }

      const cleanErr = (e.message || 'Server connection error').replace(/[A-Za-z0-9_\-\.]{25,}/g, '[REDACTED]');
      appendFn('assistant', `Processing Error: ${cleanErr}. Market telemetry remains available via standard terminal routes.`, {
        is_error: true,
        error_msg: cleanErr,
        retry_prompt: cleanPrompt,
        data_status: 'UNAVAILABLE',
      });
    } finally {
      setAiPendingState(false);
    }
  }

  function formatAiResponse(text) {
    if (!text) return '';
    let html = text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/```([a-zA-Z]*)\n([\s\S]*?)```/g, '<pre><code class="lang-$1">$2</code></pre>')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/^### (.*$)/gim, '<h4>$1</h4>')
      .replace(/^## (.*$)/gim, '<h3>$1</h3>')
      .replace(/^# (.*$)/gim, '<h2>$1</h2>')
      .replace(/^\> (.*$)/gim, '<blockquote>$1</blockquote>')
      .replace(/\n\n/g, '<br><br>')
      .replace(/\n/g, '<br>');
    return html;
  }

  // ========================================================================
  // PAGE 5: AI RESEARCH ASSISTANT WORKSPACE
  // ========================================================================
  function initAiAssistantPage() {
    const form = document.getElementById('fullAiChatForm');
    const input = document.getElementById('fullAiChatInput');
    const chipsBar = document.getElementById('fullAiQuickChips');
    const newChatBtn = document.getElementById('btnNewAiChat');
    const clearBtn = document.getElementById('btnClearChatHistory');

    if (form && input) {
      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (isAiExecuting) return;
        const q = input.value.trim();
        if (!q) return;
        appendMainAiMessage('user', q);
        input.value = '';
        await executeAiQuery(q, appendMainAiMessage, 'fullAiChatThread');
      });
    }

    if (chipsBar) {
      chipsBar.querySelectorAll('.action-chip').forEach(btn => {
        btn.addEventListener('click', async () => {
          if (isAiExecuting) return;
          const q = btn.getAttribute('data-q');
          if (!q) return;
          appendMainAiMessage('user', q);
          await executeAiQuery(q, appendMainAiMessage, 'fullAiChatThread');
        });
      });
    }

    // Preset prompts in sidebar
    document.querySelectorAll('.preset-btn').forEach(btn => {
      btn.addEventListener('click', async () => {
        if (isAiExecuting) return;
        const q = btn.getAttribute('data-q');
        if (!q) return;
        appendMainAiMessage('user', q);
        await executeAiQuery(q, appendMainAiMessage, 'fullAiChatThread');
      });
    });

    if (newChatBtn) {
      newChatBtn.addEventListener('click', () => {
        const thread = document.getElementById('fullAiChatThread');
        if (thread) {
          thread.innerHTML = `
            <div class="msg assistant">
              <div class="msg-bubble">
                Started a new quantitative session for <strong>${state.activeSymbol}</strong>. Ask any question or select a preset prompt!
              </div>
            </div>
          `;
        }
      });
    }

    if (clearBtn) {
      clearBtn.addEventListener('click', () => {
        const thread = document.getElementById('fullAiChatThread');
        if (thread) thread.innerHTML = '';
      });
    }
  }

  function updateAiContextPanel() {
    const sym = state.activeSymbol;
    const tick = state.tickers[sym];
    const ind = state.indicators?.current;
    const sig = state.signal;

    const ctxSym = document.getElementById('ctxSymbol');
    const ctxPrice = document.getElementById('ctxPrice');
    const ctxChange = document.getElementById('ctxChange');
    const ctxRegime = document.getElementById('ctxRegime');
    const ctxRsi = document.getElementById('ctxRsi');
    const ctxMacd = document.getElementById('ctxMacd');
    const ctxSig = document.getElementById('ctxSignal');
    const ctxConf = document.getElementById('ctxConfidence');
    const ctxVol = document.getElementById('ctxVol');
    const chatActiveSym = document.getElementById('aiChatActiveSym');

    if (ctxSym) ctxSym.textContent = sym;
    if (chatActiveSym) chatActiveSym.textContent = `Target: ${sym}`;

    if (tick) {
      if (ctxPrice) ctxPrice.textContent = formatCurrency(tick.price, tick.currency);
      if (ctxChange) {
        ctxChange.textContent = formatChange(tick.change_24h_pct);
        ctxChange.className = 'ctx-change ' + ((tick.change_24h_pct || 0) >= 0 ? 'text-success' : 'text-danger');
      }
    }

    if (ind) {
      if (ctxRegime) ctxRegime.textContent = ind.trend_regime || 'NEUTRAL';
      if (ctxRsi) ctxRsi.textContent = ind.rsi !== undefined ? ind.rsi.toFixed(1) : '--';
      if (ctxMacd) ctxMacd.textContent = ind.macd_hist !== undefined ? ind.macd_hist.toFixed(2) : '--';
    }

    if (sig) {
      if (ctxSig) {
        ctxSig.textContent = sig.action || 'HOLD';
        ctxSig.className = `badge-${(sig.action || 'hold').toLowerCase().replace('_', '-')}`;
      }
      if (ctxConf) ctxConf.textContent = `${Math.round((sig.confidence || 0.5) * 100)}%`;
      if (ctxVol) ctxVol.textContent = `${((sig.volatility_forecast_1h || 0.01) * 100).toFixed(2)}%`;
    }
  }

  // ========================================================================
  // PAGE 6: PAPER TRADING WORKSPACE
  // ========================================================================
  function initPaperTrading() {
    initPaperOrderForm();
    initPaperTabs();
    initPaperReset();
    initPaperAuth();
  }

  function getPaperAuthHeaders() {
    if (state.paperApiKey) {
      return { 'X-API-Key': state.paperApiKey };
    }
    return {};
  }

  function initPaperAuth() {
    const input = document.getElementById('paperApiKeyInput');
    const unlockBtn = document.getElementById('btnUnlockPaper');
    const clearBtn = document.getElementById('btnClearPaperKey');
    const inputGroup = document.getElementById('paperAuthInputGroup');
    const activeGroup = document.getElementById('paperAuthActiveGroup');
    const badge = document.getElementById('paperAuthBadge');
    const msgEl = document.getElementById('paperOrderStatusMsg');

    if (unlockBtn && input) {
      unlockBtn.addEventListener('click', () => {
        const key = input.value.trim();
        if (!key) {
          if (msgEl) msgEl.innerHTML = '<span class="text-amber">⚠️ Please enter an AGY_API_KEY.</span>';
          return;
        }
        state.paperApiKey = key;
        input.value = '';
        if (inputGroup) inputGroup.style.display = 'none';
        if (activeGroup) activeGroup.style.display = 'flex';
        if (badge) {
          badge.textContent = '🟢 Authorized';
          badge.style.background = 'rgba(14, 203, 129, 0.15)';
          badge.style.color = 'var(--color-green)';
        }
        if (msgEl) msgEl.innerHTML = '<span class="text-success">✓ Key set in volatile memory for this session.</span>';
      });
    }

    if (clearBtn) {
      clearBtn.addEventListener('click', () => {
        state.paperApiKey = '';
        if (inputGroup) inputGroup.style.display = 'flex';
        if (activeGroup) activeGroup.style.display = 'none';
        if (badge) {
          badge.textContent = '🔒 Read-Only';
          badge.style.background = 'rgba(245, 158, 11, 0.15)';
          badge.style.color = 'var(--color-amber)';
        }
        if (msgEl) msgEl.innerHTML = '<span class="text-dim">Key cleared from memory.</span>';
      });
    }
  }

  function renderPaperHud(portfolio) {
    if (!portfolio) return;
    const equityEl = document.getElementById('paperEquityVal');
    const returnSubEl = document.getElementById('paperReturnSub');
    const cashEl = document.getElementById('paperCashVal');
    const positionsEl = document.getElementById('paperPositionsVal');
    const openCountEl = document.getElementById('paperOpenCount');
    const pnlEl = document.getElementById('paperPnlVal');
    const pnlPctEl = document.getElementById('paperPnlPct');

    const equity = portfolio.total_equity ?? portfolio.portfolio_value ?? portfolio.equity;
    const cash = portfolio.cash;
    const positionsVal = portfolio.total_positions_value ?? portfolio.position_value ?? 0.0;
    const uPnl = portfolio.unrealized_pnl ?? 0.0;
    const retPct = portfolio.total_return_pct ?? portfolio.return_pct ?? 0.0;
    const openCount = (portfolio.positions || []).length;

    if (equityEl) equityEl.textContent = formatCurrency(equity);
    if (returnSubEl) {
      returnSubEl.textContent = `${formatChange(retPct)} Net Return`;
      returnSubEl.className = 'hud-sub ' + (retPct >= 0 ? 'text-success' : 'text-danger');
    }
    if (cashEl) cashEl.textContent = formatCurrency(cash);
    if (positionsEl) positionsEl.textContent = formatCurrency(positionsVal);
    if (openCountEl) openCountEl.textContent = `${openCount} Active Positions`;
    if (pnlEl) {
      pnlEl.textContent = formatCurrency(uPnl);
      pnlEl.className = 'hud-val ' + (uPnl >= 0 ? 'text-success' : 'text-danger');
    }
    if (pnlPctEl) {
      const pnlPct = portfolio.unrealized_pnl_pct ?? (positionsVal > 0 ? (uPnl / positionsVal) * 100 : 0.0);
      pnlPctEl.textContent = formatChange(pnlPct);
      pnlPctEl.className = 'hud-sub ' + (pnlPct >= 0 ? 'text-success' : 'text-danger');
    }
  }

  async function loadPaperData() {
    populatePaperSymbolSelect();
    const res = await API.get(`/api/v1/paper/account?symbol=${encodeURIComponent(state.activeSymbol)}`);
    if (res && res.ok && res.data && res.data.portfolio) {
      state.paperPortfolio = res.data.portfolio;
      renderPaperHud(res.data.portfolio);
      renderPaperPositions(res.data.portfolio);
    } else {
      console.warn(`[Paper Data Warning]: ${res?.error || 'Unavailable'}`);
    }
    updatePaperFormMarkPrice();
  }

  function populatePaperSymbolSelect() {
    const sel = document.getElementById('paperOrderSymbol');
    if (!sel) return;
    sel.innerHTML = state.instruments.map(i => `
      <option value="${sanitize(i.symbol)}" ${i.symbol === state.activeSymbol ? 'selected' : ''}>
        ${sanitize(i.symbol)} (${sanitize(i.name)})
      </option>
    `).join('');

    sel.addEventListener('change', () => {
      setActiveSymbol(sel.value, false);
      updatePaperFormMarkPrice();
    });
  }

  function updatePaperFormMarkPrice() {
    const priceEl = document.getElementById('paperOrderMarkPrice');
    const tick = state.tickers[state.activeSymbol];
    if (priceEl && tick) {
      priceEl.textContent = formatCurrency(tick.price, tick.currency);
      calculatePaperEstimate();
    }
  }

  function initPaperOrderForm() {
    const form = document.getElementById('paperOrderForm');
    const qtyInput = document.getElementById('paperOrderQty');
    const sideBtns = document.querySelectorAll('.order-side-buttons .btn-side');
    const selectedSideInput = document.getElementById('selectedOrderSide');
    const sizingPills = document.querySelectorAll('#paperSizingPills .size-pill');

    sideBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        sideBtns.forEach(b => {
          b.classList.remove('active-buy', 'active-sell', 'active-reduce');
        });
        const side = btn.getAttribute('data-side');
        selectedSideInput.value = side;
        if (side === 'BUY') btn.classList.add('active-buy');
        else if (side === 'SELL') btn.classList.add('active-sell');
        else if (side === 'REDUCE') btn.classList.add('active-reduce');
      });
    });

    sizingPills.forEach(pill => {
      pill.addEventListener('click', () => {
        sizingPills.forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        const pct = Number(pill.getAttribute('data-pct')) / 100;
        const tick = state.tickers[state.activeSymbol];
        const cash = state.paperPortfolio?.cash || 10000;
        if (tick && tick.price > 0) {
          const qty = ((cash * pct) * 0.98) / tick.price;
          if (qtyInput) {
            qtyInput.value = qty >= 1 ? qty.toFixed(2) : qty.toFixed(4);
            calculatePaperEstimate();
          }
        }
      });
    });

    if (qtyInput) {
      qtyInput.addEventListener('input', calculatePaperEstimate);
    }

    if (form) {
      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const side = selectedSideInput.value;
        const qty = parseFloat(qtyInput.value);
        const tick = state.tickers[state.activeSymbol] || { price: 100 };
        const msgEl = document.getElementById('paperOrderStatusMsg');

        if (!state.paperApiKey) {
          msgEl.innerHTML = '<span class="text-amber">🔒 Authentication required: Please enter your AGY_API_KEY in the Operator Authorization panel above to execute simulated trades.</span>';
          return;
        }

        try {
          msgEl.innerHTML = '<span class="text-cyan">Executing simulated order...</span>';
          const res = await API.post(
            '/api/v1/paper/order',
            {
              side: side,
              quantity: qty,
              price: tick.price,
              symbol: state.activeSymbol,
            },
            getPaperAuthHeaders()
          );
          msgEl.innerHTML = '<span class="text-success">✓ Order successfully executed and recorded in ledger!</span>';
          loadPaperData();
        } catch (err) {
          if (err.status === 401 || (err.message && err.message.toLowerCase().includes('x-api-key'))) {
            msgEl.innerHTML = `<span class="text-danger">🔒 Authorization failed (401): ${sanitize(err.message)}</span>`;
          } else {
            msgEl.innerHTML = `<span class="text-danger">Failed to place order: ${sanitize(err.message)}</span>`;
          }
        }
      });
    }
  }

  function calculatePaperEstimate() {
    const qtyInput = document.getElementById('paperOrderQty');
    const valEl = document.getElementById('estOrderValue');
    const remEl = document.getElementById('estOrderRemainingCash');
    const feeEl = document.getElementById('estOrderFee');
    const stopEl = document.getElementById('estOrderStop');
    const targetEl = document.getElementById('estOrderTarget');

    const tick = state.tickers[state.activeSymbol];
    if (!tick || !qtyInput) return;

    const qty = parseFloat(qtyInput.value) || 0;
    const estVal = qty * tick.price;
    const fee = estVal * 0.001; // 10 bps
    const currentCash = state.paperPortfolio?.cash !== undefined ? state.paperPortfolio.cash : 10000;
    const remainingCash = Math.max(0, currentCash - estVal - fee);

    if (valEl) valEl.textContent = formatCurrency(estVal, tick.currency);
    if (remEl) remEl.textContent = formatCurrency(remainingCash, tick.currency);
    if (feeEl) feeEl.textContent = formatCurrency(fee, tick.currency);

    const sig = state.signal;
    if (sig) {
      if (stopEl) stopEl.textContent = formatCurrency(sig.stop_loss, tick.currency);
      if (targetEl) targetEl.textContent = formatCurrency(sig.target_1 || sig.take_profit, tick.currency);
    }
  }

  function renderPaperPositions(p) {
    const body = document.getElementById('paperPositionsBody');
    const countEl = document.getElementById('paperOpenCount');
    if (!body) return;

    const positions = p.positions || [];
    if (countEl) countEl.textContent = `${positions.length} Active Positions`;

    if (positions.length === 0) {
      body.innerHTML = '<tr><td colspan="9" class="empty-state">No open positions. Use the order form to execute simulated trades.</td></tr>';
      return;
    }

    body.innerHTML = positions.map(pos => {
      const pnlClass = (pos.unrealized_pnl || 0) >= 0 ? 'text-success' : 'text-danger';
      return `
        <tr>
          <td><strong>${sanitize(pos.symbol || state.activeSymbol)}</strong></td>
          <td><span class="badge-tag">${sanitize(pos.side || 'BUY')}</span></td>
          <td>${pos.quantity?.toFixed(4) || '--'}</td>
          <td>${formatCurrency(pos.entry_price || pos.avg_entry_price)}</td>
          <td>${formatCurrency(pos.current_price || pos.mark_price)}</td>
          <td class="${pnlClass}">${formatCurrency(pos.unrealized_pnl)} (${formatChange(pos.return_pct || pos.unrealized_pnl_pct)})</td>
          <td>${formatCurrency(pos.stop_loss)}</td>
          <td>${formatCurrency(pos.target)}</td>
          <td><button class="btn-sm text-danger btn-close-pos" data-sym="${sanitize(pos.symbol)}">Close</button></td>
        </tr>
      `;
    }).join('');

    body.querySelectorAll('.btn-close-pos').forEach(btn => {
      btn.addEventListener('click', async () => {
        const sym = btn.getAttribute('data-sym');
        const pos = positions.find(x => x.symbol === sym);
        if (!pos) return;
        const tick = state.tickers[sym] || { price: pos.current_price || pos.mark_price };

        if (!state.paperApiKey) {
          const promptKey = prompt('Authentication required. Enter AGY_API_KEY to authorize closing position:');
          if (!promptKey || !promptKey.trim()) return;
          state.paperApiKey = promptKey.trim();
          const inputGroup = document.getElementById('paperAuthInputGroup');
          const activeGroup = document.getElementById('paperAuthActiveGroup');
          const badge = document.getElementById('paperAuthBadge');
          if (inputGroup) inputGroup.style.display = 'none';
          if (activeGroup) activeGroup.style.display = 'flex';
          if (badge) {
            badge.textContent = '🟢 Authorized';
            badge.style.background = 'rgba(14, 203, 129, 0.15)';
            badge.style.color = 'var(--color-green)';
          }
        }

        try {
          await API.post(
            '/api/v1/paper/order',
            {
              side: 'SELL',
              quantity: pos.quantity,
              price: tick.price,
              symbol: sym,
            },
            getPaperAuthHeaders()
          );
          loadPaperData();
        } catch (err) {
          alert(`Failed to close position: ${err.message}`);
        }
      });
    });
  }

  function initPaperTabs() {
    const tabs = document.getElementById('paperLedgerTabs');
    if (!tabs) return;
    tabs.querySelectorAll('.ledger-tab').forEach(btn => {
      btn.addEventListener('click', () => {
        tabs.querySelectorAll('.ledger-tab').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const view = btn.getAttribute('data-view');
        document.querySelectorAll('.ledger-view').forEach(v => v.classList.remove('active'));
        if (view === 'positions') {
          document.getElementById('viewPaperPositions')?.classList.add('active');
        } else {
          document.getElementById('viewPaperHistory')?.classList.add('active');
        }
      });
    });
  }

  function initPaperReset() {
    const handleReset = async () => {
      const msgEl = document.getElementById('paperOrderStatusMsg');
      if (!state.paperApiKey) {
        const promptKey = prompt('Authentication required. Enter AGY_API_KEY to authorize account reset:');
        if (!promptKey || !promptKey.trim()) {
          if (msgEl) msgEl.innerHTML = '<span class="text-amber">🔒 Reset cancelled: AGY_API_KEY is required.</span>';
          return;
        }
        state.paperApiKey = promptKey.trim();
        const inputGroup = document.getElementById('paperAuthInputGroup');
        const activeGroup = document.getElementById('paperAuthActiveGroup');
        const badge = document.getElementById('paperAuthBadge');
        if (inputGroup) inputGroup.style.display = 'none';
        if (activeGroup) activeGroup.style.display = 'flex';
        if (badge) {
          badge.textContent = '🟢 Authorized';
          badge.style.background = 'rgba(14, 203, 129, 0.15)';
          badge.style.color = 'var(--color-green)';
        }
      }

      if (confirm('Reset paper trading account? This will clear all positions and restore $10,000 initial cash.')) {
        try {
          await API.post('/api/v1/paper/reset', {}, getPaperAuthHeaders());
          if (msgEl) msgEl.innerHTML = '<span class="text-success">✓ Account successfully reset to initial $10,000 cash balance!</span>';
          loadPaperData();
        } catch (e) {
          if (msgEl) msgEl.innerHTML = `<span class="text-danger">Reset failed: ${sanitize(e.message)}</span>`;
          alert(`Reset failed: ${e.message}`);
        }
      }
    };

    const resetBtn1 = document.getElementById('btnResetPaperAccount');
    const resetBtn2 = document.getElementById('btnResetPaperLedger');
    if (resetBtn1) resetBtn1.addEventListener('click', handleReset);
    if (resetBtn2) resetBtn2.addEventListener('click', handleReset);
  }

  // ========================================================================
  // PAGE 7: BACKTEST LAB WORKSPACE
  // ========================================================================
  function initBacktestLab() {
    const form = document.getElementById('backtestForm');
    if (!form) return;
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const symbol = document.getElementById('btAsset').value;
      const strategy = document.getElementById('btStrategy').value;
      const timeframe = document.getElementById('btTimeframe').value;
      const capital = parseFloat(document.getElementById('btCapital').value) || 10000;
      const lookback = parseInt(document.getElementById('btLookback').value) || 160;

      const btn = document.getElementById('btnRunBacktest');
      btn.textContent = '⏳ Simulating...';
      btn.disabled = true;

      try {
        const res = await API.post('/api/v1/backtest/run', {
          symbol,
          strategy,
          timeframe,
          initial_capital: capital,
          lookback,
          fee_bps: 10.0,
          slippage_bps: 5.0,
        });
        if (res) {
          state.backtestData = res;
          renderBacktestResults(res);
        }
      } catch (err) {
        alert(`Backtest error: ${err.message}`);
      } finally {
        btn.textContent = '⚡ Run Backtest Simulation';
        btn.disabled = false;
      }
    });
  }

  async function loadBacktestData() {
    initBacktestLab();
    if (!state.backtestData) {
      const res = await API.get('/api/v1/backtest/results');
      if (res && res.ok && res.data) {
        state.backtestData = res.data;
        renderBacktestResults(res.data);
      }
    }
  }

  function renderBacktestResults(data) {
    const m = data.metrics || (data.ensemble_backtest ? {
      total_net_return_pct: data.ensemble_backtest.total_net_return_pct,
      buy_and_hold_return_pct: data.ensemble_backtest.buy_and_hold_return_pct,
      sharpe_ratio: data.ensemble_backtest.sharpe_ratio,
      sortino_ratio: data.ensemble_backtest.sortino_ratio,
      max_drawdown_pct: data.ensemble_backtest.max_drawdown_pct,
      win_rate_pct: data.ensemble_backtest.win_rate_pct,
      profit_factor: data.ensemble_backtest.profit_factor,
      total_trades: data.ensemble_backtest.total_trades,
    } : null);

    if (m) {
      const retEl = document.getElementById('btReturnVal');
      const bnhEl = document.getElementById('btBnhSub');
      const sharpeEl = document.getElementById('btSharpeVal');
      const sortinoEl = document.getElementById('btSortinoSub');
      const ddEl = document.getElementById('btMaxDdVal');
      const winEl = document.getElementById('btWinRateVal');
      const pfEl = document.getElementById('btProfitFactorSub');
      const tradesEl = document.getElementById('btTotalTradesVal');

      if (retEl) {
        retEl.textContent = formatChange(m.total_net_return_pct);
        retEl.className = 'bt-val ' + (m.total_net_return_pct >= 0 ? 'text-success' : 'text-danger');
      }
      if (bnhEl) bnhEl.textContent = `B&H: ${formatChange(m.buy_and_hold_return_pct)}`;
      if (sharpeEl) sharpeEl.textContent = (m.sharpe_ratio || 0).toFixed(2);
      if (sortinoEl) sortinoEl.textContent = `Sortino: ${(m.sortino_ratio || 0).toFixed(2)}`;
      if (ddEl) ddEl.textContent = `${(m.max_drawdown_pct || 0).toFixed(2)}%`;
      if (winEl) winEl.textContent = `${(m.win_rate_pct || 0).toFixed(1)}%`;
      if (pfEl) pfEl.textContent = `Profit Factor: ${(m.profit_factor || 1.0).toFixed(2)}`;
      if (tradesEl) tradesEl.textContent = m.total_trades || 0;
    }

    // Render Canvas Charts
    if (data.equity_curve) {
      drawCanvasCurve('btEquityCanvas', data.equity_curve.map(p => p.equity), data.equity_curve.map(p => p.benchmark), '#0ecb81', '#64748b');
    }
    if (data.drawdown_curve) {
      drawCanvasCurve('btDrawdownCanvas', data.drawdown_curve.map(p => p.drawdown), null, '#f6465d', null);
    }

    // Distribution
    const distEl = document.getElementById('btDistHistogram');
    if (distEl && data.trade_distribution) {
      distEl.innerHTML = data.trade_distribution.map(b => `
        <div class="dist-bar-row">
          <span class="dist-lbl">${b.range}</span>
          <div class="dist-bar-track">
            <div class="dist-bar-fill" style="width: ${Math.min(100, b.count * 15)}%;"></div>
          </div>
          <span class="dist-cnt">${b.count}</span>
        </div>
      `).join('');
    }

    // Comparison Table
    const compBody = document.getElementById('btComparisonBody');
    if (compBody && data.comparison_table) {
      compBody.innerHTML = data.comparison_table.map(row => `
        <tr>
          <td><strong>${sanitize(row.Model)}</strong></td>
          <td>${row['RMSE (1h)'] || '--'}</td>
          <td>${row['MAE (1h)'] || '--'}</td>
          <td>${row['Sharpe Ratio'] || row.Sharpe || '--'}</td>
          <td class="${(row['Net Return (%)'] || 0) >= 0 ? 'text-success' : 'text-danger'}">
            ${formatChange(row['Net Return (%)'])}
          </td>
        </tr>
      `).join('');
    }

    // Trades Table
    const tradesBody = document.getElementById('btTradesBody');
    if (tradesBody && data.trades) {
      tradesBody.innerHTML = data.trades.map(t => `
        <tr>
          <td>#${t.trade_id}</td>
          <td>${sanitize(t.entry_time?.slice(11, 16) || '--')}</td>
          <td>${sanitize(t.exit_time?.slice(11, 16) || '--')}</td>
          <td><span class="badge-tag">${sanitize(t.side)}</span></td>
          <td>${formatCurrency(t.entry_price)}</td>
          <td>${formatCurrency(t.exit_price)}</td>
          <td class="${t.pnl_pct >= 0 ? 'text-success' : 'text-danger'}">${formatChange(t.pnl_pct)}</td>
          <td class="${t.pnl_amount >= 0 ? 'text-success' : 'text-danger'}">${formatCurrency(t.pnl_amount)}</td>
          <td>${t.bars_held} bars</td>
        </tr>
      `).join('');
    }
  }

  function drawCanvasCurve(canvasId, values1, values2, color1, color2) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !values1 || values1.length === 0) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    const allVals = values2 ? [...values1, ...values2] : values1;
    const minVal = Math.min(...allVals);
    const maxVal = Math.max(...allVals);
    const range = (maxVal - minVal) || 1;

    function getY(v) {
      return h - 20 - ((v - minVal) / range) * (h - 40);
    }

    // Draw grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let y = 20; y < h; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Draw benchmark line if present
    if (values2 && color2) {
      ctx.strokeStyle = color2;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      values2.forEach((v, i) => {
        const x = (i / (values2.length - 1)) * w;
        const y = getY(v);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Draw strategy line
    ctx.strokeStyle = color1;
    ctx.lineWidth = 2;
    ctx.beginPath();
    values1.forEach((v, i) => {
      const x = (i / (values1.length - 1)) * w;
      const y = getY(v);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  // ========================================================================
  // PAGE 8: TRADING PRINCIPLES WORKSPACE
  // ========================================================================
  function loadPrinciplesData() {
    const quotes = [
      { text: "Don't focus on making money; focus on protecting what you have.", author: "— Paul Tudor Jones" },
      { text: "The market is not a random walk when volatility clusters. Manage the regime, not the price.", author: "— Benoit Mandelbrot" },
      { text: "It was never my thinking that made the big money for me. It was always my sitting.", author: "— Jesse Livermore" },
      { text: "The essence of investment management is the management of risks, not the management of returns.", author: "— Benjamin Graham" },
      { text: "If you don't bet, you can't win. If you lose all your chips, you can't bet.", author: "— Larry Hite" },
    ];
    let qIdx = Math.floor(Math.random() * quotes.length);
    const qText = document.getElementById('principleQuoteText');
    const qAuth = document.getElementById('principleQuoteAuthor');
    if (qText && qAuth) {
      qText.textContent = `"${quotes[qIdx].text}"`;
      qAuth.textContent = quotes[qIdx].author;
    }

    // Navigation smooth scroll inside reader
    document.querySelectorAll('#principlesNav .h-link').forEach(link => {
      link.addEventListener('click', (e) => {
        e.preventDefault();
        document.querySelectorAll('#principlesNav .h-link').forEach(l => l.classList.remove('active'));
        link.classList.add('active');
        const targetId = link.getAttribute('href').replace('#', '');
        const targetEl = document.getElementById(targetId);
        if (targetEl) {
          targetEl.scrollIntoView({ behavior: 'smooth' });
        }
      });
    });
  }

  // ========================================================================
  // WEBSOCKETS HUB - REAL-TIME TICK & CANDLE STREAMING
  // ========================================================================
  let notificationsWs = null;
  let lastWsMessageTime = 0;
  let lastIndicatorRecalcTime = 0;
  let staleCheckInterval = null;
  let currentSubscribedSymbol = null;
  let currentSubscribedTimeframe = null;

  function closeMarketWs(isIntentional = true) {
    if (wsReconnectTimer) {
      clearTimeout(wsReconnectTimer);
      wsReconnectTimer = null;
    }
    if (marketWs) {
      const oldWs = marketWs;
      marketWs = null;
      isWsConnected = false;
      currentSubscribedSymbol = null;
      currentSubscribedTimeframe = null;

      // Detach handlers before closing so intentional closes never trigger reconnect
      oldWs.onopen = null;
      oldWs.onmessage = null;
      oldWs.onerror = null;
      oldWs.onclose = null;

      try {
        oldWs.close(1000, isIntentional ? "Intentional close" : "Switching channel");
      } catch (e) {
        console.warn('[WS close warning]:', e);
      }
    }
  }

  function ensureWebSockets() {
    const wsDot = document.getElementById('wsStatusDot');
    const wsText = document.getElementById('wsStatusText');
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;

    const targetSymbol = state.activeSymbol;
    const targetTimeframe = state.activeTimeframe;

    // Ensure one active market socket per page/symbol/timeframe: if already active, do not duplicate
    if (
      marketWs &&
      (marketWs.readyState === WebSocket.OPEN || marketWs.readyState === WebSocket.CONNECTING) &&
      currentSubscribedSymbol === targetSymbol &&
      currentSubscribedTimeframe === targetTimeframe
    ) {
      return;
    }

    // Cleanly close existing market socket before opening new one
    closeMarketWs(false);

    try {
      const url = `${proto}//${host}/ws/market/${encodeURIComponent(targetSymbol)}?timeframe=${encodeURIComponent(targetTimeframe)}`;
      console.log(`[WS CONNECTING] Subscribing to ${targetSymbol} (${targetTimeframe})...`);
      
      currentSubscribedSymbol = targetSymbol;
      currentSubscribedTimeframe = targetTimeframe;
      const ws = new WebSocket(url);
      marketWs = ws;

      ws.onopen = () => {
        if (marketWs !== ws) return;
        isWsConnected = true;
        lastWsMessageTime = Date.now();
        console.log(`[WS CONNECTED] Subscribed: ${targetSymbol} (${targetTimeframe})`);
        if (wsDot) wsDot.className = 'status-dot green';
        if (wsText) wsText.textContent = 'LIVE WS';
      };

      ws.onmessage = (event) => {
        if (marketWs !== ws) return;
        try {
          lastWsMessageTime = Date.now();
          const msg = JSON.parse(event.data);
          if (msg.type === 'candle' && msg.data) {
            handleLiveCandle(msg.data);
          } else if (msg.type === 'tick' && msg.data) {
            handleLiveTick(msg.data);
          } else if (msg.type === 'ping') {
            try { ws.send(JSON.stringify({ type: 'pong' })); } catch (e) { console.warn('[WS pong warning]:', e); }
          }
        } catch (e) {
          console.warn('[WS message handle warning]:', e);
        }
      };

      ws.onclose = (event) => {
        if (marketWs !== ws) return;
        marketWs = null;
        isWsConnected = false;
        currentSubscribedSymbol = null;
        currentSubscribedTimeframe = null;

        const code = event ? event.code : 1005;
        const reason = event ? event.reason : '';
        console.warn(`[WS CLOSED] Code: ${code}, Reason: ${reason}`);

        // Do not reconnect after an intentional/normal close (code 1000)
        if (code === 1000) {
          if (wsDot) wsDot.className = 'status-dot gray';
          if (wsText) wsText.textContent = 'DISCONNECTED';
          return;
        }

        if (wsDot) wsDot.className = 'status-dot yellow';
        if (wsText) wsText.textContent = 'RECONNECTING...';

        // Exactly one reconnect timer per page/symbol/timeframe
        if (wsReconnectTimer) {
          clearTimeout(wsReconnectTimer);
        }
        wsReconnectTimer = setTimeout(() => {
          wsReconnectTimer = null;
          ensureWebSockets();
        }, 2500);
      };

      ws.onerror = (err) => {
        if (marketWs !== ws) return;
        console.warn('[WS ERROR] Market socket error:', err);
      };
    } catch (err) {
      if (wsDot) wsDot.className = 'status-dot yellow';
      if (wsText) wsText.textContent = 'POLLING';
    }

    // Stale Stream Watchdog (checks every 2.5s <= 3s)
    if (!staleCheckInterval) {
      staleCheckInterval = setInterval(() => {
        if (isWsConnected && lastWsMessageTime > 0 && Date.now() - lastWsMessageTime > 15000) {
          if (wsDot) wsDot.className = 'status-dot red';
          if (wsText) wsText.textContent = 'STALE';
          const statusEl = document.getElementById('terminalStatusBadge');
          if (statusEl) {
            statusEl.textContent = '⚠ Stale Feed • Reconnecting...';
            statusEl.className = 'market-status-pill status-warning';
          }
        }
      }, 2500);
    }

    // Connect Notifications Stream
    if (!notificationsWs || notificationsWs.readyState !== WebSocket.OPEN) {
      try {
        const notifUrl = `${proto}//${host}/ws/notifications`;
        notificationsWs = new WebSocket(notifUrl);
        notificationsWs.onmessage = (evt) => {
          try {
            const data = JSON.parse(evt.data);
            if (data.type === 'alert' && data.data) {
              showNotificationToast(data.data);
            }
          } catch (e) {
            console.warn('[Notification parse warning]:', e);
          }
        };
      } catch (e) {
        console.warn('[Notification connection warning]:', e);
      }
    }
  }

  function handleLiveCandle(candle) {
    if (!candle || candle.time === undefined) return;
    if (candle.symbol !== state.activeSymbol) return;

    const candleTime = Number(candle.time);
    const chartTime = toChartTime(candleTime);
    const open = Number(candle.open);
    const high = Number(candle.high);
    const low = Number(candle.low);
    const close = Number(candle.close);
    const volume = Number(candle.volume) || 0;

    if (!state.candles) state.candles = [];
    const last = state.candles[state.candles.length - 1];

    if (!last || candleTime > last.time) {
      // NEW BAR INTERVAL ROLLED OVER
      const newBar = {
        time: candleTime,
        chartTime: chartTime,
        open: open,
        high: high,
        low: low,
        close: close,
        volume: volume,
        timestamp: candle.timestamp || new Date(candleTime * 1000).toISOString(),
        timestamp_ms: candleTime * 1000,
      };
      state.candles.push(newBar);
      if (state.candles.length > 250) state.candles.shift();

      if (candleSeries && state.chartType === 'candles') {
        candleSeries.update({
          time: chartTime,
          open: open,
          high: high,
          low: low,
          close: close,
        });
      }
      if (lineSeries && state.chartType === 'line') {
        lineSeries.update({ time: chartTime, value: close });
      }
      if (volumeSeries) {
        volumeSeries.update({
          time: chartTime,
          value: volume,
          color: close >= open ? 'rgba(14, 203, 129, 0.4)' : 'rgba(246, 70, 93, 0.4)',
        });
      }
      recalculateIncrementalIndicators(close);
    } else if (candleTime === last.time) {
      // IN-FLIGHT BAR UPDATE (sub-second live ticks)
      last.high = Math.max(last.high, high);
      last.low = Math.min(last.low, low);
      last.close = close;
      last.volume = volume;
      last.chartTime = chartTime;

      const updatedBar = {
        time: chartTime,
        open: last.open,
        high: last.high,
        low: last.low,
        close: close,
      };

      if (candleSeries && state.chartType === 'candles') {
        candleSeries.update(updatedBar);
      }
      if (lineSeries && state.chartType === 'line') {
        lineSeries.update({ time: chartTime, value: close });
      }
      if (volumeSeries) {
        volumeSeries.update({
          time: chartTime,
          value: volume,
          color: close >= last.open ? 'rgba(14, 203, 129, 0.4)' : 'rgba(246, 70, 93, 0.4)',
        });
      }

      // Throttled indicator update (every 2.5s <= 3s)
      const now = Date.now();
      if (now - lastIndicatorRecalcTime > 2500) {
        lastIndicatorRecalcTime = now;
        recalculateIncrementalIndicators(close);
      }
    }

    // Update Live Terminal Header Telemetry
    const prev = state.candles.length > 1 ? state.candles[state.candles.length - 2] : last;
    const chgPct = prev && prev.close > 0 ? ((close - prev.close) / prev.close) * 100 : 0;
    updateTerminalHeader(close, chgPct, high, low, volume, candle.freshness_text, candle.data_status);
  }

  function handleLiveTick(tick) {
    if (!tick || !tick.symbol) return;
    state.tickers[tick.symbol] = tick;

    // Synchronize matching Watchlist row if rendered
    const symSafe = tick.symbol.replace(/[!"#$%&'()*+,.\/:;<=>?@[\\\]^`{|}~]/g, '\\$&');
    const wlRow = document.querySelector(`.wl-row[data-sym="${symSafe}"]`) || document.querySelector(`.wl-row[data-sym="${tick.symbol}"]`);
    if (wlRow) {
      const priceEl = wlRow.querySelector('.wl-price');
      const chgEl = wlRow.querySelector('.wl-chg');
      const item = state.instruments.find(i => i.symbol === tick.symbol);
      const curr = item ? item.currency : 'USD';
      if (priceEl) priceEl.textContent = formatCurrency(tick.price, curr);
      if (chgEl) {
        chgEl.textContent = formatChange(tick.change_24h_pct);
        chgEl.className = 'wl-chg ' + ((tick.change_24h_pct || 0) >= 0 ? 'text-success' : 'text-danger');
      }
    }

    if (tick.symbol === state.activeSymbol) {
      updateHeaderPill();
      const lastCandle = state.candles && state.candles.length ? state.candles[state.candles.length - 1] : null;
      let liveHigh = tick.high_24h || tick.price;
      let liveLow = tick.low_24h || tick.price;
      if (lastCandle && lastCandle.time) {
        liveHigh = Math.max(lastCandle.high, tick.price);
        liveLow = Math.min(lastCandle.low, tick.price);
        lastCandle.close = tick.price;
        lastCandle.high = liveHigh;
        lastCandle.low = liveLow;

        const chartTime = lastCandle.chartTime || toChartTime(lastCandle.time);

        if (candleSeries && state.chartType === 'candles') {
          candleSeries.update({
            time: chartTime,
            open: lastCandle.open,
            high: liveHigh,
            low: liveLow,
            close: tick.price,
          });
        } else if (lineSeries && state.chartType === 'line') {
          lineSeries.update({
            time: chartTime,
            value: tick.price,
          });
        }
      }
      updateTerminalHeader(tick.price, tick.change_24h_pct, liveHigh, liveLow, tick.volume_24h, tick.freshness_text, tick.data_status);
    }
  }

  function recalculateIncrementalIndicators(liveClose) {
    if (!state.indicators || !state.indicators.current) return;
    state.indicators.current.price = liveClose;

    // Incremental EMA 20 & 50 smoothing
    if (state.indicators.current.ema20) {
      const alpha20 = 2 / (20 + 1);
      state.indicators.current.ema20 = (liveClose * alpha20) + (state.indicators.current.ema20 * (1 - alpha20));
    }
    if (state.indicators.current.ema50) {
      const alpha50 = 2 / (50 + 1);
      state.indicators.current.ema50 = (liveClose * alpha50) + (state.indicators.current.ema50 * (1 - alpha50));
    }

    // Dynamic RSI update
    if (state.candles && state.candles.length >= 15) {
      const recent = state.candles.slice(-15);
      let gains = 0, losses = 0;
      for (let i = 1; i < recent.length; i++) {
        const diff = recent[i].close - recent[i - 1].close;
        if (diff >= 0) gains += diff;
        else losses += Math.abs(diff);
      }
      const avgGain = gains / 14;
      const avgLoss = (losses / 14) || 1e-6;
      const rs = avgGain / avgLoss;
      state.indicators.current.rsi = Math.round((100 - (100 / (1 + rs))) * 10) / 10;
    }

    // Update Subchart display text
    const valEl = document.getElementById('terminalSubChartVal');
    if (valEl) {
      if (state.subChartType === 'rsi' && state.indicators.current.rsi !== undefined) {
        const rsi = state.indicators.current.rsi;
        valEl.textContent = `RSI: ${rsi.toFixed(1)} • ${rsi > 70 ? 'Overbought' : (rsi < 30 ? 'Oversold' : 'Neutral')}`;
      } else if (state.subChartType === 'macd' && state.indicators.current.macd_hist !== undefined) {
        valEl.textContent = `MACD Hist: ${state.indicators.current.macd_hist.toFixed(2)}`;
      } else if (state.subChartType === 'stoch' && state.indicators.current.stoch_k !== undefined) {
        valEl.textContent = `Stoch %K: ${state.indicators.current.stoch_k.toFixed(1)}`;
      }
    }
  }

  function showNotificationToast(alert) {
    let container = document.getElementById('toastContainer');
    if (!container) {
      container = document.createElement('div');
      container.id = 'toastContainer';
      container.className = 'toast-container';
      document.body.appendChild(container);
    }
    const toast = document.createElement('div');
    toast.className = `toast-alert ${alert.severity || 'HIGH'}`;
    const icon = alert.signal?.includes('BUY') ? '🟢' : (alert.signal?.includes('SELL') ? '🔴' : '⚠️');
    toast.innerHTML = `
      <div style="font-weight:600; margin-bottom:4px;">${icon} ${sanitize(alert.signal)} • ${sanitize(alert.symbol)} (${alert.confidence_pct || 75}%)</div>
      <div style="color:var(--text-muted); font-size:11px;">${sanitize(alert.explanation || '')}</div>
      <div style="color:var(--text-dim); font-size:10px; margin-top:4px;">${sanitize(alert.timestamp_ist || '')}</div>
    `;
    container.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      setTimeout(() => {
        try {
          if (container && toast.parentElement === container) {
            container.removeChild(toast);
          }
        } catch (e) {
          console.warn('[Toast removal warning]:', e);
        }
      }, 300);
    }, 6000);
  }

  async function hydrateAuthoritativeStore() {
    try {
      // 1. Authoritative spot ticker hydration for active symbol (BTC/USDT by default)
      const tickRes = await API.get(`/api/v1/market/${encodeURIComponent(state.activeSymbol)}/ticker`);
      if (tickRes && tickRes.ok && tickRes.data) {
        handleLiveTick(tickRes.data);
      }

      // 2. Authoritative explainable ML signal hydration for active symbol
      const sigRes = await API.get(`/api/v1/signal/${encodeURIComponent(state.activeSymbol)}?timeframe=${state.activeTimeframe}`);
      if (sigRes && sigRes.ok && sigRes.data) {
        state.signal = sigRes.data;
        renderTerminalSignal(sigRes.data);
      }
    } catch (err) {
      console.warn('[Store Hydration Warning]:', err);
    }
  }

  // ========================================================================
  // GLOBAL APPLICATION INITIALIZATION
  // ========================================================================
  function initApp() {
    initHeader();
    initGlobalSearch();
    initRouting();
    initOverviewCategoryFilters();
    initTerminal();
    initAiAssistantPage();
    initPaperTrading();

    // Hydrate price store and ML signal immediately on startup regardless of route
    hydrateAuthoritativeStore();

    // Start background live sync (<= 3s live refresh contract; poll only when !isWsConnected or WS stale)
    setInterval(async () => {
      if (!isWsConnected || (lastWsMessageTime > 0 && Date.now() - lastWsMessageTime > 5000)) {
        // 1. Live spot ticker sync fallback
        const activeTickRes = await API.get(`/api/v1/market/${encodeURIComponent(state.activeSymbol)}/ticker`);
        if (activeTickRes && activeTickRes.ok && activeTickRes.data) {
          handleLiveTick(activeTickRes.data);
        }

        // 2. Active Chart Fallback Polling (<= 3s contract):
        if (state.currentPage === 'terminal') {
          await loadTerminalCandles(false);
        }
      }
    }, 2500);
  }

  // Expose global window API for error-state retry buttons and debugging
  window.AlphaVol = {
    loadOverviewData,
    loadTerminalCandles,
    loadTerminalSignal,
    loadTerminalIndicators,
    loadResearchData,
    loadNewsEventsPage,
    loadPaperData,
    hydrateAuthoritativeStore,
    retryOverview: () => loadOverviewData(),
    retrySignal: () => loadTerminalSignal(),
    retryCandles: () => loadTerminalCandles(true),
  };

  // Run on DOM Ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
  } else {
    initApp();
  }

})();
