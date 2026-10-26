/* Coin — front-end. No framework, no build step, no dependencies. */
(() => {
  const $ = (id) => document.getElementById(id);
  const form = $('form'), qEl = $('q'), ctxEl = $('ctx');
  const result = $('result'), probEl = $('prob'), fillEl = $('fill');
  const verdictEl = $('verdict'), noteEl = $('verdict-note'), qechoEl = $('qecho');
  const sentenceEl = $('sentence'), evidenceEl = $('evidence'), stabilityEl = $('stability');
  const readingsEl = $('readings'), spreadEl = $('spread'), discloseEl = $('disclose');
  const submitEl = $('submit'), wallList = $('wall-list'), wallEmpty = $('wall-empty');
  const ctxToggle = $('ctx-toggle'), ctxPanel = $('ctx-panel');

  let last = null;
  let wallSort = 'closest';

  /* ---------- helpers ---------- */
  const pct = (p) => Math.round(p * 100);
  const clamp = (n, a, b) => Math.max(a, Math.min(b, n));

  function verdict(p, spread, evidence) {
    const d = Math.abs(p - 0.5);
    if (evidence < 0.15) return { label: 'guessing', kind: 'guess' };
    if (spread >= 0.18) return { label: 'unstable', kind: 'unstable' };
    if (d <= 0.06) return { label: 'coin flip', kind: 'flip' };
    const side = p > 0.5 ? 'yes' : 'no';
    if (d <= 0.12) return { label: `barely ${side}`, kind: 'lean' };
    if (d <= 0.25) return { label: `leans ${side}`, kind: 'lean' };
    return { label: side, kind: 'clear' };
  }

  function sentence(p, spread, evidence) {
    const d = Math.abs(p - 0.5);
    const side = p > 0.5 ? 'yes' : 'no';
    if (evidence < 0.15) {
      return `There is nothing checkable in what you gave me, so ${pct(p)}% is a prior, not a prediction. Add the situation and the number starts to mean something.`;
    }
    if (spread >= 0.18) {
      return `The three readings landed ${Math.round(spread * 100)} points apart. The phrasing is doing more work than the facts, so treat this as unstable.`;
    }
    if (d <= 0.06) {
      return 'This is a coin flip. A number this close to 50 is decoration, not a forecast, and I will not pretend otherwise.';
    }
    if (evidence < 0.4) {
      return `It leans ${side}, but the text carries only a hint of evidence. That is a prior with a nice font, not a judgment.`;
    }
    if (d <= 0.12) {
      return `It leans ${side} and holds up across phrasings, but the margin is thin enough that a single new fact flips it.`;
    }
    return `It leans ${side} and the readings held steady. The number is only as good as the evidence in the text, which is ${evidenceLabel(evidence).toLowerCase()}.`;
  }

  function evidenceLabel(e) {
    if (e < 0.17) return 'None';
    if (e < 0.5) return 'A hint';
    if (e < 0.84) return 'Some';
    return 'Strong';
  }

  function stabilityLabel(spread) {
    if (spread <= 0.06) return 'tight';
    if (spread <= 0.18) return 'loose';
    return 'unstable';
  }

  /* ---------- render ---------- */
  function render(d) {
    last = d;
    const p = d.p, spread = d.spread || 0, evidence = d.evidence ?? 0.5;
    const v = verdict(p, spread, evidence);

    result.hidden = false;
    result.className = 'result';
    verdictEl.textContent = v.label;
    noteEl.textContent =
      v.kind === 'guess'
        ? `evidence ${evidence.toFixed(2)} · a prior, not a read`
        : `p(yes) = ${pct(p)}/100`;
    probEl.innerHTML = `${pct(p)}<span class="pct">%</span>`;
    probEl.className = 'prob' + (v.kind === 'flip' ? ' mid' : v.kind === 'guess' ? ' guess' : '');
    qechoEl.textContent = d.question;
    fillEl.style.width = clamp(p * 100, 2, 100) + '%';
    fillEl.className = 'fill' + (v.kind === 'guess' ? ' guess' : '');
    sentenceEl.textContent = sentence(p, spread, evidence);
    evidenceEl.textContent = evidenceLabel(evidence);
    stabilityEl.textContent = stabilityLabel(spread);
    spreadEl.textContent = spread.toFixed(2);
    readingsEl.textContent = `${d.readings || 3} phrasings`;
    discloseEl.textContent =
      `${d.model || 'pplx-decider-flash-v1.2'} · ${d.params || '4B'} · temperature ${Number(d.temperature || 1.5539).toFixed(3)} · ${d.device || 'mps'} · zero output tokens`;
    $('share').disabled = false;
  }

  function failed(msg) {
    result.hidden = false;
    result.className = 'result error';
    verdictEl.textContent = 'no read';
    noteEl.textContent = '';
    probEl.innerHTML = '&mdash;';
    probEl.className = 'prob';
    qechoEl.textContent = '';
    fillEl.style.width = '0%';
    sentenceEl.textContent = msg;
    evidenceEl.textContent = '-';
    stabilityEl.textContent = '-';
    readingsEl.textContent = '-';
    spreadEl.textContent = '-';
    discloseEl.textContent = '';
  }

  function loading() {
    result.hidden = false;
    result.className = 'result loading';
    verdictEl.textContent = 'reading';
    noteEl.textContent = 'three phrasings, one forward pass each';
    probEl.innerHTML = '&hellip;';
    probEl.className = 'prob';
    qechoEl.textContent = qEl.value.trim();
    sentenceEl.textContent = 'Reading the same question three different ways to see if the number holds.';
    evidenceEl.textContent = '-';
    stabilityEl.textContent = '-';
    readingsEl.textContent = '-';
    spreadEl.textContent = '-';
    discloseEl.textContent = '';
    $('share').disabled = true;
  }

  /* ---------- decide ---------- */
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const question = qEl.value.trim();
    if (question.length < 3) return;
    const context = ctxEl.value.trim();

    submitEl.disabled = true;
    submitEl.querySelector('.lbl').textContent = 'Reading';
    loading();
    try {
      const res = await fetch('/api/decide', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ question, context })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'the model did not answer');
      render(data);
      loadWall();
    } catch (err) {
      failed(`${err.message || 'Something broke'}. If the model is still loading, wait a few seconds and flip again.`);
    } finally {
      submitEl.disabled = false;
      submitEl.querySelector('.lbl').textContent = 'Flip';
    }
  });

  ctxToggle.addEventListener('click', () => {
    const open = ctxPanel.hidden;
    ctxPanel.hidden = !open;
    ctxToggle.setAttribute('aria-expanded', String(open));
    ctxToggle.textContent = open ? 'Hide the situation' : 'Add the situation';
    if (open) ctxEl.focus();
  });

  $('again').addEventListener('click', () => {
    qEl.value = '';
    qEl.focus();
    result.hidden = true;
  });

  /* ---------- wall ---------- */
  async function loadWall() {
    try {
      const res = await fetch(`/api/wall?sort=${wallSort}&limit=12`);
      const rows = await res.json();
      wallList.innerHTML = '';
      wallEmpty.hidden = rows.length > 0;
      for (const r of rows) {
        const li = document.createElement('li');
        const q = document.createElement('span');
        q.className = 'wall-q';
        q.textContent = r.question;
        q.title = r.question;
        const bar = document.createElement('span');
        bar.className = 'wall-bar';
        const fill = document.createElement('i');
        fill.style.width = clamp(r.p * 100, 2, 100) + '%';
        bar.appendChild(fill);
        const p = document.createElement('span');
        p.className = 'wall-p';
        p.textContent = pct(r.p) + '%';
        li.append(q, bar, p);
        wallList.appendChild(li);
      }
    } catch { /* the wall is decoration; the read is the product */ }
  }

  document.querySelectorAll('.tab').forEach((t) => {
    t.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach((x) => {
        x.classList.remove('on');
        x.setAttribute('aria-selected', 'false');
      });
      t.classList.add('on');
      t.setAttribute('aria-selected', 'true');
      wallSort = t.dataset.sort;
      loadWall();
    });
  });

  /* ---------- share card ---------- */
  function wrap(ctx, text, maxWidth) {
    const words = text.split(/\s+/);
    const lines = [];
    let line = '';
    for (const w of words) {
      const test = line ? line + ' ' + w : w;
      if (ctx.measureText(test).width > maxWidth && line) {
        lines.push(line);
        line = w;
      } else {
        line = test;
      }
    }
    if (line) lines.push(line);
    return lines;
  }

  $('share').addEventListener('click', () => {
    if (!last) return;
    const W = 1080, H = 1350;
    const c = document.createElement('canvas');
    c.width = W; c.height = H;
    const x = c.getContext('2d');
    const SANS = '-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Helvetica,Arial,sans-serif';
    const MONO = 'ui-monospace,SFMono-Regular,Menlo,monospace';

    x.fillStyle = '#0C0C0D';
    x.fillRect(0, 0, W, H);

    x.fillStyle = '#EFECE4';
    x.font = `600 22px ${MONO}`;
    x.letterSpacing = '10px';
    x.fillText('COIN', 78, 108);
    x.letterSpacing = '0px';

    x.fillStyle = '#65645F';
    x.font = `400 20px ${MONO}`;
    x.textAlign = 'right';
    x.fillText(new Date().toISOString().slice(0, 10), W - 78, 108);
    x.textAlign = 'left';

    x.fillStyle = '#26262B';
    x.fillRect(78, 148, W - 156, 1);

    x.fillStyle = '#EFECE4';
    x.font = `500 46px ${SANS}`;
    const lines = wrap(x, last.question, W - 156).slice(0, 4);
    let y = 250;
    for (const ln of lines) {
      x.fillText(ln, 78, y);
      y += 60;
    }

    const v = verdict(last.p, last.spread || 0, last.evidence ?? 0.5);
    y += 34;
    x.fillStyle = v.kind === 'guess' ? '#8A8880' : '#E4552B';
    x.font = `600 20px ${MONO}`;
    x.letterSpacing = '6px';
    x.fillText(v.label.toUpperCase(), 78, y);
    x.letterSpacing = '0px';

    y += 40;
    x.fillStyle = v.kind === 'flip' ? '#E4552B' : v.kind === 'guess' ? '#A3A199' : '#EFECE4';
    x.font = `600 240px ${MONO}`;
    x.letterSpacing = '-14px';
    const num = String(pct(last.p));
    x.fillText(num, 72, y + 200);
    x.letterSpacing = '0px';
    x.fillStyle = '#93918A';
    x.font = `500 64px ${MONO}`;
    x.fillText('%', 72 + x.measureText(num).width + 8, y + 200);

    const ty = y + 280;
    x.fillStyle = '#1B1B1E';
    x.beginPath();
    x.roundRect(78, ty, W - 156, 22, 11);
    x.fill();
    x.fillStyle = v.kind === 'guess' ? '#3C3C42' : '#E4552B';
    x.beginPath();
    x.roundRect(78, ty, Math.max(24, (W - 156) * last.p), 22, 11);
    x.fill();
    x.fillStyle = '#3C3C42';
    x.fillRect(78 + (W - 156) / 2, ty - 10, 2, 42);

    x.fillStyle = '#65645F';
    x.font = `400 20px ${MONO}`;
    x.fillText('0', 78, ty + 62);
    x.textAlign = 'right';
    x.fillText('100', W - 78, ty + 62);
    x.textAlign = 'left';

    x.fillStyle = '#93918A';
    x.font = `400 26px ${SANS}`;
    const sl = wrap(x, sentence(last.p, last.spread || 0, last.evidence ?? 0.5), W - 156).slice(0, 3);
    let sy = ty + 130;
    for (const ln of sl) {
      x.fillText(ln, 78, sy);
      sy += 38;
    }

    x.fillStyle = '#26262B';
    x.fillRect(78, H - 190, W - 156, 1);

    x.fillStyle = '#93918A';
    x.font = `500 24px ${SANS}`;
    x.fillText('Built By Harish Kotra', 78, H - 130);
    x.fillText('Checkout my other builds', 78, H - 92);
    x.fillStyle = '#EFECE4';
    x.textAlign = 'right';
    x.fillText('harishkotra.me', W - 78, H - 130);
    x.fillText('dailybuild.xyz', W - 78, H - 92);
    x.textAlign = 'left';

    x.fillStyle = '#65645F';
    x.font = `400 18px ${MONO}`;
    x.fillText(`${last.model || 'pplx-decider-flash-v1.2'} · 4B decision model · on a Mac Mini`, 78, H - 44);

    c.toBlob((blob) => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `coin-${pct(last.p)}.png`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 4000);
    }, 'image/png');
  });

  /* ---------- boot ---------- */
  fetch('/api/health')
    .then((r) => r.json())
    .then((h) => {
      const meta = $('meta');
      if (h.model_loaded) {
        meta.className = 'meta live';
        meta.innerHTML = '<span class="dot"></span>' + (h.device || 'mps').toUpperCase() + ' · 4B ready';
      } else {
        meta.textContent = 'model loading';
      }
    })
    .catch(() => {});

  const deep = new URLSearchParams(location.search).get('q');
  if (deep) {
    qEl.value = deep.slice(0, 220);
    form.requestSubmit();
  }
  loadWall();
})();
