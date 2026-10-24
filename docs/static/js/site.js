(function () {
  'use strict';

  var D = window.ONQ;
  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var SVGNS = 'http://www.w3.org/2000/svg';
  // Series colors are CSS variables so they follow the light/dark theme.
  var COLORS = {
    'Full precision': 'var(--c-fp)',
    'EfficientQAT': 'var(--c-eqat)',
    'BitDistiller': 'var(--c-bd)',
    'ReasoningQAT': 'var(--c-rqat)',
    'OnQ': 'var(--c-onq)'
  };
  var tooltip = document.getElementById('tooltip');

  function el(tag, attrs, parent) {
    var node = document.createElement(tag);
    for (var k in attrs || {}) {
      if (k === 'text') node.textContent = attrs[k];
      else if (k === 'html') node.innerHTML = attrs[k];
      else node.setAttribute(k, attrs[k]);
    }
    if (parent) parent.appendChild(node);
    return node;
  }
  function sv(tag, attrs, parent) {
    var node = document.createElementNS(SVGNS, tag);
    for (var k in attrs || {}) {
      if (k === 'text') node.textContent = attrs[k];
      else node.setAttribute(k, attrs[k]);
    }
    if (parent) parent.appendChild(node);
    return node;
  }
  function fmt(n) { return Math.round(n).toLocaleString('en-US'); }
  function showTip(evt, html) {
    tooltip.innerHTML = html;
    tooltip.style.opacity = 1;
    tooltip.style.left = (evt.clientX + 12) + 'px';
    tooltip.style.top = (evt.clientY - 10) + 'px';
  }
  function hideTip() { tooltip.style.opacity = 0; }
  function whenVisible(node, fn) {
    if (!('IntersectionObserver' in window)) { fn(); return; }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { if (e.isIntersecting) { io.disconnect(); fn(); } });
    }, { threshold: 0.25 });
    io.observe(node);
  }

  /* ===================== Figure 1: one question, five models ===================== */
  // Token positions measured with the Qwen3 tokenizer on the seed-42 MATH-500 outputs.
  var BUDGET = 32768;
  var LANES = [
    { name: 'Full precision', sub: 'BF16', color: 'Full precision', len: 1900, ans: 402, loop: null },
    { name: 'EfficientQAT', sub: '2-bit', color: 'EfficientQAT', len: 32768, ans: 330, loop: 1596 },
    { name: 'BitDistiller', sub: '2-bit', color: 'BitDistiller', len: 32768, ans: null, loop: 4427 },
    { name: 'ReasoningQAT', sub: '2-bit', color: 'ReasoningQAT', len: 32768, ans: 358, loop: 963 },
    { name: 'OnQ', sub: '2-bit · ours', color: 'OnQ', len: 2612, ans: 740, loop: null, cls: 'onq' }
  ];

  function buildLanes() {
    var host = document.getElementById('lanes');
    LANES.forEach(function (t) {
      var row = el('div', { 'class': 'lane ' + (t.cls || '') }, host);
      el('div', { 'class': 'lane-name', html: t.name + '<small>' + t.sub + '</small>' }, row);
      var track = el('div', { 'class': 'lane-track' }, row);
      t.fill = el('div', { 'class': 'lane-fill' }, track);
      t.fill.style.background = COLORS[t.color];
      if (t.loop !== null) {
        t.loopEl = el('div', { 'class': 'lane-loop' }, track);
        t.loopEl.style.left = (100 * t.loop / BUDGET) + '%';
      }
      if (t.ans !== null) {
        t.tick = el('div', { 'class': 'lane-ans' }, track);
        t.tick.style.left = (100 * t.ans / BUDGET) + '%';
      }
      var out = el('div', { 'class': 'lane-out' }, row);
      t.count = el('span', { 'class': 'lane-count', text: '0' }, out);
      t.chip = el('span', { 'class': 'chip', text: '…' }, out);
    });
    var axis = document.getElementById('scale-axis');
    [0, 8192, 16384, 24576, 32768].forEach(function (v) {
      var s = el('span', { text: v === 0 ? '0' : (v / 1024) + 'K' }, axis);
      s.style.left = (100 * v / BUDGET) + '%';
    });
  }

  function renderLanes(tokens) {
    LANES.forEach(function (t) {
      var cur = Math.min(tokens, t.len);
      t.fill.style.width = (100 * cur / BUDGET) + '%';
      if (t.loopEl) t.loopEl.style.width = (100 * Math.max(0, cur - t.loop) / BUDGET) + '%';
      if (t.tick) t.tick.classList.toggle('on', cur >= t.ans);
      t.count.textContent = fmt(cur);
      if (tokens >= t.len) {
        var done = t.len < BUDGET;
        t.chip.textContent = done ? 'DONE · 9' : 'LIMIT';
        t.chip.className = 'chip ' + (done ? 'done' : 'limit');
      } else {
        t.chip.textContent = '…';
        t.chip.className = 'chip';
      }
    });
  }

  var raceRun = 0;
  function playRace() {
    var run = ++raceRun;
    if (reduceMotion) { renderLanes(BUDGET); return; }
    var DURATION = 7000, HOLD = 4000, start = null;
    function frame(ts) {
      if (run !== raceRun) return;
      if (start === null) start = ts;
      var t = ts - start;
      renderLanes(BUDGET * Math.min(1, t / DURATION));
      if (t < DURATION + HOLD) requestAnimationFrame(frame);
      else { start = null; requestAnimationFrame(frame); }
    }
    requestAnimationFrame(frame);
  }

  buildLanes();
  renderLanes(0);
  whenVisible(document.getElementById('race'), playRace);
  document.getElementById('replay').addEventListener('click', playRace);

  /* ===================== Figure 2: accuracy / truncation by length ===================== */
  (function () {
    var f = D.fig1;
    var order = ['Full precision', 'EfficientQAT', 'BitDistiller', 'ReasoningQAT', 'OnQ'];
    var binLabels = ['< 2K', '2–4K', '4–8K', '8–16K', '≥ 16K'];
    var W = 760, H = 320, m = { l: 56, r: 18, t: 14, b: 58 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;
    var svg = sv('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Accuracy or truncation rate by full-precision response length' });
    document.getElementById('fig2-svg').appendChild(svg);
    var x = function (i) { return m.l + (i + 0.5) * iw / binLabels.length; };
    var y = function (v) { return m.t + ih * (1 - v / 100); };

    var grid = sv('g', { 'class': 'grid' }, svg);
    [0, 25, 50, 75, 100].forEach(function (v) {
      sv('line', { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }, grid);
      sv('text', { x: m.l - 8, y: y(v) + 4, 'text-anchor': 'end', text: v }, svg);
    });
    binLabels.forEach(function (b, i) {
      sv('text', { x: x(i), y: H - m.b + 20, 'text-anchor': 'middle', text: b }, svg);
    });
    sv('line', { 'class': 'baseline', x1: m.l, x2: W - m.r, y1: y(0), y2: y(0) }, svg);
    sv('text', { x: m.l + iw / 2, y: H - 10, 'text-anchor': 'middle', text: 'Full-precision response length (tokens)' }, svg);
    var yTitle = sv('text', { transform: 'translate(14,' + (m.t + ih / 2) + ') rotate(-90)', 'text-anchor': 'middle' }, svg);

    var state = { metric: 'acc', focus: null };
    var series = {};
    order.forEach(function (name) {
      var g = sv('g', { 'class': 'series' }, svg);
      var path = sv('path', {}, g);
      path.style.stroke = COLORS[name];
      path.style.strokeWidth = name === 'OnQ' ? 3 : 2.2;
      if (name === 'Full precision') path.style.strokeDasharray = '5 4';
      path.style.transition = reduceMotion ? 'none' : 'd 0.5s ease, opacity 0.2s';
      var dots = binLabels.map(function (_, i) {
        var c = sv('circle', { cx: x(i), r: name === 'OnQ' ? 5 : 4 }, g);
        c.style.fill = COLORS[name];
        c.style.transition = reduceMotion ? 'none' : 'cy 0.5s ease';
        c.addEventListener('mousemove', function (evt) {
          var v = f[state.metric][name][i];
          showTip(evt, '<b>' + name + '</b><br>' + binLabels[i] + ' (n = ' + f.n[i] + ')<br>' +
            (state.metric === 'acc' ? 'Accuracy: ' : 'Truncated: ') + v.toFixed(1) + '%');
        });
        c.addEventListener('mouseleave', hideTip);
        return c;
      });
      series[name] = { g: g, path: path, dots: dots };
    });

    function draw() {
      var data = f[state.metric];
      yTitle.textContent = state.metric === 'acc' ? 'Accuracy (%)' : 'Truncated responses (%)';
      order.forEach(function (name) {
        var s = series[name];
        var pts = data[name].map(function (v, i) { return [x(i), y(v)]; });
        s.path.setAttribute('d', 'M' + pts.map(function (p) { return p[0].toFixed(1) + ',' + p[1].toFixed(1); }).join('L'));
        s.dots.forEach(function (c, i) { c.setAttribute('cy', pts[i][1]); });
        s.g.classList.toggle('dim', state.focus !== null && state.focus !== name);
      });
    }

    var legend = document.getElementById('fig2-legend');
    order.forEach(function (name) {
      var item = el('span', { html: '<i style="background:' + COLORS[name] + '"></i>' + (name === 'OnQ' ? '<span class="me">OnQ</span>' : name) }, legend);
      item.addEventListener('click', function () {
        state.focus = state.focus === name ? null : name;
        draw();
      });
    });
    document.querySelectorAll('#fig2 [data-metric]').forEach(function (b) {
      b.addEventListener('click', function () {
        state.metric = b.getAttribute('data-metric');
        document.querySelectorAll('#fig2 [data-metric]').forEach(function (o) { o.setAttribute('aria-pressed', String(o === b)); });
        draw();
      });
    });
    draw();
  })();

  /* ===================== Figure 3: trace excerpts ===================== */
  (function () {
    var CUT = ['cut', ' ✂ \n'];
    var panels = [
      {
        title: 'ReasoningQAT · 2-bit', meta: '9 at token 358 · loop from 963',
        segs: [
          ['', 'Next, 42 divided by 25. 25 times 1 is 25, 25 times 2 is 50, which is more than 42. So 1. So that\'s 1. So total so far is 8 + 1 = '],
          ['ans', '9'], ['', '.\n'], CUT,
          ['', 'So the answer is '], ['ans', '9'],
          ['', ' zeros at the end of 42!.\n\nBut wait, let me check with another way. Let me list out the multiples of 5 in 1-42:\n\n'],
          ['loop', 'Multiples of 5: 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, ']
        ],
        more: function (k) { return (65 + 5 * k) + ', '; }, loops: 24,
        tail: [CUT, ['loop', '24310, 24315, 24320, 24325, 2']],
        foot: ['32,768 tokens', 'LIMIT', 'limit']
      },
      {
        title: 'BitDistiller · 2-bit', meta: 'never states 9 · loop from 4,427',
        segs: [
          ['', 'Wait, the formula is that the number of trailing zeros in n! is equal to the number of times 5 divides into n!.'],
          CUT,
          ['', 'But for n=10, the number of 5s is 1, so the number of trailing zeros is 1. But according to the formula, floor(10/5) + floor(10/25) = 2. Therefore, there\'s a problem here.\n'],
          CUT
        ],
        more: function () { return 'but in reality, the number of 5s is floor(n/5) + floor(n/25) + ... '; }, loops: 4,
        tail: [CUT, ['loop', 'but in reality, the number of 5s is floor(n/5) +']],
        foot: ['32,768 tokens', 'LIMIT', 'limit']
      },
      {
        title: 'OnQ · 2-bit', meta: '9 at token 740 · stops',
        segs: [
          ['', 'For 5: 42 divided by 5 is 8.4, so 8.4, so floor is 8. Then 42/25 is 1.68, floor is 1. So total 8+1='],
          ['ans', '9'],
          ['', '. Correct.\n\nFor 2: 42 divided by 2 is 21, then 42/4 is 10.5, floor 10, 42/8 is 5.25, floor 5, 42/16 is 2.625, floor 2, 42/32 is 1.3125, floor 1, then 42/64 is 0. So 21+10+5+2+1=29. Correct. So yes, 29.\n'],
          CUT,
          ['', 'So, the number of 2s is 29, the number of 5s is 9, so 9 zeros.\n'],
          CUT,
          ['', '**Final Answer**\nThe number of zeros at the end of $42!$ is \\boxed{'], ['ans', '9'], ['', '}.']
        ],
        more: null, loops: 0, tail: [],
        foot: ['2,612 tokens', 'DONE · 9', 'done']
      }
    ];

    var host = document.getElementById('streams');
    panels.forEach(function (p) {
      var box = el('div', { 'class': 'trace' }, host);
      var head = el('div', { 'class': 'trace-head' }, box);
      el('b', { text: p.title }, head);
      el('span', { text: p.meta }, head);
      p.outer = el('div', { 'class': 'trace-body tex2jax_ignore' }, box);
      p.body = el('div', {}, p.outer);
      var foot = el('div', { 'class': 'trace-foot' }, box);
      el('span', { text: p.foot[0] }, foot);
      p.stateEl = el('span', { 'class': 'chip', text: '…' }, foot);
    });

    function appendSeg(body, cls, text) {
      var last = body.lastChild;
      if (last && last.getAttribute('data-cls') === cls) { last.textContent += text; return; }
      var span = el('span', { 'data-cls': cls }, body);
      if (cls) span.className = cls;
      span.textContent = text;
    }
    function finish(p) {
      p.tail.forEach(function (s) { appendSeg(p.body, s[0], s[1]); });
      p.stateEl.textContent = p.foot[1];
      p.stateEl.className = 'chip ' + p.foot[2];
      p.outer.scrollTop = p.outer.scrollHeight;
    }
    function renderAll(p) {
      p.body.innerHTML = '';
      p.segs.forEach(function (s) { appendSeg(p.body, s[0], s[1]); });
      for (var k = 0; k < p.loops; k++) appendSeg(p.body, 'loop', p.more(k));
      finish(p);
    }

    function play() {
      if (reduceMotion) { panels.forEach(renderAll); return; }
      panels.forEach(function (p) {
        var si = 0, ci = 0, extra = 0, CHARS = 6;
        var timer = setInterval(function () {
          if (si < p.segs.length) {
            var s = p.segs[si];
            appendSeg(p.body, s[0], s[1].slice(ci, ci + CHARS));
            ci += CHARS;
            if (ci >= s[1].length) { si++; ci = 0; }
          } else if (extra < p.loops) {
            appendSeg(p.body, 'loop', p.more(extra));
            extra++;
          } else {
            clearInterval(timer);
            finish(p);
            return;
          }
          p.outer.scrollTop = p.outer.scrollHeight;
        }, 28);
      });
    }
    whenVisible(host, play);
  })();

  /* ===================== Figure 4 (left): prediction errors by entropy ===================== */
  (function () {
    var c = D.fig1c;
    var SER = [
      { key: 'under', label: 'Under', color: 'var(--student)', dash: null },
      { key: 'over', label: 'Over', color: 'var(--c-over)', dash: null },
      { key: 'both', label: 'Both', color: 'var(--c-fp)', dash: '5 4' }
    ];
    var W = 380, H = 300, m = { l: 44, r: 12, t: 12, b: 48 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b, XMAX = 2;
    var svg = sv('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Share of prefixes with under-, over-, or both kinds of prediction errors by student entropy' });
    document.getElementById('fig4a-svg').appendChild(svg);
    var x = function (v) { return m.l + iw * v / XMAX; };
    var y = function (v) { return m.t + ih * (1 - v / 100); };
    var grid = sv('g', { 'class': 'grid' }, svg);
    [0, 25, 50, 75, 100].forEach(function (v) {
      sv('line', { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }, grid);
      sv('text', { x: m.l - 8, y: y(v) + 4, 'text-anchor': 'end', text: v }, svg);
    });
    [0, 0.5, 1, 1.5, 2].forEach(function (v) {
      sv('text', { x: x(v), y: H - m.b + 20, 'text-anchor': 'middle', text: v.toFixed(1) }, svg);
    });
    sv('line', { 'class': 'baseline', x1: m.l, x2: W - m.r, y1: y(0), y2: y(0) }, svg);
    sv('text', { x: m.l + iw / 2, y: H - 8, 'text-anchor': 'middle', text: 'Student entropy (nats)' }, svg);
    sv('text', { transform: 'translate(12,' + (m.t + ih / 2) + ') rotate(-90)', 'text-anchor': 'middle', text: 'Prefixes (%)' }, svg);

    var groups = {};
    SER.forEach(function (s) {
      var g = sv('g', { 'class': 'series' }, svg);
      var band = sv('polygon', { 'class': 'band', points: c.band[s.key].map(function (p) { return x(p[0]).toFixed(1) + ',' + y(p[1]).toFixed(1); }).join(' ') }, g);
      band.style.fill = s.color;
      band.style.fillOpacity = 0.13;
      var path = sv('path', { d: 'M' + c.line[s.key].map(function (p) { return x(p[0]).toFixed(1) + ',' + y(p[1]).toFixed(1); }).join('L') }, g);
      path.style.stroke = s.color;
      path.style.strokeWidth = 2.2;
      if (s.dash) path.style.strokeDasharray = s.dash;
      c.line[s.key].forEach(function (p) {
        var dot = sv('circle', { cx: x(p[0]), cy: y(p[1]), r: 3.2 }, g);
        dot.style.fill = 'var(--surface)';
        dot.style.stroke = s.color;
        dot.style.strokeWidth = 1.6;
      });
      groups[s.key] = g;
    });

    // Hover: snap to the nearest entropy bin and show all three shares.
    var guide = sv('line', { 'class': 'hover-line', y1: m.t, y2: y(0), opacity: 0 }, svg);
    var hit = sv('rect', { x: m.l, y: m.t, width: iw, height: ih, fill: 'transparent' }, svg);
    hit.addEventListener('mousemove', function (evt) {
      var pt = svg.createSVGPoint();
      pt.x = evt.clientX; pt.y = evt.clientY;
      var loc = pt.matrixTransform(svg.getScreenCTM().inverse());
      var v = (loc.x - m.l) / iw * XMAX, best = 0;
      c.line.under.forEach(function (p, i) { if (Math.abs(p[0] - v) < Math.abs(c.line.under[best][0] - v)) best = i; });
      var b = c.bins[best];
      guide.setAttribute('x1', x(c.line.under[best][0]));
      guide.setAttribute('x2', x(c.line.under[best][0]));
      guide.setAttribute('opacity', 1);
      showTip(evt, '<b>Entropy ' + b[0] + (b[1] === null ? '+' : '–' + b[1]) + '</b> (' + fmt(b[2]) + ' prefixes)<br>' +
        SER.map(function (s) { return s.label + ': ' + c.line[s.key][best][1].toFixed(1) + '%'; }).join('<br>'));
    });
    hit.addEventListener('mouseleave', function () { guide.setAttribute('opacity', 0); hideTip(); });

    var focus = null;
    var legend = document.getElementById('fig4a-legend');
    SER.forEach(function (s) {
      var sw = s.dash ? 'background:repeating-linear-gradient(90deg,' + s.color + ' 0 5px,transparent 5px 8px)' : 'background:' + s.color;
      var item = el('span', { html: '<i style="' + sw + '"></i>' + s.label }, legend);
      item.addEventListener('click', function () {
        focus = focus === s.key ? null : s.key;
        SER.forEach(function (o) { groups[o.key].classList.toggle('dim', focus !== null && focus !== o.key); });
      });
    });
  })();

  /* ===================== Figure 4 (right): Dynamic KL explorer ===================== */
  (function () {
    var cands = D.dkl.cands.map(function (c) { return { tok: c[0], p: c[1], q: c[2] }; });
    cands.sort(function (a, b) { return Math.max(b.p, b.q) - Math.max(a.p, a.q); });
    var ctx = document.getElementById('dkl-context');
    el('span', { text: '…' + D.dkl.context + ' ' }, ctx);
    var slot = el('span', { 'class': 'slot' }, ctx);

    var W = 460, H = 210, m = { l: 34, r: 6, t: 26, b: 34 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b, YMAX = 0.5;
    var svg = sv('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Teacher and student probabilities for eight candidates' });
    document.getElementById('dkl-chart').appendChild(svg);
    var bw = iw / cands.length;
    var y = function (v) { return m.t + ih * (1 - Math.min(v, YMAX) / YMAX); };
    [0, 0.25, 0.5].forEach(function (v) {
      sv('line', { 'class': 'gl', x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }, svg);
      sv('text', { x: m.l - 6, y: y(v) + 4, 'text-anchor': 'end', 'font-size': 11, text: v }, svg);
    });
    cands.forEach(function (c, i) {
      var g = sv('g', { 'class': 'cand', tabindex: 0, role: 'button', 'aria-label': 'candidate ' + c.tok.trim() }, svg);
      var x0 = m.l + i * bw;
      c.bg = sv('rect', { 'class': c.p > c.q ? 'bg-under' : 'bg-over', x: x0 + 2, y: m.t, width: bw - 4, height: ih, rx: 4 }, g);
      sv('rect', { 'class': 'bar-p', x: x0 + bw / 2 - 13, y: y(c.p), width: 12, height: y(0) - y(c.p), rx: 2 }, g);
      sv('rect', { 'class': 'bar-q', x: x0 + bw / 2 + 1, y: y(c.q), width: 12, height: y(0) - y(c.q), rx: 2 }, g);
      sv('text', { 'class': 'tok', x: x0 + bw / 2, y: H - m.b + 16, 'text-anchor': 'middle', 'font-size': 12, text: c.tok.trim() }, g);
      g.addEventListener('click', function () { select(i); });
      g.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(i); } });
    });
    var lg = sv('g', { transform: 'translate(' + (W - 176) + ',6)' }, svg);
    sv('rect', { 'class': 'bar-p', x: 0, y: 0, width: 10, height: 10, rx: 2 }, lg);
    sv('text', { x: 14, y: 9, 'font-size': 11, text: 'teacher p' }, lg);
    sv('rect', { 'class': 'bar-q', x: 84, y: 0, width: 10, height: 10, rx: 2 }, lg);
    sv('text', { x: 98, y: 9, 'font-size': 11, text: 'student q' }, lg);

    var readout = document.getElementById('dkl-readout');
    var cells = ['teacher p', 'student q', 'ratio', 'α (FKL weight)'].map(function (k) {
      var d = el('div', {}, readout);
      el('span', { 'class': 'k', text: k }, d);
      return el('span', { 'class': 'v' }, d);
    });
    var mixF = document.querySelector('#dkl-mix .f'), mixR = document.querySelector('#dkl-mix .r');
    var note = document.getElementById('dkl-note');

    function select(i) {
      var c = cands[i];
      var alpha = c.p / (c.p + c.q);
      cands.forEach(function (o, j) { o.bg.style.opacity = j === i ? 1 : 0.35; });
      slot.textContent = c.tok.trim();
      cells[0].textContent = c.p.toFixed(3);
      cells[1].textContent = c.q < 0.001 ? c.q.toExponential(1) : c.q.toFixed(3);
      var r = c.p > c.q ? c.p / c.q : c.q / c.p;
      var rs = r >= 100 ? fmt(r) : r.toFixed(1);
      cells[2].textContent = (c.p > c.q ? 'p/q = ' : 'q/p = ') + rs;
      cells[3].textContent = alpha.toFixed(2);
      mixF.style.width = (100 * alpha) + '%';
      mixR.style.width = (100 * (1 - alpha)) + '%';
      mixF.textContent = alpha > 0.12 ? 'FKL ' + Math.round(100 * alpha) + '%' : '';
      mixR.textContent = alpha < 0.88 ? 'RKL ' + Math.round(100 * (1 - alpha)) + '%' : '';
      var t = '“' + c.tok.trim() + '”';
      if (r < 1.3) note.textContent = t + ' is nearly matched, so Dynamic KL weights the two directions almost equally.';
      else if (c.p > c.q) note.textContent = t + ' is underpredicted: the teacher gives it ' + rs + '× the student\'s probability, so α > 1/2 and forward KL dominates, pulling q up.';
      else note.textContent = t + ' is overpredicted: the student gives it ' + rs + '× the teacher\'s probability, so α < 1/2 and reverse KL dominates, pushing q down.';
    }
    select(0);
  })();

  // Shared axis scaffolding for the small charts below.
  function frame(host, W, H, m, label) {
    var svg = sv('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': label });
    host.appendChild(svg);
    return { svg: svg, W: W, H: H, m: m, iw: W - m.l - m.r, ih: H - m.t - m.b };
  }
  function yAxis(f, ticks, y, fmtTick) {
    var g = sv('g', { 'class': 'grid' }, f.svg);
    ticks.forEach(function (v) {
      sv('line', { x1: f.m.l, x2: f.W - f.m.r, y1: y(v), y2: y(v) }, g);
      sv('text', { x: f.m.l - 7, y: y(v) + 4, 'text-anchor': 'end', text: fmtTick ? fmtTick(v) : v }, f.svg);
    });
  }
  function legendItems(id, items) {
    var host = document.getElementById(id);
    items.forEach(function (it) { el('span', { html: it }, host); });
  }

  /* ===================== Figure 6 (left): correction direction ===================== */
  (function () {
    var d = D.fig6b;
    var OBJ = [
      { k: 'FKL', color: 'var(--fkl)' },
      { k: 'RKL', color: 'var(--rkl)' },
      { k: 'DKL', color: 'var(--student)', label: 'Dynamic KL' }
    ];
    var f = frame(document.getElementById('fig6a-svg'), 440, 300, { l: 44, r: 10, t: 14, b: 62 }, 'Share of candidates moved toward the teacher by each objective');
    var slots = 9, x = function (i) { return f.m.l + (i + 0.5) * f.iw / slots; };
    var y = function (v) { return f.m.t + f.ih * (1 - (v - 60) / 40); };
    sv('rect', { 'class': 'zone-under', x: f.m.l, y: f.m.t, width: 4 * f.iw / slots, height: f.ih, opacity: 0.6 }, f.svg);
    sv('rect', { 'class': 'zone-over', x: f.m.l + 5 * f.iw / slots, y: f.m.t, width: 4 * f.iw / slots, height: f.ih, opacity: 0.6 }, f.svg);
    yAxis(f, [60, 70, 80, 90, 100], y);
    sv('text', { transform: 'translate(12,' + (f.m.t + f.ih / 2) + ') rotate(-90)', 'text-anchor': 'middle', text: 'Moved toward teacher (%)' }, f.svg);
    ['under', 'over'].forEach(function (side, si) {
      var off = si * 5;
      d.bins.forEach(function (b, i) { sv('text', { x: x(off + i), y: f.H - f.m.b + 18, 'text-anchor': 'middle', 'class': 'small', text: b }, f.svg); });
      sv('text', { x: x(off + 1.5), y: f.H - 14, 'text-anchor': 'middle', text: side === 'under' ? 'Underpredicted (p/q)' : 'Overpredicted (q/p)' }, f.svg);
      OBJ.forEach(function (o) {
        var vals = d[side][o.k];
        var path = sv('path', { d: 'M' + vals.map(function (v, i) { return x(off + i).toFixed(1) + ',' + y(v).toFixed(1); }).join('L') }, f.svg);
        path.style.stroke = o.color; path.style.strokeWidth = o.k === 'DKL' ? 2.6 : 2; path.style.fill = 'none';
        vals.forEach(function (v, i) {
          var c = sv('circle', { cx: x(off + i), cy: y(v), r: o.k === 'DKL' ? 4.2 : 3.4 }, f.svg);
          c.style.fill = o.color; c.style.stroke = 'var(--surface)'; c.style.strokeWidth = 1.2; c.style.cursor = 'pointer';
          c.addEventListener('mousemove', function (evt) {
            showTip(evt, '<b>' + (o.label || o.k) + '</b><br>' + (side === 'under' ? 'p/q ' : 'q/p ') + d.bins[i] + '<br>' + v.toFixed(2) + '% moved toward teacher');
          });
          c.addEventListener('mouseleave', hideTip);
        });
      });
    });
    legendItems('fig6a-legend', OBJ.map(function (o) { return '<i style="background:' + o.color + '"></i>' + (o.label || o.k); }));
  })();

  /* ===================== Figure 6 (right): repetition during training ===================== */
  (function () {
    var d = D.fig6c;
    var SER = [
      { k: 'Initialization', color: 'var(--c-onq)', w: 2.6 },
      { k: 'Step 50', color: 'var(--c-eqat)', w: 2.2 },
      { k: 'Step 100', color: 'var(--student)', w: 2 },
      { k: 'Step 200', color: 'var(--c-rqat)', w: 2 }
    ];
    var f = frame(document.getElementById('fig6b-svg'), 440, 300, { l: 44, r: 14, t: 26, b: 62 }, 'Share of repeated tokens by response position during training');
    var x = function (v) { return f.m.l + f.iw * v / 16; };
    var y = function (v) { return f.m.t + f.ih * (1 - v / 25); };
    yAxis(f, [0, 5, 10, 15, 20, 25], y);
    [0, 4, 8, 12, 16].forEach(function (v) { sv('text', { x: x(v), y: f.H - f.m.b + 18, 'text-anchor': 'middle', text: v === 0 ? '0' : v + 'K' }, f.svg); });
    sv('text', { x: f.m.l + f.iw / 2, y: f.H - 14, 'text-anchor': 'middle', text: 'Position in response (tokens)' }, f.svg);
    sv('text', { transform: 'translate(12,' + (f.m.t + f.ih / 2) + ') rotate(-90)', 'text-anchor': 'middle', text: 'Repeated tokens (%)' }, f.svg);
    [[4, 'cap · steps 1–50'], [8, 'steps 51–100'], [16, '101+']].forEach(function (c) {
      sv('line', { 'class': 'cap', x1: x(c[0]), x2: x(c[0]), y1: f.m.t - 6, y2: y(0) }, f.svg);
      sv('text', { 'class': 'small', x: x(c[0]) - 4, y: f.m.t - 10, 'text-anchor': 'end', text: c[1] }, f.svg);
    });
    SER.forEach(function (s) {
      var v = d.series[s.k], xs = d.x;
      var pts = [];
      v.forEach(function (val, i) { pts.push([x(xs[i]), y(val)], [x(xs[i + 1]), y(val)]); });
      var path = sv('path', { d: 'M' + pts.map(function (p) { return p[0].toFixed(1) + ',' + p[1].toFixed(1); }).join('L') }, f.svg);
      path.style.stroke = s.color; path.style.strokeWidth = s.w; path.style.fill = 'none';
      v.forEach(function (val, i) {
        var hit = sv('rect', { x: x(xs[i]), y: y(val) - 7, width: x(xs[i + 1]) - x(xs[i]), height: 14, fill: 'transparent' }, f.svg);
        hit.style.cursor = 'pointer';
        hit.addEventListener('mousemove', function (evt) {
          showTip(evt, '<b>' + s.k + '</b><br>tokens ' + xs[i] + 'K–' + xs[i + 1] + 'K<br>' + val.toFixed(2) + '% repeated');
        });
        hit.addEventListener('mouseleave', hideTip);
        if (s.k === 'Initialization') sv('text', { 'class': 'small', x: (x(xs[i]) + x(xs[i + 1])) / 2, y: y(val) - 6, 'text-anchor': 'middle', text: ['6.7', '15.6', '21.1'][i] }, f.svg);
      });
    });
    legendItems('fig6b-legend', SER.map(function (s) { return '<i style="background:' + s.color + '"></i>' + s.k; }));
  })();

  /* ===================== Figure 7: accuracy versus mean generated tokens ===================== */
  (function () {
    var METHODS = ['EfficientQAT', 'BitDistiller', 'ReasoningQAT', 'OnQ'];
    var SIZES = ['1.7B', '4B', '8B'];
    var BENCH = [['math500', 'MATH-500'], ['aime90', 'AIME 24–26'], ['livecodebench', 'LiveCodeBench']];
    var host = document.getElementById('fig7');
    function marker(g, size, cx, cy, color, big) {
      var r = big ? 5.6 : 4.8, node;
      if (size === '1.7B') node = sv('circle', { cx: cx, cy: cy, r: r }, g);
      else if (size === '4B') node = sv('path', { d: 'M' + cx + ',' + (cy - r * 1.15) + 'L' + (cx + r * 1.1) + ',' + (cy + r * 0.8) + 'L' + (cx - r * 1.1) + ',' + (cy + r * 0.8) + 'Z' }, g);
      else node = sv('rect', { x: cx - r * 0.9, y: cy - r * 0.9, width: r * 1.8, height: r * 1.8, rx: 1 }, g);
      node.style.fill = color; node.style.stroke = 'var(--surface)'; node.style.strokeWidth = 1.2; node.style.cursor = 'pointer';
      return node;
    }
    function draw(bits) {
      host.innerHTML = '';
      var rows = D.fig7[bits];
      BENCH.forEach(function (b) {
        var cell = el('div', {}, host);
        var pts = rows.filter(function (r) { return r[2] === b[0]; });
        var tx = pts.map(function (r) { return r[3]; }), ay = pts.map(function (r) { return r[4]; });
        var step = 2000;
        var x0 = Math.floor(Math.min.apply(null, tx) / step) * step, x1 = Math.ceil(Math.max.apply(null, tx) / step) * step;
        var ys = 10, y0 = Math.max(0, Math.floor((Math.min.apply(null, ay) - 3) / ys) * ys), y1 = Math.min(100, Math.ceil((Math.max.apply(null, ay) + 3) / ys) * ys);
        var f = frame(cell, 300, 270, { l: 38, r: 12, t: 28, b: 40 }, b[1] + ' accuracy versus mean tokens');
        var x = function (v) { return f.m.l + f.iw * (v - x0) / (x1 - x0); };
        var y = function (v) { return f.m.t + f.ih * (1 - (v - y0) / (y1 - y0)); };
        var yt = []; for (var v = y0; v <= y1; v += (y1 - y0 > 40 ? 20 : 10)) yt.push(v);
        yAxis(f, yt, y);
        var span = x1 - x0, xstep = span > 12000 ? 4000 : 2000, xt = [];
        for (var u = x0; u <= x1; u += xstep) xt.push(u);
        xt.forEach(function (u) { sv('text', { x: x(u), y: f.H - f.m.b + 16, 'text-anchor': 'middle', 'class': 'small', text: (u / 1000) + 'K' }, f.svg); });
        sv('text', { 'class': 'title', x: f.m.l, y: 16, text: b[1] }, f.svg);
        sv('text', { x: f.m.l + f.iw / 2, y: f.H - 4, 'text-anchor': 'middle', 'class': 'small', text: 'mean generated tokens' }, f.svg);
        METHODS.forEach(function (meth) {
          var mp = SIZES.map(function (s) { return pts.filter(function (r) { return r[0] === meth && r[1] === s; })[0]; });
          var line = sv('path', { d: 'M' + mp.map(function (r) { return x(r[3]).toFixed(1) + ',' + y(r[4]).toFixed(1); }).join('L') }, f.svg);
          line.style.stroke = COLORS[meth]; line.style.strokeWidth = meth === 'OnQ' ? 2 : 1.4; line.style.strokeDasharray = '4 3'; line.style.fill = 'none';
          line.style.opacity = meth === 'OnQ' ? 1 : 0.75;
        });
        METHODS.forEach(function (meth) {
          pts.filter(function (r) { return r[0] === meth; }).forEach(function (r) {
            var node = marker(f.svg, r[1], x(r[3]), y(r[4]), COLORS[meth], meth === 'OnQ');
            node.addEventListener('mousemove', function (evt) {
              showTip(evt, '<b>' + meth + '</b> · Qwen3-' + r[1] + ' · ' + bits + '-bit<br>' + b[1] + ': ' + r[4].toFixed(2) + '%<br>mean ' + fmt(r[3]) + ' tokens');
            });
            node.addEventListener('mouseleave', hideTip);
          });
        });
      });
    }
    legendItems('fig7-legend', METHODS.map(function (m) { return '<i style="background:' + COLORS[m] + '"></i>' + (m === 'OnQ' ? '<span class="me">OnQ</span>' : m); })
      .concat(['<span class="shape">● 1.7B &nbsp;▲ 4B &nbsp;■ 8B</span>']));
    var buttons = document.querySelectorAll('#fig7-bits button');
    buttons.forEach(function (btn) {
      btn.addEventListener('click', function () {
        buttons.forEach(function (o) { o.setAttribute('aria-pressed', String(o === btn)); });
        draw(btn.getAttribute('data-bits'));
      });
    });
    draw('2');
  })();

  /* ===================== Figure 8: generation outcomes ===================== */
  (function () {
    var d = D.fig8;
    var PARTS = [['correct', 'Correct', 'var(--o-correct)'], ['rep', 'Repetitive error', 'var(--o-rep)'], ['other', 'Other error', 'var(--o-other)']];
    var host = document.getElementById('fig8');
    ['math', 'aime'].forEach(function (k, pi) {
      var cell = el('div', {}, host);
      var f = frame(cell, 460, 230, { l: pi === 0 ? 110 : 110, r: 12, t: 28, b: 34 }, d[k].title + ' generation outcomes');
      var x = function (v) { return f.m.l + f.iw * v / 100; };
      var bh = f.ih / d.methods.length;
      sv('text', { 'class': 'title', x: f.m.l, y: 16, text: d[k].title }, f.svg);
      [0, 25, 50, 75, 100].forEach(function (v) { sv('text', { 'class': 'small', x: x(v), y: f.H - f.m.b + 16, 'text-anchor': 'middle', text: v + '%' }, f.svg); });
      d.methods.forEach(function (meth, i) {
        var yy = f.m.t + i * bh + bh * 0.18, h = bh * 0.64, acc = 0;
        var lab = sv('text', { x: f.m.l - 8, y: yy + h / 2 + 4, 'text-anchor': 'end', text: meth }, f.svg);
        if (meth === 'OnQ') { lab.style.fill = 'var(--c-onq)'; lab.style.fontWeight = 600; }
        PARTS.forEach(function (p) {
          var v = d[k][p[0]][i];
          if (v <= 0) return;
          var r = sv('rect', { 'class': 'seg-rect', x: x(acc), y: yy, width: x(acc + v) - x(acc), height: h }, f.svg);
          r.style.fill = p[2];
          r.addEventListener('mousemove', function (evt) { showTip(evt, '<b>' + meth + '</b><br>' + p[1] + ': ' + v.toFixed(1) + '%'); });
          r.addEventListener('mouseleave', hideTip);
          if (p[0] === 'rep' && v >= 9) sv('text', { 'class': 'val', x: x(acc + v / 2), y: yy + h / 2 + 4, 'text-anchor': 'middle', text: v.toFixed(1) }, f.svg);
          acc += v;
        });
      });
    });
    legendItems('fig8-legend', PARTS.map(function (p) { return '<i style="background:' + p[2] + ';height:10px;width:12px"></i>' + p[1]; }));
  })();

  /* ===================== Table 1: main results ===================== */
  (function () {
    var BENCH = ['GSM8K', 'MATH-<br>500', 'AIME<br>24–26', 'Human<br>Eval+', 'MBPP+', 'LiveCode<br>Bench', 'MMLU-<br>Redux', 'GPQA-<br>Diamond', 'IFEval', 'Avg.'];
    var PTQ = ['GPTQ', 'AutoRound'], QAT = ['EfficientQAT', 'BitDistiller', 'ReasoningQAT', 'OnQ'];
    var state = { model: '8B', bits: '2', trunc: false };
    var table = document.getElementById('res-table');

    function row(model, bits, method) {
      return D.results.filter(function (r) { return r.model === model && r.bits === bits && r.method === method; })[0];
    }
    function cell(v, isAvg) {
      var cls = (v[2] ? 'best' : '') + (isAvg ? ' div' : '');
      return '<td class="' + cls + '">' + v[0].toFixed(2) + (state.trunc ? '<span class="tr">' + v[1].toFixed(2) + '</span>' : '') + '</td>';
    }
    function rowHtml(r, cls, label) {
      return '<tr class="' + cls + '"><td>' + label + '</td><td>' + (r.bits === '16' ? '16/16' : r.bits + '/16') + '</td>' +
        r.v.map(function (v, i) { return cell(v, i === r.v.length - 1); }).join('') + '</tr>';
    }
    function draw() {
      var head = '<thead><tr><th>Method</th><th>W/A</th>' + BENCH.map(function (b, i) {
        return '<th' + (i === BENCH.length - 1 ? ' class="div"' : '') + '>' + b + '</th>';
      }).join('') + '</tr></thead>';
      var body = '<tbody>' + rowHtml(row(state.model, '16', 'Full precision'), 'fp', 'Full precision');
      body += '<tr class="group"><td colspan="12">Post-training quantization</td></tr>';
      PTQ.forEach(function (n) { body += rowHtml(row(state.model, state.bits, n), '', n); });
      body += '<tr class="group"><td colspan="12">Quantization-aware training</td></tr>';
      QAT.forEach(function (n) { body += rowHtml(row(state.model, state.bits, n), n === 'OnQ' ? 'onq' : '', n === 'OnQ' ? 'OnQ (ours)' : n); });
      table.innerHTML = head + body + '</tbody>';
    }
    function bindSeg(id, key, attr) {
      var buttons = document.querySelectorAll('#' + id + ' button');
      buttons.forEach(function (b) {
        b.addEventListener('click', function () {
          state[key] = b.getAttribute(attr);
          buttons.forEach(function (o) { o.setAttribute('aria-pressed', String(o === b)); });
          draw();
        });
      });
    }
    bindSeg('model-seg', 'model', 'data-model');
    bindSeg('bits-seg', 'bits', 'data-bits');
    document.getElementById('show-trunc').addEventListener('change', function (e) { state.trunc = e.target.checked; draw(); });
    draw();
  })();

  /* ===================== Table 2: recovery stages ===================== */
  (function () {
    var AVG = 9, LCB = 5;
    function stage(model, bits, method) {
      return D.stages.filter(function (r) { return r.model === model && r.bits === bits && r.method === method; })[0];
    }
    function fp(model) {
      return D.results.filter(function (r) { return r.model === model && r.method === 'Full precision'; })[0];
    }
    var html = '<thead><tr><th rowspan="2">Setting</th><th colspan="4">Nine-benchmark average</th><th colspan="3" class="div">LiveCodeBench</th></tr>' +
      '<tr><th>RTN</th><th>+ Calibration</th><th>OnQ</th><th>Full precision</th><th class="div">+ Calibration</th><th>OnQ</th><th>Full precision</th></tr></thead><tbody>';
    ['1.7B', '4B', '8B'].forEach(function (model) {
      ['3', '2'].forEach(function (bits) {
        var rtn = stage(model, bits, 'RTN'), cal = stage(model, bits, '+ Calibration'), onq = stage(model, bits, 'OnQ'), f = fp(model);
        html += '<tr><td>Qwen3-' + model + ' W' + bits + '</td>' +
          '<td>' + rtn.v[AVG][0].toFixed(2) + '</td><td>' + cal.v[AVG][0].toFixed(2) + '</td>' +
          '<td class="hi">' + onq.v[AVG][0].toFixed(2) + '</td><td class="dim">' + f.v[AVG][0].toFixed(2) + '</td>' +
          '<td class="div">' + cal.v[LCB][0].toFixed(2) + '</td><td class="hi">' + onq.v[LCB][0].toFixed(2) + '</td>' +
          '<td class="dim">' + f.v[LCB][0].toFixed(2) + '</td></tr>';
      });
    });
    document.getElementById('stage-table').innerHTML = html + '</tbody>';
  })();

})();
