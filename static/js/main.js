(function () {
  "use strict";

  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  const PALETTE = ["#3df5a7", "#34d1ff", "#ff5fa2", "#ffc94d", "#a78bfa", "#ff8a4c", "#5eead4", "#f472b6", "#93c5fd", "#bef264"];

  /* ---------------- Scroll progress + nav ---------------- */
  const progress = $("#scrollProgress");
  const nav = $("#nav");
  function onScroll() {
    const h = document.documentElement;
    const p = h.scrollTop / Math.max(1, h.scrollHeight - h.clientHeight);
    progress.style.transform = `scaleX(${p})`;
    nav.classList.toggle("scrolled", h.scrollTop > 40);
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  const navLinks = $$(".nav-links a");
  const sectionObs = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        navLinks.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + e.target.id));
      });
    },
    { rootMargin: "-45% 0px -50% 0px" }
  );
  navLinks.forEach((a) => {
    const s = $(a.getAttribute("href"));
    if (s) sectionObs.observe(s);
  });

  $$('a[aria-disabled="true"]').forEach((a) => a.addEventListener("click", (e) => e.preventDefault()));

  /* ---------------- Reveal + counters ---------------- */
  function countUp(el) {
    const target = parseFloat(el.dataset.count);
    const dec = parseInt(el.dataset.decimals || "0", 10);
    const pre = el.dataset.prefix || "";
    const suf = el.dataset.suffix || "";
    if (reduceMotion) {
      el.textContent = pre + target.toFixed(dec) + suf;
      return;
    }
    const dur = 1400;
    const t0 = performance.now();
    (function tick(t) {
      const k = Math.min(1, (t - t0) / dur);
      const e = 1 - Math.pow(1 - k, 4);
      el.textContent = pre + (target * e).toFixed(dec) + suf;
      if (k < 1) requestAnimationFrame(tick);
    })(t0);
  }

  const revealObs = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        e.target.classList.add("in");
        $$("[data-count]", e.target).forEach(countUp);
        if (e.target.matches("[data-count]")) countUp(e.target);
        revealObs.unobserve(e.target);
      });
    },
    { threshold: 0.12 }
  );
  $$(".reveal").forEach((el, i) => {
    el.style.setProperty("--d", `${(i % 6) * 60}ms`);
    revealObs.observe(el);
  });

  /* ---------------- Spotlight cards ---------------- */
  $$(".spot").forEach((el) => {
    el.addEventListener("pointermove", (ev) => {
      const r = el.getBoundingClientRect();
      el.style.setProperty("--mx", `${ev.clientX - r.left}px`);
      el.style.setProperty("--my", `${ev.clientY - r.top}px`);
    });
  });

  /* ---------------- Tilt ---------------- */
  if (!reduceMotion) {
    $$(".tilt").forEach((el) => {
      el.addEventListener("pointermove", (ev) => {
        const r = el.getBoundingClientRect();
        const x = (ev.clientX - r.left) / r.width - 0.5;
        const y = (ev.clientY - r.top) / r.height - 0.5;
        el.style.transform = `perspective(1400px) rotateX(${-y * 4}deg) rotateY(${x * 5}deg)`;
      });
      el.addEventListener("pointerleave", () => (el.style.transform = ""));
    });
  }

  /* ---------------- Hero typer ---------------- */
  const PHRASES = [
    '{"type":"run_skill","skill":"visual_appearance","start_sec":0.0,"end_sec":5.2}',
    '{"type":"run_skill","skill":"physical_consistency","start_sec":2.5,"end_sec":4.5}',
    '{"type":"run_skill","skill":"ocr_consistency","at_sec":2.0}',
    '{"type":"run_skill","skill":"common_sense","start_sec":0.0,"end_sec":5.0}',
    '{"type":"run_skill","skill":"voice_profile"}',
    '{"type":"answer","content":"fake"}'
  ];
  (async function typer() {
    const el = $("#typer");
    if (reduceMotion) {
      el.textContent = PHRASES[0];
      return;
    }
    let i = 0;
    for (;;) {
      const s = PHRASES[i % PHRASES.length];
      for (let k = 1; k <= s.length; k++) {
        el.textContent = s.slice(0, k);
        await sleep(22 + Math.random() * 30);
      }
      await sleep(1800);
      for (let k = s.length; k >= 0; k--) {
        el.textContent = s.slice(0, k);
        await sleep(10);
      }
      await sleep(300);
      i++;
    }
  })();

  /* ---------------- Hero canvas: windows of the background are being "inspected" ---------------- */
  (function heroCanvas() {
    const cv = $("#heroCanvas");
    const ctx = cv.getContext("2d");
    let W = 0, H = 0, dpr = 1;
    const mouse = { x: -9999, y: -9999 };
    const items = [];
    const LABELS = ["appearance · issue", "physics · issue", "OCR · text_suspicious", "common sense", "no_issue", "inconclusive", "voice profile", "glyph drift", "texture flicker", "hand merge", "ASR · timing"];
    let running = true;
    let lastSpawn = 0;

    function resize() {
      dpr = Math.min(2, window.devicePixelRatio || 1);
      W = cv.clientWidth;
      H = cv.clientHeight;
      cv.width = W * dpr;
      cv.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    resize();
    window.addEventListener("resize", resize);
    cv.parentElement.addEventListener("pointermove", (e) => {
      const r = cv.getBoundingClientRect();
      mouse.x = e.clientX - r.left;
      mouse.y = e.clientY - r.top;
    });
    cv.parentElement.addEventListener("pointerleave", () => { mouse.x = mouse.y = -9999; });

    function spawn(t) {
      let x, y, w, h, tries = 0;
      do {
        w = 60 + Math.random() * 180;
        h = 50 + Math.random() * 160;
        x = Math.random() * (W - w);
        y = 70 + Math.random() * (H - h - 90);
        tries++;
      } while (tries < 8 && Math.abs(x + w / 2 - W / 2) < W * 0.26 && Math.abs(y + h / 2 - H / 2) < H * 0.3);
      const isPoint = Math.random() < 0.25;
      const s = (v) => ((v / W) * 8).toFixed(1);
      items.push({
        x, y, w, h, t0: t, life: 3600 + Math.random() * 1600, isPoint,
        color: PALETTE[Math.floor(Math.random() * 5)],
        label: LABELS[Math.floor(Math.random() * LABELS.length)],
        coords: isPoint ? `@${s(x + w / 2)}s` : `[${s(x)}s – ${s(x + w)}s]`
      });
    }

    function hexA(hex, a) {
      const v = parseInt(hex.slice(1), 16);
      return `rgba(${(v >> 16) & 255},${(v >> 8) & 255},${v & 255},${a})`;
    }

    function drawGrid() {
      const gap = 30;
      for (let gx = gap / 2; gx < W; gx += gap) {
        for (let gy = gap / 2; gy < H; gy += gap) {
          const d = Math.hypot(gx - mouse.x, gy - mouse.y);
          const k = Math.max(0, 1 - d / 180);
          ctx.fillStyle = k > 0 ? `rgba(61,245,167,${0.1 + k * 0.6})` : "rgba(255,255,255,0.07)";
          ctx.beginPath();
          ctx.arc(gx, gy, 0.9 + k * 1.4, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    function drawItem(it, t) {
      const age = t - it.t0;
      const appear = Math.min(1, age / 550);
      const fade = Math.max(0, Math.min(1, (it.life - age) / 900));
      const a = Math.min(appear, fade);
      const ease = 1 - Math.pow(1 - appear, 3);
      ctx.save();
      if (it.isPoint) {
        const cx = it.x + it.w / 2, cy = it.y + it.h / 2;
        const ring = (age % 1400) / 1400;
        ctx.strokeStyle = hexA(it.color, a * (1 - ring) * 0.8);
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(cx, cy, 6 + ring * 22, 0, Math.PI * 2); ctx.stroke();
        ctx.fillStyle = hexA(it.color, a * 0.95);
        ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2); ctx.fill();
        ctx.font = "500 10.5px 'JetBrains Mono', monospace";
        ctx.fillStyle = hexA(it.color, a * 0.7);
        ctx.fillText(it.coords, cx + 10, cy - 8);
      } else {
        const { x, y, w, h } = it;
        ctx.fillStyle = hexA(it.color, a * 0.05);
        ctx.fillRect(x, y, w, h);
        ctx.strokeStyle = hexA(it.color, a * 0.28);
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 4]);
        ctx.strokeRect(x, y, w * ease, h * ease);
        ctx.setLineDash([]);
        const c = Math.min(16, w / 4, h / 4) * ease;
        ctx.strokeStyle = hexA(it.color, a * 0.95);
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x, y + c); ctx.lineTo(x, y); ctx.lineTo(x + c, y);
        ctx.moveTo(x + w - c, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + c);
        ctx.moveTo(x + w, y + h - c); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - c, y + h);
        ctx.moveTo(x + c, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - c);
        ctx.stroke();
        ctx.font = "600 10.5px 'JetBrains Mono', monospace";
        const txt = it.label;
        const tw = ctx.measureText(txt).width + 10;
        ctx.fillStyle = hexA(it.color, a * 0.9);
        ctx.fillRect(x, y - 17, tw, 16);
        ctx.fillStyle = `rgba(4,7,10,${a})`;
        ctx.fillText(txt, x + 5, y - 5);
        ctx.font = "400 10px 'JetBrains Mono', monospace";
        ctx.fillStyle = hexA(it.color, a * 0.55);
        ctx.fillText(it.coords.slice(0, Math.ceil(it.coords.length * ease)), x + 2, y + h + 13);
      }
      ctx.restore();
    }

    function frame(t) {
      if (!running) return;
      ctx.clearRect(0, 0, W, H);
      drawGrid();
      if (t - lastSpawn > 520 && items.length < 9) {
        spawn(t);
        lastSpawn = t;
      }
      for (let i = items.length - 1; i >= 0; i--) {
        if (t - items[i].t0 > items[i].life) items.splice(i, 1);
        else drawItem(items[i], t);
      }
      requestAnimationFrame(frame);
    }

    if (reduceMotion) {
      drawGrid();
      return;
    }
    new IntersectionObserver((es) => {
      const vis = es[0].isIntersecting;
      if (vis && !running) { running = true; requestAnimationFrame(frame); }
      else if (!vis) running = false;
    }).observe(cv);
    requestAnimationFrame(frame);
  })();

  /* ---------------- Benchmark explorer ---------------- */
  (function bench() {
    const B = window.BENCH;
    const tabs = $("#benchTabs");
    const chips = $("#benchChips");
    const meta = $("#benchMeta");
    const chart = $("#benchChart");
    let cat = 0;

    B.forEach((c, i) => {
      const b = document.createElement("button");
      b.className = "bench-tab";
      b.type = "button";
      b.textContent = c.label;
      b.addEventListener("click", () => selectCat(i));
      tabs.appendChild(b);
    });

    function selectCat(i) {
      cat = i;
      $$(".bench-tab", tabs).forEach((t, k) => t.classList.toggle("active", k === i));
      chips.innerHTML = "";
      B[i].items.forEach((it, k) => {
        const c = document.createElement("button");
        c.className = "chip";
        c.type = "button";
        c.textContent = it.name;
        c.addEventListener("click", () => selectItem(k));
        chips.appendChild(c);
      });
      selectItem(B[i].default || 0);
    }

    function selectItem(k) {
      const it = B[cat].items[k];
      $$(".chip", chips).forEach((c, j) => c.classList.toggle("active", j === k));
      meta.innerHTML = `<span class="metric">${it.metric}</span><span class="dir ${it.lower ? "down" : "up"}">${it.lower ? "↓ lower is better" : "↑ higher is better"}</span>`;
      const rows = it.rows.slice().sort((a, b) => (it.lower ? a.value - b.value : b.value - a.value));
      const max = Math.max(...rows.map((r) => r.value)) || 1;
      chart.innerHTML = "";
      rows.forEach((r, idx) => {
        const row = document.createElement("div");
        row.className = `bar-row g-${r.group}`;
        row.style.setProperty("--i", idx);
        row.innerHTML = `
          <span class="bar-name">${r.model}</span>
          <span class="bar-track"><span class="bar-fill"></span></span>
          <span class="bar-val">0</span>`;
        chart.appendChild(row);
        const fill = $(".bar-fill", row);
        const val = $(".bar-val", row);
        const pct = Math.max(1.5, (r.value / max) * 100);
        requestAnimationFrame(() => requestAnimationFrame(() => (fill.style.width = pct + "%")));
        if (reduceMotion) val.textContent = r.value.toFixed(1);
        else {
          const t0 = performance.now() + idx * 40;
          (function tick(t) {
            const p = Math.max(0, Math.min(1, (t - t0) / 900));
            val.textContent = (r.value * (1 - Math.pow(1 - p, 3))).toFixed(1);
            if (p < 1) requestAnimationFrame(tick);
          })(performance.now());
        }
      });
    }

    let started = false;
    new IntersectionObserver((es) => {
      if (es[0].isIntersecting && !started) {
        started = true;
        selectCat(0);
      }
    }, { threshold: 0.2 }).observe($(".bench"));
  })();

  /* ---------------- Lightbox ---------------- */
  const lb = $("#lightbox");
  const lbImg = $("#lbImg");
  const lbCap = $("#lbCap");
  function openLB(src, cap) {
    lbImg.src = src;
    lbCap.textContent = cap || "";
    lb.classList.add("open");
    lb.setAttribute("aria-hidden", "false");
    document.body.style.overflow = "hidden";
  }
  function closeLB() {
    lb.classList.remove("open");
    lb.setAttribute("aria-hidden", "true");
    document.body.style.overflow = "";
  }
  $$(".zoomable").forEach((el) => el.addEventListener("click", () => openLB(el.dataset.full, el.dataset.caption)));
  lb.addEventListener("click", (e) => { if (e.target !== lbImg) closeLB(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeLB(); });

  /* ---------------- Copy buttons ---------------- */
  $$(".copy").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const text = $("#" + btn.dataset.copy).innerText;
      try {
        await navigator.clipboard.writeText(text);
      } catch (_) {
        const ta = document.createElement("textarea");
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        ta.remove();
      }
      btn.textContent = "Copied ✓";
      btn.classList.add("ok");
      setTimeout(() => { btn.textContent = "Copy"; btn.classList.remove("ok"); }, 1600);
    });
  });
})();
