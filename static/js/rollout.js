(function () {
  "use strict";

  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  const ROLES = {
    visual_appearance: { en: "Visual appearance", zh: "视觉外观", color: "#34d1ff" },
    physical_consistency: { en: "Physical consistency", zh: "物理一致性", color: "#3df5a7" },
    common_sense: { en: "Common sense", zh: "常识", color: "#a78bfa" },
    ocr_consistency: { en: "OCR consistency", zh: "OCR 一致性", color: "#ffc94d" },
    voice_profile: { en: "Voice profile", zh: "声纹画像", color: "#ff5fa2" },
    asr_analysis: { en: "ASR analysis", zh: "语音转写分析", color: "#ff8a4c" },
    real: { en: "Real video", zh: "真实视频", color: "#c9d4ce" }
  };
  const VERDICT_ZH = {
    issue: "存在问题", text_suspicious: "文字可疑", timing_anomaly: "时序异常", profile_complete: "画像完成",
    no_issue: "无问题", ok: "正常", text_ok: "文字正常", inconclusive: "无法判断", uncertain: "不确定"
  };
  const CONF_ZH = { high: "高", medium: "中", low: "低", info: "信息" };
  const UI = {
    en: { turn: "Turn", investigator: "Investigator", call: "call", inspecting: "inspector running", verdict: "Verdict", gt: "ground truth", whole: "whole video", calls: "calls", issues: "issues raised", done: "done", thinking: "reasoning", inspectNow: "Inspecting", audio: "audio", noAudio: "no audio", more: "more", less: "less", running: "inspectors running", intervals: "evidence intervals", annotation: "annotation", annotated: "Annotated", case: "Case" },
    zh: { turn: "第", investigator: "主智能体", call: "调用", inspecting: "子智能体分析中", verdict: "判定", gt: "真实标签", whole: "整段视频", calls: "次调用", issues: "处问题", done: "完成", thinking: "推理中", inspectNow: "检查中", audio: "含音频", noAudio: "无音频", more: "展开", less: "收起", running: "个子智能体并行分析中", intervals: "条证据区间", annotation: "标注", annotated: "标注", case: "案例" }
  };
  const ALERT = new Set(["issue", "text_suspicious", "timing_anomaly"]);
  const OK = new Set(["no_issue", "ok", "text_ok"]);

  const SVG = {
    off: '<svg class="off" viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M16.5 12A4.5 4.5 0 0 0 14 7.97v2.21l2.45 2.45c.03-.2.05-.41.05-.63zM19 12c0 .94-.2 1.82-.54 2.64l1.51 1.51A8.8 8.8 0 0 0 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3 3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06a8.99 8.99 0 0 0 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4 9.91 6.09 12 8.18V4z"/></svg>',
    on: '<svg class="on" viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M3 9v6h4l5 5V4L7 9H3zm13.5 3A4.5 4.5 0 0 0 14 7.97v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/></svg>',
    skip: '<svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M4 18l8.5-6L4 6v12zm9-12v12l8.5-6L13 6z"/></svg>',
    replay: '<svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M12 5V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z"/></svg>'
  };

  function shell(opts) {
    return `
      <div class="demo-tabs" data-r="tabs" role="tablist"></div>
      ${opts.checkpoints ? '<div class="ckpt" data-r="ckpt"></div>' : ""}
      <div class="demo-body">
        <div class="demo-stage-wrap">
          <div class="case-meta" data-r="meta"></div>
          <div class="vstage" data-r="stage">
            <video data-r="video" muted playsinline preload="auto"></video>
            <div class="stage-scan" data-r="scan"></div>
            <div class="vbadge" data-r="badge"></div>
            <div class="vstamp" data-r="stamp"></div>
            <button class="vsound" data-r="sound" type="button" aria-label="Toggle sound" title="Toggle sound">${SVG.off}${SVG.on}</button>
          </div>
          <div class="vtimeline">
            <div class="vt-lanes" data-r="lanes"></div>
            <div class="vt-axis" data-r="axis"></div>
            <div class="vt-head" data-r="head"></div>
          </div>
          <div class="stage-legend" data-r="legend"></div>
        </div>
        <div class="terminal">
          <div class="term-head">
            <span class="dots"><i></i><i></i><i></i></span>
            <span class="term-title" data-r="title"></span>
            ${opts.langToggle ? '<div class="lang-toggle" data-r="lang" role="group" aria-label="Language"><button type="button" data-lang="en" class="active">EN</button><button type="button" data-lang="zh">中文</button></div>' : ""}
            <button class="replay" data-r="skip" type="button" aria-label="Show all turns">${SVG.skip}Skip</button>
            <button class="replay" data-r="replay" type="button" aria-label="Replay">${SVG.replay}Replay</button>
          </div>
          <div class="rollout" data-r="log"></div>
          <div class="term-foot">
            <span><i class="led"></i><span data-r="status">ready</span></span>
            <span data-r="stats"></span>
          </div>
        </div>
      </div>`;
  }

  function Rollout(root, CASES, opts) {
    if (!root || !CASES || !CASES.length) return;
    root.innerHTML = shell(opts);
    const R = (k) => $(`[data-r="${k}"]`, root);
    const tabs = R("tabs"), ckptBar = R("ckpt"), meta = R("meta"), stageEl = R("stage"), video = R("video");
    const scan = R("scan"), badge = R("badge"), stamp = R("stamp"), sound = R("sound");
    const lanes = R("lanes"), axis = R("axis"), head = R("head"), legend = R("legend");
    const titleEl = R("title"), log = R("log"), status = R("status"), stats = R("stats");

    let lang = "en";
    let runId = 0;
    let current = 0;
    let ck = 0;
    let skip = false;
    let win = null;
    const seenCk = CASES.map(() => new Set());

    const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    const t = (k) => UI[lang][k];
    const roleName = (r) => (ROLES[r] ? ROLES[r][lang] : r);
    const fmt = (s) => `${Number(s).toFixed(1)}s`;
    const scrollEnd = () => { log.scrollTop = log.scrollHeight; };

    function winLabel(w) {
      if (w.type === "point") return `@ ${fmt(w.at)}`;
      if (w.type === "whole") return t("whole");
      return `${fmt(w.start)} – ${fmt(w.end)}`;
    }
    function verdictClass(v) {
      if (ALERT.has(v)) return "v-alert";
      if (OK.has(v)) return "v-ok";
      if (v === "profile_complete") return "v-info";
      return "v-warn";
    }
    function verdictText(r) {
      if (lang === "zh") return `${VERDICT_ZH[r.verdict] || r.verdict}${r.confidence ? " · " + (CONF_ZH[r.confidence] || r.confidence) : ""}`;
      return `${r.verdict.replace(/_/g, " ")}${r.confidence ? " · " + r.confidence : ""}`;
    }
    const stepsOf = (c) => (c.checkpoints ? c.checkpoints[ck].steps : c.steps);

    const groups = [];
    CASES.forEach((c, i) => {
      const key = opts.group ? c.role : i;
      let g = groups.find((x) => x.key === key);
      if (!g) groups.push((g = { key, items: [] }));
      g.items.push(i);
    });
    const groupOf = (i) => groups.find((g) => g.items.includes(i));
    groups.forEach((g) => {
      const b = document.createElement("button");
      b.className = "demo-tab";
      b.type = "button";
      b.role = "tab";
      b.style.setProperty("--c", ROLES[CASES[g.items[0]].role].color);
      b.addEventListener("click", () => play(g.items.includes(current) ? current : g.items[0], { ck: 0, auto: true }));
      tabs.appendChild(b);
    });
    function renderTabs() {
      $$(".demo-tab", tabs).forEach((b, k) => {
        const g = groups[k], c = CASES[g.items[0]];
        const count = g.items.length > 1 ? `<span class="tab-count">${g.items.length}</span>` : "";
        b.innerHTML = `<span class="tab-dot"></span>${esc(opts.tabLabel ? opts.tabLabel(c, roleName) : roleName(c.role))}${count}`;
        b.classList.toggle("active", g.items.includes(current));
      });
    }
    function caseSwitch() {
      const g = groupOf(current);
      if (g.items.length < 2) return "";
      return `<div class="case-switch" role="group">${g.items.map((i, n) =>
        `<button type="button" class="cs-btn${i === current ? " active" : ""}" data-i="${i}"><img src="${esc(CASES[i].poster)}" alt="" />${t("case")} ${n + 1}</button>`).join("")}</div>`;
    }
    function renderCkpt() {
      if (!ckptBar) return;
      const c = CASES[current];
      ckptBar.innerHTML = `<span class="ck-label">Checkpoint</span>` + c.checkpoints.map((p, k) => {
        const seen = seenCk[current].has(k);
        const ok = p.answer === c.label;
        const ans = seen ? `<span class="ck-ans ${ok ? "ok" : "bad"}">${esc(p.answer.toUpperCase())} ${ok ? "✓" : "✗"}</span>` : `<span class="ck-ans">· · ·</span>`;
        return `${k ? '<i class="ck-line"></i>' : ""}<button class="ck ${k === ck ? "active" : ""}" type="button" data-k="${k}"><b>${esc(p.stage)}</b>${ans}</button>`;
      }).join("");
      $$(".ck", ckptBar).forEach((b) => b.addEventListener("click", () => play(current, { ck: +b.dataset.k })));
    }

    function focusWindow(w) {
      win = w;
      if (!video.duration) return;
      if (w.type === "point") {
        video.pause();
        video.currentTime = Math.min(w.at, video.duration - 0.05);
      } else {
        video.currentTime = w.start;
        video.play().catch(() => {});
      }
    }
    video.addEventListener("timeupdate", () => {
      if (!win || win.type === "point") return;
      if (video.currentTime >= Math.min(win.end, video.duration) - 0.06) {
        video.currentTime = win.start;
        video.play().catch(() => {});
      }
    });
    video.addEventListener("ended", () => {
      video.currentTime = win && win.type !== "point" ? win.start : 0;
      video.play().catch(() => {});
    });
    (function trackHead() {
      if (video.duration) head.style.left = `${(video.currentTime / video.duration) * 100}%`;
      requestAnimationFrame(trackHead);
    })();
    sound.addEventListener("click", () => {
      video.muted = !video.muted;
      sound.classList.toggle("on", !video.muted);
    });

    function buildAxis(dur) {
      axis.innerHTML = "";
      const step = dur > 6 ? 2 : dur < 1.5 ? 0.25 : 1;
      for (let s = 0; s <= dur + 1e-6; s += step) {
        const tick = document.createElement("span");
        tick.style.left = `${(s / dur) * 100}%`;
        tick.textContent = `${+s.toFixed(2)}s`;
        axis.appendChild(tick);
      }
    }
    function addLane(skill, w, dur) {
      const lane = document.createElement("div");
      lane.className = "vt-lane";
      lane.style.setProperty("--c", ROLES[skill].color);
      if (w.type === "point") {
        lane.innerHTML = `<i class="vt-pt" style="left:${(w.at / dur) * 100}%"></i>`;
      } else {
        const a = (w.start / dur) * 100;
        const b = (Math.min(w.end, dur) / dur) * 100;
        lane.innerHTML = `<i class="vt-seg" style="left:${a}%;width:${Math.max(1.5, b - a)}%"></i>`;
      }
      lane.title = `${roleName(skill)} · ${winLabel(w)}`;
      lanes.appendChild(lane);
      return lane;
    }
    let seen = [];
    function setLegend() {
      const counts = {};
      seen.forEach(({ skill, verdict }) => {
        counts[skill] = counts[skill] || { alert: 0 };
        if (ALERT.has(verdict)) counts[skill].alert++;
      });
      legend.innerHTML = Object.keys(counts)
        .map((k) => `<span><i style="background:${ROLES[k].color}"></i>${esc(roleName(k))}${counts[k].alert ? `<b class="alert">!</b>` : ""}</span>`)
        .join("");
    }
    function showBadge(skill, w, label) {
      badge.style.setProperty("--c", ROLES[skill].color);
      badge.innerHTML = `<i></i>${esc(label)} · ${esc(roleName(skill))} · ${esc(winLabel(w))}`;
      badge.classList.add("show");
    }

    async function typeInto(el, text, id) {
      if (skip || reduceMotion) { el.textContent = text; return; }
      const parts = lang === "zh" ? Array.from(text) : text.split(/(\s+)/);
      const per = Math.max(1, Math.ceil(parts.length / 70));
      let acc = "";
      for (let i = 0; i < parts.length; i += per) {
        if (id !== runId) return;
        if (skip) { el.textContent = text; return; }
        acc += parts.slice(i, i + per).join("");
        el.textContent = acc;
        scrollEnd();
        await sleep(24);
      }
    }
    const wait = (ms) => (skip || reduceMotion ? Promise.resolve() : sleep(ms));

    function clampToggle(card) {
      const more = $(".rep-more", card);
      const txt = $(".rep-text", card);
      if (!more) return;
      more.addEventListener("click", () => {
        const open = txt.classList.toggle("open");
        more.textContent = open ? t("less") : t("more");
      });
      requestAnimationFrame(() => { if (txt.scrollHeight <= txt.clientHeight + 2) more.remove(); });
    }
    function reportCard(skill, w, r, marks) {
      const body = r[lang];
      const card = document.createElement("div");
      card.className = `report ${verdictClass(r.verdict)}`;
      card.style.setProperty("--c", ROLES[skill].color);
      let text = body.text;
      if (body.kind === "raw") text = text.replace(/;\s*(SPEAKER_)/g, ";\n$1");
      let html = esc(text);
      if (marks) {
        card.classList.add("focus");
        marks.forEach((m) => { html = html.split(esc(m)).join(`<mark>${esc(m)}</mark>`); });
      }
      const issues = body.issues && body.issues.length
        ? `<ul class="rep-issues">${body.issues.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : "";
      card.innerHTML = `
        <div class="rep-head">
          <span class="rep-role"><i></i>${esc(roleName(skill))}</span>
          <span class="rep-win">${esc(winLabel(w))}</span>
          <span class="rep-verdict">${esc(verdictText(r))}</span>
        </div>
        ${issues}
        <div class="rep-text ${body.kind === "raw" ? "raw" : ""} ${marks ? "open" : ""}">${html}</div>
        ${marks ? "" : `<button class="rep-more" type="button">${t("more")}</button>`}`;
      clampToggle(card);
      return card;
    }
    function evidenceCard(it) {
      const card = document.createElement("div");
      card.className = "report v-alert evidence";
      card.style.setProperty("--c", ROLES[it.skill].color);
      card.innerHTML = `
        <div class="rep-head">
          <span class="rep-role"><i></i>${esc(roleName(it.skill))}</span>
          <span class="rep-win">${esc(winLabel(it.window))}</span>
          <span class="rep-verdict">${t("annotation")} · ${esc(it.confidence)}</span>
        </div>
        <div class="rep-text open">${esc(it.text)}</div>`;
      return card;
    }
    function phase(tag, sub) {
      const el = document.createElement("div");
      el.className = "phase";
      el.innerHTML = `<span>${esc(tag)}</span>${sub ? `<em>${esc(sub)}</em>` : ""}`;
      log.appendChild(el);
      return el;
    }

    async function runTurn(c, step, turnNo, id, counter) {
      const turn = document.createElement("div");
      turn.className = "turn";
      const label = lang === "zh" ? `${t("turn")} ${turnNo} 轮` : `${t("turn")} ${turnNo}`;
      turn.innerHTML = `<div class="turn-head"><span class="turn-no">${label}</span><span class="turn-who">${t("investigator")}</span></div><p class="turn-reason"></p>`;
      log.appendChild(turn);
      status.textContent = t("thinking") + "…";
      await typeInto($(".turn-reason", turn), step.reasoning[lang], id);
      if (id !== runId) return false;
      await wait(250);

      if (step.answer) {
        const correct = step.answer === c.label;
        const fin = document.createElement("div");
        fin.className = `final ${correct ? "ok" : "bad"}`;
        fin.innerHTML = `<span class="fin-k">${t("verdict")}</span><span class="fin-v">${esc(step.answer.toUpperCase())}</span><span class="fin-gt">${t("gt")}: ${esc(c.label)} ${correct ? "✓" : "✗"}</span>`;
        turn.appendChild(fin);
        stamp.textContent = step.answer.toUpperCase();
        stamp.classList.toggle("real", step.answer === "real");
        stamp.classList.add("show");
        badge.classList.remove("show");
        focusWindow({ type: "whole", start: 0, end: c.duration });
        scrollEnd();
        return true;
      }

      const color = ROLES[step.skill].color;
      const act = document.createElement("div");
      act.className = "act";
      act.style.setProperty("--c", color);
      act.innerHTML = `<span class="act-fn">${t("call")}</span><span class="act-role">${esc(step.skill)}</span><span class="act-win">${esc(winLabel(step.window))}</span>`;
      turn.appendChild(act);
      counter.calls++;

      const lane = addLane(step.skill, step.window, c.duration);
      lane.classList.add("live");
      focusWindow(step.window);
      showBadge(step.skill, step.window, t("inspectNow"));
      scan.classList.add("active");
      status.textContent = t("inspecting") + "…";

      const pending = document.createElement("div");
      pending.className = "pending";
      pending.innerHTML = `<span class="spin"></span>${esc(roleName(step.skill))} ${t("inspecting")}`;
      turn.appendChild(pending);
      scrollEnd();
      await wait(1100);
      if (id !== runId) return false;
      pending.remove();

      const marks = c.focus === step.skill ? c.marks : null;
      turn.appendChild(reportCard(step.skill, step.window, step.report, marks));
      if (ALERT.has(step.report.verdict)) counter.alert++;
      lane.classList.remove("live");
      lane.classList.add(verdictClass(step.report.verdict));
      scan.classList.remove("active");
      seen.push({ skill: step.skill, verdict: step.report.verdict });
      setLegend();
      scrollEnd();
      await wait(900);
      return false;
    }

    async function runNote(step, id) {
      const el = document.createElement("div");
      el.className = "note";
      const chips = (step.chips || [])
        .map((x) => `<span class="note-chip" style="--c:${ROLES[x.skill].color}">${esc(x.skill)} <em>${esc(winLabel(x.window))}</em></span>`)
        .join("");
      el.innerHTML = `
        <div class="note-head"><span class="note-tag">${esc(step.tag)}</span>${step.pill ? `<span class="note-pill">${esc(step.pill)}</span>` : ""}</div>
        ${step.sub ? `<div class="note-sub">${esc(step.sub)}</div>` : ""}
        <p class="note-text"></p>
        ${chips ? `<div class="note-chips">${chips}</div>` : ""}`;
      log.appendChild(el);
      status.textContent = t("thinking") + "…";
      await typeInto($(".note-text", el), step.text, id);
      scrollEnd();
      await wait(500);
    }

    async function runParallel(c, step, id, counter) {
      phase(step.tag, step.sub);
      const live = step.items.map((it) => {
        const lane = addLane(it.skill, it.window, c.duration);
        lane.classList.add("live");
        return lane;
      });
      focusWindow({ type: "whole", start: 0, end: c.duration });
      scan.classList.add("active");
      badge.style.setProperty("--c", ROLES[c.role].color);
      badge.innerHTML = `<i></i>${step.items.length} ${t("running")}`;
      badge.classList.add("show");
      const pending = document.createElement("div");
      pending.className = "pending";
      pending.innerHTML = `<span class="spin"></span>${step.items.map((it) => esc(roleName(it.skill))).join(" · ")}`;
      log.appendChild(pending);
      status.textContent = t("inspecting") + "…";
      scrollEnd();
      await wait(1400);
      if (id !== runId) return;
      pending.remove();
      step.items.forEach((it, k) => {
        log.appendChild(reportCard(it.skill, it.window, it.report));
        live[k].classList.remove("live");
        live[k].classList.add(verdictClass(it.report.verdict));
        seen.push({ skill: it.skill, verdict: it.report.verdict });
        counter.calls++;
        if (ALERT.has(it.report.verdict)) counter.alert++;
      });
      scan.classList.remove("active");
      badge.classList.remove("show");
      setLegend();
      scrollEnd();
      await wait(900);
    }

    async function runEvidence(c, step, id, counter) {
      phase(step.tag);
      for (const it of step.items) {
        if (id !== runId) return;
        const lane = addLane(it.skill, it.window, c.duration);
        lane.classList.add("live");
        focusWindow(it.window);
        showBadge(it.skill, it.window, t("annotated"));
        scan.classList.add("active");
        await wait(700);
        if (id !== runId) return;
        log.appendChild(evidenceCard(it));
        lane.classList.remove("live");
        lane.classList.add("v-alert");
        scan.classList.remove("active");
        seen.push({ skill: it.skill, verdict: "issue" });
        counter.intervals++;
        setLegend();
        scrollEnd();
        await wait(1100);
      }
      badge.classList.remove("show");
      focusWindow({ type: "whole", start: 0, end: c.duration });
    }

    async function play(i, o = {}) {
      const id = ++runId;
      skip = !!o.instant;
      current = i;
      if (o.ck !== undefined) ck = o.ck;
      const c = CASES[i];
      renderTabs();
      renderCkpt();
      titleEl.textContent = typeof opts.title === "function" ? opts.title(c, ck) : opts.title;
      log.innerHTML = "";
      lanes.innerHTML = "";
      legend.innerHTML = "";
      stats.textContent = "";
      seen = [];
      badge.classList.remove("show");
      stamp.classList.remove("show");
      win = { type: "whole", start: 0, end: c.duration };

      meta.innerHTML = `${caseSwitch()}
        <span>${c.duration.toFixed(1)}s</span>
        <span>${c.audio ? "♪ " + t("audio") : t("noAudio")}</span>`;
      $$(".cs-btn", meta).forEach((b) => b.addEventListener("click", () => {
        if (+b.dataset.i !== current) play(+b.dataset.i, { ck: 0, auto: true });
      }));
      buildAxis(c.duration);

      if (video.getAttribute("src") !== c.video) {
        stageEl.classList.add("loading");
        video.poster = c.poster;
        video.src = c.video;
        await new Promise((res) => {
          const done = () => { video.removeEventListener("loadedmetadata", done); res(); };
          video.addEventListener("loadedmetadata", done);
          setTimeout(done, 4000);
        });
        stageEl.classList.remove("loading");
      }
      if (id !== runId) return;
      video.currentTime = 0;
      video.play().catch(() => {});
      root.classList.add("running");

      const counter = { calls: 0, alert: 0, intervals: 0 };
      let turnNo = 0;
      for (const step of stepsOf(c)) {
        if (id !== runId) return;
        if (step.kind === "note") await runNote(step, id);
        else if (step.kind === "parallel") await runParallel(c, step, id, counter);
        else if (step.kind === "evidence") await runEvidence(c, step, id, counter);
        else if (step.kind === "gate") {
          const g = document.createElement("div");
          g.className = `gate ${step.ok ? "ok" : "bad"}`;
          g.innerHTML = `<b>${step.ok ? "✓" : "✗"} ${esc(step.tag)}</b><span>${esc(step.text)}</span>`;
          log.appendChild(g);
          scrollEnd();
          await wait(900);
        } else if (step.kind === "divider") {
          const d = document.createElement("div");
          d.className = "divider";
          d.innerHTML = `<span>${esc(step.text)}</span>`;
          log.appendChild(d);
          lanes.innerHTML = "";
          seen = [];
          setLegend();
          counter.calls = counter.alert = 0;
          scrollEnd();
          await wait(700);
        } else if (await runTurn(c, step, ++turnNo, id, counter)) break;
      }
      if (id !== runId) return;
      scan.classList.remove("active");
      root.classList.remove("running");
      status.textContent = t("done");
      stats.innerHTML = counter.intervals
        ? `<b>${counter.intervals}</b> ${t("intervals")}`
        : `<b>${counter.calls}</b> ${t("calls")} · <b>${counter.alert}</b> ${t("issues")}`;
      if (c.checkpoints) {
        seenCk[i].add(ck);
        renderCkpt();
        if (o.auto && ck < c.checkpoints.length - 1 && !skip) {
          await sleep(2600);
          if (id === runId) play(i, { ck: ck + 1, auto: true });
        }
      }
    }

    R("replay").addEventListener("click", () => play(current));
    R("skip").addEventListener("click", () => { skip = true; });
    if (opts.langToggle) {
      $$("button", R("lang")).forEach((b) => {
        b.addEventListener("click", () => {
          if (b.dataset.lang === lang) return;
          lang = b.dataset.lang;
          $$("button", R("lang")).forEach((x) => x.classList.toggle("active", x === b));
          play(current, { instant: true });
        });
      });
    }

    renderTabs();
    let started = false;
    new IntersectionObserver((es) => {
      if (es[0].isIntersecting && !started) {
        started = true;
        play(0, { ck: 0, auto: true });
      }
    }, { threshold: 0.3 }).observe(root);
  }

  Rollout($('[data-rollout="demo"]'), window.CASES, { langToggle: true, group: true, title: "investigator.rollout()" });
  Rollout($('[data-rollout="evo"]'), window.EVO_CASES, {
    checkpoints: true,
    title: (c, k) => `investigator.rollout(checkpoint="${c.checkpoints[k].stage.toLowerCase()}")`,
    tabLabel: (c) => c.title
  });
  Rollout($('[data-rollout="data"]'), window.DATA_CASES, {
    title: (c) => ({ sft: "forencue30k.trajectory()", rollout: "forencue30k.rl_pool().rollout()" }[c.kind] || "forencue30k.rl_pool()"),
    tabLabel: (c, roleName) => `${roleName(c.role)} · ${c.title}`
  });
})();
