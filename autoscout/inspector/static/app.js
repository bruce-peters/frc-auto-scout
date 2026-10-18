// auto-scout match inspector
"use strict";

// team colours: three distinct hues per alliance, all readable over video and on near-black
const RED_COLS = ["#f2471c", "#ff9a3c", "#ff7a9c"];
const BLUE_COLS = ["#7ea6d4", "#5cc4ae", "#d6dce4"];
const FRAME_W = 1920, FRAME_H = 1080;
// design tokens from style.css, for the canvases
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const C = { ground: css("--ground"), surface: css("--s1"), surface2: css("--s2"), text: css("--ink"),
            text2: css("--ink-2"), text3: css("--ink-3"), accent: css("--accent"), red: css("--red"), blue: css("--blue"),
            neg: css("--neg"), plateInk: css("--plate-ink"), font: css("--font"), display: css("--display") };
const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
// entrance: the parts of a region rise into place in reading order, once (never hidden if this doesn't run)
function enter(els, { y = 20, stagger = 70, dur = 720, delay = 0 } = {}) {
  if (REDUCED || !Element.prototype.animate) return;
  [...els].filter(Boolean).forEach((e, i) => e.animate(
    [{ opacity: 0, transform: `translateY(${y}px)` }, { opacity: 1, transform: "none" }],
    { duration: dur, delay: delay + i * stagger, easing: "cubic-bezier(0.22, 1, 0.36, 1)", fill: "backwards" }));
}
// a team number as it is printed on the robot's bumper, in that robot's tracking colour
const plate = (t, col) => `<span class="plate" style="--c:${col || "#9a9a9a"}">${t}</span>`;
const teamCol = (m, t) => { for (const al of ["red", "blue"]) { const i = (m[al] || []).indexOf(t); if (i >= 0) return (al === "red" ? RED_COLS : BLUE_COLS)[i % 3]; } return null; };
const ICON = {
  play: `<svg viewBox="0 0 16 16"><path d="M4.5 2.8v10.4c0 .6.6.9 1.1.6l8-5.2c.5-.3.5-.9 0-1.2l-8-5.2c-.5-.3-1.1 0-1.1.6z"/></svg>`,
  pause: `<svg viewBox="0 0 16 16"><rect x="3.5" y="2.5" width="3" height="11" rx="1"/><rect x="9.5" y="2.5" width="3" height="11" rx="1"/></svg>`,
};
const EVENT_NAMES = { "2026casnv": "Silicon Valley", "2026cagle": "Glendale", "2026cancmp": "NorCal State Champs",
                      "2026joh": "Johnson Division", "2026cc": "Chezy Champs" };
const LEVEL = { qm: ["Qualification", "Qual"], ef: ["Eighth-final", "EF"], qf: ["Quarterfinal", "QF"], sf: ["Playoff", "Playoff"], f: ["Final", "Final"] };
function matchName(key, short = false) {
  const m = /^(qm|ef|qf|sf|f)(\d+)(?:m(\d+))?$/.exec(key.split("_")[1] || ""); if (!m) return key;
  const num = m[1] === "f" ? m[3] : m[2];  // finals are f1m1, f1m2…; double-elim playoffs are sfNm1
  return `${LEVEL[m[1]][short ? 1 : 0]} ${num}`;
}
const eventName = (ev) => EVENT_NAMES[ev] || ev;
function hatch(ctx, dpr) {  // the diagonal hatching used for "inactive" / empty areas
  const t = document.createElement("canvas"), s = Math.round(7 * dpr); t.width = t.height = s;
  const g = t.getContext("2d"); g.strokeStyle = "rgba(255,255,255,.09)"; g.lineWidth = Math.max(1, dpr);
  g.beginPath(); g.moveTo(0, s); g.lineTo(s, 0); g.moveTo(-1, 1); g.lineTo(1, -1); g.moveTo(s - 1, s + 1); g.lineTo(s + 1, s - 1); g.stroke();
  return ctx.createPattern(t, "repeat");
}

const S = {
  matches: [], D: null, key: null,
  color: {}, alliance: {},
  detTimes: [], detGroups: new Map(),
  counts: { red: [], blue: [] },
  layers: { boxes: true, unassigned: true, shots: true, credit: true, counters: true, segids: false, views: false, solo: false },
  sel: { det: null, seg: null, team: null },
  ocr: null, ocrLoading: false,
  contradicted: new Set(),
  sub: "teams", segFilter: false,
  label: { on: false, name: "", bursts: [], teams: { red: [], blue: [] }, i: 0, labels: {}, cur: null,
           stopAt: null, prevLayers: null, saveTimer: null, status: "" },
};
try { S.label.name = localStorage.getItem("as_labeller") || ""; } catch (e) { /* storage blocked */ }

const $ = (s) => document.querySelector(s);
const el = (tag, attrs = {}, html = "") => { const e = document.createElement(tag); Object.assign(e, attrs); if (html) e.innerHTML = html; return e; };
const fmt = (x, d = 0) => (x === null || x === undefined || Number.isNaN(x)) ? "–" : Number(x).toFixed(d);

// ---------------------------------------------------------------- time helpers
function toMT(vt) {
  const tl = S.D.result.timeline; const a0 = tl.auto_start ?? 6.13, ts = tl.teleop_start;
  return (ts && vt >= ts) ? 20 + vt - ts : vt - a0;
}
function toVT(mt) {
  const tl = S.D.result.timeline; const a0 = tl.auto_start ?? 6.13, ts = tl.teleop_start;
  return (ts && mt >= 20) ? ts + mt - 20 : a0 + mt;
}
function periodOf(mt) {
  if (mt < 0) return "pre-match"; if (mt < 20) return "auto"; if (mt < 30) return "transition";
  if (mt < 55) return "shift 1"; if (mt < 80) return "shift 2"; if (mt < 105) return "shift 3";
  if (mt < 130) return "shift 4"; if (mt <= 160) return "endgame"; return "post-match";
}
function bsearch(arr, x) { let lo = 0, hi = arr.length - 1; while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < x) lo = m + 1; else hi = m; } return lo; }
function nearestIdx(arr, x) {
  if (!arr.length) return -1; let i = bsearch(arr, x);
  if (i > 0 && Math.abs(arr[i - 1] - x) < Math.abs(arr[i] - x)) i--; return i;
}
function countAt(al, vt) { let v = 0; for (const [t, c] of S.counts[al]) { if (t > vt) break; v = c; } return v; }

// ---------------------------------------------------------------- match list
async function loadMatches() {
  S.matches = await (await fetch("/api/matches")).json();
  const evs = [...new Set(S.matches.map((m) => m.event))].sort();
  for (const e of evs) $("#event-filter").append(el("option", { value: e, textContent: eventName(e) }));
  renderList();
  enter(document.querySelectorAll("#match-list > *"), { y: 14, stagger: 28, dur: 560 });
  const h = new URLSearchParams(location.hash.slice(1));
  if (h.get("match")) openMatch(h.get("match"), parseFloat(h.get("t") || "NaN"));
  else renderEmpty();
}

const ORDER = { qm: 0, ef: 1, qf: 2, sf: 3, f: 4 };
const isIssue = (m) => m.flags.some((f) => f.level === "bad");

function renderList() {
  const q = $("#filter").value.trim().toLowerCase();
  const ev = $("#event-filter").value, flagged = $("#flagged-only").checked;
  const box = $("#match-list"); box.innerHTML = "";
  let lastEv = null;
  const rows = S.matches.filter((m) => !m.error).sort((a, b) =>
    a.event.localeCompare(b.event) || (ORDER[a.comp_level] ?? 9) - (ORDER[b.comp_level] ?? 9) ||
    a.key.localeCompare(b.key, undefined, { numeric: true }));
  for (const m of rows) {
    const teams = [...(m.red || []), ...(m.blue || [])].map(String);
    if (ev && m.event !== ev) continue;
    if (flagged && !isIssue(m)) continue;
    if (q && !(m.key.includes(q) || teams.some((t) => t.startsWith(q)) || eventName(m.event).toLowerCase().includes(q))) continue;
    if (m.event !== lastEv) { box.append(el("div", { className: "ev-head" }, `<b>${eventName(m.event)}</b><span>${m.event}</span>`)); lastEv = m.event; }
    const bad = m.flags.filter((f) => f.level === "bad").length, warn = m.flags.length - bad;
    const idp = m.identity == null ? null : Math.round(100 * m.identity);
    const stat = [];
    const lp = Math.max(0, ...Object.values(m.labels || {}));
    if (lp) stat.push(`<span class="lab" title="Shooter-labelled bursts (most by one person)">${lp}/${m.n_bursts}</span>`);
    if (idp != null) stat.push(`<span class="${idp < 75 ? "bad" : idp < 85 ? "warn" : ""}" title="Bumper reads that agree with the assigned teams">ID ${idp}%</span>`);
    if (bad) stat.push(`<span class="issue" title="${bad} issue${bad > 1 ? "s" : ""}, ${warn} note${warn === 1 ? "" : "s"}">${bad} issue${bad > 1 ? "s" : ""}</span>`);
    const open = m.key === S.key;
    const tl = (al) => (m[al] || []).map((t) => (String(t) === "5026" ? `<b>${t}</b>` : t)).join(" ");
    const body = open
      ? `<div class="m-plates">${[...(m.red || []), ...(m.blue || [])].map((t) => plate(t, teamCol(m, t))).join("")}</div>` +
        (m.flags.length ? `<div class="m-note">${m.flags[0].text}${m.flags.length > 1 ? `, and ${m.flags.length - 1} more` : ""}</div>` : "")
      : `<div class="m-teams"><span class="r">${tl("red")}</span><span class="b">${tl("blue")}</span></div>`;
    const it = el("div", { className: "m-item" + (open ? " active" : ""), tabIndex: 0, role: "button" },
      `<div class="m-top"><span class="m-key">${matchName(m.key, true)}</span><span class="m-stat">${stat.join("")}</span></div>${body}`);
    it.onclick = () => openMatch(m.key);
    it.onkeydown = (e) => { if (e.key === "Enter") openMatch(m.key); };
    box.append(it);
  }
}

// no match open: the whole dataset as a board, one row per event
function renderEmpty() {
  const E = $("#empty"); const ok = S.matches.filter((m) => !m.error);
  const evs = [...new Set(ok.map((m) => m.event))].sort((a, b) => ok.filter((m) => m.event === b).length - ok.filter((m) => m.event === a).length);
  const med = (xs) => { const s = xs.filter((x) => x != null).sort((a, b) => a - b); return s.length ? s[Math.floor(s.length / 2)] : null; };
  E.innerHTML = `<h1>${ok.length} matches from ${evs.length} events</h1>
    <p>Pick one to replay it with the robot boxes, launch arcs and fuel credits drawn over the broadcast.</p>
    <div class="ev-board">${evs.map((ev) => {
      const ms = ok.filter((m) => m.event === ev); const id = med(ms.map((m) => m.identity));
      const has5026 = (m) => [...(m.red || []), ...(m.blue || [])].includes(5026);
      const ours = ms.filter(has5026).length;
      const pic = ms.find((m) => m.has_video && has5026(m)) || ms.find((m) => m.has_video);
      return `<button class="ev-row" data-ev="${ev}">${pic ? `<img class="ev-pic" src="/frame/${pic.key}?t=75&w=320" alt="${eventName(ev)} broadcast, ${matchName(pic.key, true)}" width="320" height="180" onload="this.classList.add('in')">` : `<span class="ev-pic"></span>`}<span class="ev-name">${eventName(ev)}</span>
        <span class="ev-fact"><b>${ms.length}</b>matches</span><span class="ev-fact"><b>${ours}</b>with 5026</span>
        <span class="ev-fact"><b>${id == null ? "–" : Math.round(100 * id) + "%"}</b>median identity</span>
        <span class="ev-fact"><b>${ms.filter(isIssue).length}</b>with issues</span></button>`;
    }).join("")}</div>`;
  E.querySelectorAll(".ev-row").forEach((b) => (b.onclick = () => { $("#event-filter").value = b.dataset.ev; renderList(); $("#match-list").scrollTop = 0;
    if (window.matchMedia("(max-width: 860px)").matches) $("#sidebar").scrollIntoView({ behavior: REDUCED ? "auto" : "smooth" }); }));
  enter([E.querySelector("h1"), E.querySelector("p"), ...E.querySelectorAll(".ev-row")], { stagger: 80 });
}

// ---------------------------------------------------------------- open a match
async function openMatch(key, seekVT) {
  if (S.label.saveTimer || S.label.dirty) await flushSave();  // never lose labels when switching matches
  S.key = key; renderList();
  { // bring the open row into the list's view without scrolling the page
    const list = $("#match-list"), act = list.querySelector(".m-item.active");
    if (act && !window.matchMedia("(max-width: 860px)").matches) { const lr = list.getBoundingClientRect(), ar = act.getBoundingClientRect();
      if (ar.top < lr.top + 48 || ar.bottom > lr.bottom) list.scrollTop += ar.top - lr.top - lr.height / 3; } }
  $("#empty").hidden = true; $("#match").hidden = false;
  const D = await (await fetch(`/api/match/${key}`)).json();
  if (D.error) { $("#match-header").textContent = D.error; return; }
  S.D = D; S.ocr = null; S.sel = { det: null, seg: null, team: null };
  S.color = {}; S.alliance = {};
  ["red", "blue"].forEach((al) => (D.tba[al] || []).forEach((t, i) => {
    S.color[t] = (al === "red" ? RED_COLS : BLUE_COLS)[i % 3]; S.alliance[t] = al;
  }));
  // detections grouped by sample time
  S.detGroups = new Map();
  D.dets.forEach((r, i) => { const k = r[0]; if (!S.detGroups.has(k)) S.detGroups.set(k, []); S.detGroups.get(k).push(i); });
  S.detTimes = [...S.detGroups.keys()].sort((a, b) => a - b);
  S.counts = { red: D.result.timeline.counts?.red || [], blue: D.result.timeline.counts?.blue || [] };
  S.contradicted = new Set(D.identity_stats.contradicted.map((c) => `${c.alliance}:${c.seg}`));
  S.segIndex = {};
  for (const al of ["red", "blue"]) for (const [sid, s] of Object.entries(D.result.identity?.[al] || {})) S.segIndex[sid] = { ...s, alliance: al, id: +sid };
  // team stats derived client side
  S.teamStats = teamStats();
  S.credit = creditCurves();
  S.jumps = creditJumps(); S.feedAt = null; S.selJump = null;
  renderHeader(); renderSelection();
  const v = $("#video");
  v.src = `/video/${key}`;
  v.onloadedmetadata = () => {
    v.currentTime = Number.isFinite(seekVT) ? seekVT : Math.max(0, toVT(0) - 1);
    resize(); drawAll();
  };
  shutter(); renderFilmstrip();
  renderSub();
  enter([...$("#match-header").querySelectorAll(".mh-title > *, .board > *, .board-note")], { stagger: 70 });
  enter($("#match-header").querySelectorAll(".bar .fin"), { y: 0, stagger: 40, dur: 900, delay: 200 });
  enter([$("#controls"), $("#layers"), ...document.querySelectorAll("#side-col > .panel:not([hidden])"), $("#timeline-card"), $("#tables")],
        { stagger: 80, delay: 260 });
  history.replaceState(null, "", `#match=${key}`);
  if (S.label.on) await loadLabelData();
}

function teamStats() {
  const D = S.D, out = {};
  for (const al of ["red", "blue"]) for (const t of D.tba[al] || []) {
    const r = D.result.teams[String(t)] || {};
    let tracked = 0;
    for (const s of Object.values(D.result.identity?.[al] || {})) if (s.team === t) tracked += s.t1 - s.t0;
    const shots = D.result.shots.filter((s) => s.team === t).length;
    out[t] = { al, fuel: r.fuel ?? 0, video: r.fuel_video ?? r.fuel ?? 0, per: r.fuel_by_period || {}, shots, tracked, opr: D.opr[String(t)],
               tower: `${r.auto_tower ?? "–"} / ${r.endgame_tower ?? "–"}`, points: r.points ?? 0 };
  }
  for (const al of ["red", "blue"]) {
    const ts = (D.tba[al] || []); const tot = ts.reduce((a, t) => a + (out[t]?.fuel || 0), 0);
    const osum = ts.reduce((a, t) => a + (out[t]?.opr || 0), 0);
    ts.forEach((t) => { out[t].oprShare = osum > 0 ? tot * (out[t].opr || 0) / osum : null; });
  }
  return out;
}

// fuel credited to each team up to a match time, from the per-event credit log (scaled to the final total)
function creditCurves() {
  const out = {};
  for (const t of Object.keys(S.color).map(Number)) {
    const ev = (S.D.result.fuel_log || []).filter((r) => r[1] === t).map((r) => [r[0], r[2]]).sort((a, b) => a[0] - b[0]);
    const tot = ev.reduce((a, r) => a + r[1], 0); let c = 0;
    out[t] = { t: ev.map((r) => r[0]), cum: ev.map((r) => (c += r[1])), tot };
  }
  return out;
}
function creditFrac(t, mt) {
  const cv = S.credit?.[t]; if (!cv || !cv.tot) return mt >= 160 ? 1 : 0;
  let lo = 0, hi = cv.t.length;  // count of credits at or before mt
  while (lo < hi) { const m = (lo + hi) >> 1; if (cv.t[m] <= mt) lo = m + 1; else hi = m; }
  return lo === 0 ? 0 : cv.cum[lo - 1] / cv.tot;
}

// every scoreboard jump and how it was split: fuel_log rows sharing a stamp and an alliance. The stamp is
// the counter time - 2 s (attribution.split_increments); the split used launches 0.8-5.5 s before the counter
const LAG_LO = 0.8, LAG_HI = 5.5, POP_S = 1.8;
function creditJumps() {
  const D = S.D, by = new Map();
  for (const [mt, t, b] of D.result.fuel_log || []) {
    const al = S.alliance[t]; if (!al) continue;
    const k = `${al}:${mt}`;
    if (!by.has(k)) by.set(k, { mt, al, vt: toVT(mt) + 2, n: 0, split: [] });
    const j = by.get(k); j.n += b; j.split.push([t, b]);
  }
  const out = [...by.values()].sort((a, b) => a.vt - b.vt);
  for (const j of out) {
    j.split.sort((a, b) => b[1] - a[1]);
    j.launches = {};  // launches in the window the split looked at, per team ("null": not tied to a robot)
    for (const sh of D.result.shots) {
      if (sh.alliance !== j.al || sh.t < j.vt - LAG_HI || sh.t > j.vt - LAG_LO) continue;
      j.launches[sh.team] = (j.launches[sh.team] || 0) + 1;
    }
  }
  return out;
}
function jumpsBefore(vt) { let lo = 0, hi = S.jumps.length; while (lo < hi) { const m = (lo + hi) >> 1; if (S.jumps[m].vt <= vt) lo = m + 1; else hi = m; } return lo; }
const mclock = (mt) => { const t = Math.max(0, mt); return `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`; };

// the credit feed: the latest jumps up to the playhead, newest first; click one to replay its launch window
function updateCreditFeed() {
  const box = $("#credit-feed"); if (!box || !S.jumps) return;
  const vt = $("#video").currentTime, n = jumpsBefore(vt);
  const fresh = n > 0 && vt - S.jumps[n - 1].vt < POP_S;
  const key = `${n}:${fresh}:${S.selJump}`; if (S.feedAt === key) return; S.feedAt = key;
  if (!S.jumps.length) { box.innerHTML = `<p class="faint cf-empty">No scoreboard jumps were read in this match.</p>`; return; }
  const pin = S.selJump != null ? `<p class="cf-pin">Pinned: every launch in its window is drawn on the video. Click it again or press <kbd>Esc</kbd> to unpin.</p>${feedRow(S.jumps[S.selJump], S.selJump, " sel")}` : "";
  if (!n && !pin) { box.innerHTML = `<p class="faint cf-empty">Play the match: each time a scoreboard counter goes up, the jump and the robots it was credited to appear here. Click one to pin it and see every launch it was split by.</p>`; return; }
  const rows = S.jumps.slice(Math.max(0, n - 7), n).map((j, k) => [j, Math.max(0, n - 7) + k]).reverse()
    .map(([j, idx], i) => feedRow(j, idx, i === 0 && fresh ? " now" : ""));
  box.innerHTML = pin + rows.join("");
  box.querySelectorAll(".cf-row").forEach((b) => (b.onclick = () => pinJump(+b.dataset.idx)));
}
function pinJump(idx) {
  if (S.selJump === idx) { S.selJump = null; drawAll(); return; }
  S.selJump = idx; seekVT(S.jumps[idx].vt - LAG_HI - 0.5);
}
function feedRow(j, idx, cls) {
  const split = j.split.filter(([, b]) => b >= 0.05).map(([t, b]) =>
    `<span class="cf-team">${plate(t, S.color[t])}<b>${b.toFixed(1)}</b><small>${j.launches[t] ? `${j.launches[t]} launch${j.launches[t] > 1 ? "es" : ""}` : "no launch"}</small></span>`).join("");
  const nl = j.launches["null"];
  const unl = nl ? `<span class="cf-unl">${nl} launch${nl > 1 ? "es" : ""} not tied to a robot</span>` : "";
  return `<button class="cf-row ${j.al}${cls}" data-idx="${idx}" title="Pin: replay from the start of the launch window (${LAG_HI} to ${LAG_LO} s before the jump) with all its launches drawn">
    <span class="cf-head"><span class="cf-n">+${Math.round(j.n)}</span><span class="cf-al">${j.al}</span><span class="cf-t">${mclock(j.mt + 2)}</span></span>
    <span class="cf-split">${split}</span>${unl}</button>`;
}

// the header is the broadcast score bar: live counter at the playhead, clock between the alliances
function renderHeader() {
  const D = S.D, h = $("#match-header"); h.innerHTML = "";
  const idp = D.identity_stats.agreement;
  h.append(el("div", { className: "mh-title" }, `<h1>${eventName(D.tba.event)} ${matchName(D.key, true)}</h1>`));
  document.title = `${matchName(D.key, true)}, ${eventName(D.tba.event)} · auto-scout`;
  const board = el("div", { className: "board" });
  const side = (al) => {
    const meta = D.result.teams[`_${al}`] || {};
    const ts = D.tba[al] || [];
    const maxv = Math.max(1, ...ts.map((t) => Math.max(S.teamStats[t].fuel, S.teamStats[t].oprShare || 0)));
    const live = S.counts[al].length > 0;
    const box = el("div", { className: `side ${al}` });
    const score = el("div", { className: "sb-score", title: "Scoreboard FUEL counter read at the playhead, out of the official TBA hub count" },
      `<span class="sb-now" data-live="${al}">${live ? 0 : fmt(meta.tba_total)}</span>
       <span class="sb-of">${live ? `of ${fmt(meta.tba_total)} fuel` : "fuel, TBA"}</span>
       <span class="sb-src">${meta.source === "overlay" ? `scoreboard read ${fmt(meta.timeline_total)}` : `total from ${meta.source || "?"}`}</span>`);
    const rows = el("div", { className: "sb-teams" });
    for (const t of ts) {
      const st = S.teamStats[t];
      const row = el("div", { className: "trow", title: "Bar: fuel credited from video, filled up to the playhead. White tick: what the fuel-OPR split would predict." },
        `${plate(t, S.color[t])}
         <span class="bar" style="--c:${S.color[t]}"><i class="fin" style="width:${100 * st.fuel / maxv}%"></i><i class="now" data-team="${t}" style="width:${100 * st.fuel / maxv}%"></i>
         ${st.oprShare != null ? `<em style="left:${100 * st.oprShare / maxv}%"></em>` : ""}</span>
         <span class="tval" data-team="${t}" title="Credited up to the playhead, of ${fmt(st.fuel)} for the match">${fmt(st.fuel)}</span><span class="topr">OPR ${fmt(st.oprShare)}</span>`);
      row.onclick = () => selectTeam(t);
      rows.append(row);
    }
    if (al === "red") box.append(rows, score); else box.append(score, rows);
    return box;
  };
  board.append(side("red"), el("div", { className: "sb-clock" }, `<span class="sb-time" id="sb-time">0:00</span><span class="sb-period" id="sb-period">pre-match</span>`), side("blue"));
  h.append(board, el("p", { className: "board-note" },
    `<span>${D.production === "chezy" ? "Chezy" : "FIRST"} broadcast</span><span>${D.result.n_segments} track segments</span>
     <span>${idp == null ? "no bumper reads" : `${Math.round(100 * idp)}% of bumper reads agree with the assigned teams`}</span>`));
  updateScore();
}

function updateScore() {
  if (!S.D) return;
  const mt = toMT($("#video").currentTime);
  for (const al of ["red", "blue"]) {
    const n = document.querySelector(`[data-live="${al}"]`);
    if (n && S.counts[al].length) { const v = String(countAt(al, $("#video").currentTime)); if (n.textContent !== v) n.textContent = v; }
  }
  document.querySelectorAll(".bar .now[data-team]").forEach((b) => { b.style.transform = `scaleX(${creditFrac(+b.dataset.team, mt).toFixed(4)})`; });
  document.querySelectorAll(".tval[data-team]").forEach((n) => {  // fuel credited so far, rising with the playhead
    const t = +n.dataset.team, v = fmt(S.teamStats[t].fuel * creditFrac(t, mt)); if (n.textContent !== v) n.textContent = v; });
  const t = Math.max(0, mt), m = Math.floor(t / 60), s = Math.floor(t % 60);
  const time = $("#sb-time"); if (time) time.textContent = `${m}:${String(s).padStart(2, "0")}`;
  const per = $("#sb-period"); if (per) { const p = periodOf(mt); per.textContent = p.charAt(0).toUpperCase() + p.slice(1); }
}

function shutter() {
  const w = $("#video-wrap"); if (REDUCED) return;
  w.classList.remove("opening"); w.classList.add("shut"); void w.offsetWidth;
  w.classList.add("opening"); w.classList.remove("shut");
  setTimeout(() => w.classList.remove("opening"), 1600);
}

// a strip of real broadcast frames above the timeline, one per 16 s of match time; click to seek
const FILM_N = 10, FILM_DT = 160 / FILM_N;
function renderFilmstrip() {
  const f = $("#filmstrip"); f.innerHTML = "";
  const m = S.matches.find((x) => x.key === S.key);
  f.hidden = !(m && m.has_video); if (f.hidden) return;
  for (let i = 0; i < FILM_N; i++) {
    const mt = (i + 0.5) * FILM_DT, vt = toVT(mt);
    const b = el("button", { className: "film", title: `${mt.toFixed(0)} s, ${periodOf(mt)}` });
    const img = el("img", { alt: `Broadcast at match ${mt.toFixed(0)} s (${periodOf(mt)})`, width: 240, height: 135 });
    img.onload = () => img.classList.add("in");
    img.src = `/frame/${S.key}?t=${vt.toFixed(1)}&w=240`;
    b.append(img); b.onclick = () => seekVT(toVT(i * FILM_DT));
    f.append(b);
  }
  placeFilmstrip();
}
function placeFilmstrip() {
  const f = $("#filmstrip"), c = $("#timeline"); if (!f || f.hidden || !c.clientWidth) return;
  const w = c.clientWidth, x = (mt) => TL.x0 + (mt - TL.mt0) / (TL.mt1 - TL.mt0) * (w - TL.x0 - 8);
  f.style.marginLeft = `${x(0)}px`; f.style.width = `${x(160) - x(0)}px`;
}
function markFilm(mt) {
  const k = Math.floor(mt / FILM_DT);
  document.querySelectorAll("#filmstrip .film").forEach((b, i) => b.classList.toggle("cur", i === k));
}

function renderTabCounts() {
  const D = S.D, L = S.label; if (!D) return;
  const n = { teams: (D.tba.red || []).length + (D.tba.blue || []).length, segments: Object.keys(S.segIndex).length,
              shots: D.result.shots.length, flags: D.flags.length,
              labels: L.on && L.bursts.length ? `${L.bursts.filter((b) => L.labels[b.id]).length} of ${L.bursts.length}` : "off" };
  document.querySelectorAll("[data-count]").forEach((s) => (s.textContent = n[s.dataset.count]));
}

// ---------------------------------------------------------------- drawing
function resize() {
  const dpr = window.devicePixelRatio || 1;
  for (const id of ["#overlay", "#fieldmap", "#timeline"]) {
    const c = $(id); const r = c.getBoundingClientRect();
    let h = r.height;
    if (id === "#timeline") { h = 30 + 2 * (42 + 3 * 30 + 10) + 4; c.style.height = h + "px"; }
    c.width = Math.round(r.width * dpr); c.height = Math.round(h * dpr);
  }
  placeFilmstrip();
}

function drawAll() { if (!S.D) return; drawOverlay(); drawField(); drawTimeline(); updateClock(); updateScore(); updateCreditFeed(); drawZoom(); markFilm(toMT($("#video").currentTime)); }

function currentGroup() {
  const vt = $("#video").currentTime; const i = nearestIdx(S.detTimes, vt);
  if (i < 0 || Math.abs(S.detTimes[i] - vt) > 0.06) return [];
  return S.detGroups.get(S.detTimes[i]);
}

function visibleDet(r) {
  const team = r[7];
  if (team == null && !S.layers.unassigned) return false;
  if (S.layers.solo && S.sel.team != null && team !== S.sel.team) return false;
  return true;
}

function drawOverlay() {
  const c = $("#overlay"), ctx = c.getContext("2d"), D = S.D, v = $("#video");
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, c.width, c.height);
  const fw = v.videoWidth || FRAME_W, fh = v.videoHeight || FRAME_H;
  ctx.setTransform(c.width / fw, 0, 0, c.height / fh, 0, 0);
  const sx = fw / FRAME_W, sy = fh / FRAME_H;
  const vt = v.currentTime;
  const px = (n) => n * fw / Math.max(1, c.clientWidth);  // n screen pixels, in frame pixels
  if (S.layers.views) {
    ctx.lineWidth = 2; ctx.setLineDash([10, 8]); ctx.strokeStyle = "rgba(255,255,255,.55)"; ctx.font = `600 22px ${C.font}`;
    for (const vw of D.views) {
      const [x0, y0, x1, y1] = vw.box; ctx.strokeRect(x0 * sx + 2, y0 * sy + 2, (x1 - x0) * sx - 4, (y1 - y0) * sy - 4);
      ctx.fillStyle = "rgba(255,255,255,.8)"; ctx.fillText(vw.name, x0 * sx + 12, y1 * sy - 12);
    }
    ctx.setLineDash([]);
  }
  if (S.layers.counters) {
    for (const al of ["red", "blue"]) {
      const roi = D.overlay_rois[`${al}_cnt`]; if (!roi) continue;
      const [x0, y0, x1, y1] = roi; ctx.lineWidth = 4; ctx.strokeStyle = C.accent;
      ctx.strokeRect(x0 * sx - 4, y0 * sy - 4, (x1 - x0) * sx + 8, (y1 - y0) * sy + 8);
      label(ctx, `read ${countAt(al, vt)}`, x0 * sx - 4, y1 * sy + 4 + px(19), C.accent, px(13));
    }
  }
  if (S.layers.boxes) {
    for (const i of currentGroup()) {
      const r = D.dets[i]; if (!visibleDet(r)) continue;
      const [, , x0, y0, x1, y1, , team, sid] = r;
      const col = team != null ? S.color[team] : "#9a9a9a";
      const selected = S.sel.det === i || (S.sel.seg != null && sid === S.sel.seg) || (S.sel.team != null && team === S.sel.team && !S.layers.solo);
      const bad = sid != null && S.contradicted.has(`${r[6]}:${sid}`);
      ctx.lineWidth = selected ? 7 : 4; ctx.strokeStyle = col;
      if (team == null) ctx.setLineDash([12, 8]);
      ctx.strokeRect(x0 * sx, y0 * sy, (x1 - x0) * sx, (y1 - y0) * sy); ctx.setLineDash([]);
      if (bad) { ctx.lineWidth = 2; ctx.strokeStyle = C.neg; ctx.strokeRect(x0 * sx - 6, y0 * sy - 6, (x1 - x0) * sx + 12, (y1 - y0) * sy + 12); }
      let txt = team != null ? String(team) : "?";
      if (S.layers.segids && sid != null) txt += ` s${sid}`;
      label(ctx, txt, x0 * sx - (selected ? 3.5 : 2), y0 * sy - (selected ? 3.5 : 2), col, px(13));
    }
  }
  if (S.layers.shots) {
    const offs = Object.fromEntries(D.views.map((w) => [w.name, [w.box[0] * sx, w.box[1] * sy]]));
    ctx.lineCap = "round";
    for (const sh of D.result.shots) {
      if (sh.t > vt || vt > sh.t + 0.8) continue;
      if (S.layers.solo && S.sel.team != null && sh.team !== S.sel.team) continue;
      const [ox, oy] = offs[sh.view] || [0, 0];
      const pts = sh.pts.filter((p) => p[0] <= vt); if (pts.length < 2) continue;
      ctx.strokeStyle = sh.team != null ? S.color[sh.team] : "#ffffff"; ctx.lineWidth = 5;
      ctx.beginPath(); pts.forEach((p, k) => (k ? ctx.lineTo(p[1] + ox, p[2] + oy) : ctx.moveTo(p[1] + ox, p[2] + oy))); ctx.stroke();
      const last = pts[pts.length - 1]; ctx.fillStyle = ctx.strokeStyle; ctx.beginPath(); ctx.arc(last[1] + ox, last[2] + oy, 7, 0, 7); ctx.fill();
    }
  }
  if (S.layers.credit && S.jumps?.length) {
    // the launches a jump was split by: the pinned jump, else the latest one for POP_S after it. Every launch in
    // its window (LAG_HI to LAG_LO s before the counter moved) is drawn whole, in the colour of the robot it was
    // tied to (white, dashed: tied to none), with that robot's number where it left
    let fj = S.selJump != null ? S.jumps[S.selJump] : null;
    if (!fj) { const k = jumpsBefore(vt) - 1; if (k >= 0 && vt - S.jumps[k].vt < POP_S) fj = S.jumps[k]; }
    if (fj) {
      const offs = Object.fromEntries(D.views.map((w) => [w.name, [w.box[0] * sx, w.box[1] * sy]]));
      ctx.lineCap = "round";
      for (const sh of D.result.shots) {
        if (sh.alliance !== fj.al || sh.t < fj.vt - LAG_HI || sh.t > fj.vt - LAG_LO || sh.pts.length < 2) continue;
        if (S.layers.solo && S.sel.team != null && sh.team !== S.sel.team) continue;
        const [ox, oy] = offs[sh.view] || [0, 0], col = sh.team != null ? S.color[sh.team] : "#ffffff";
        ctx.strokeStyle = col; ctx.lineWidth = 3; ctx.globalAlpha = 0.9;
        if (sh.team == null) ctx.setLineDash([8, 6]);
        ctx.beginPath(); sh.pts.forEach((p, k) => (k ? ctx.lineTo(p[1] + ox, p[2] + oy) : ctx.moveTo(p[1] + ox, p[2] + oy))); ctx.stroke();
        ctx.setLineDash([]); ctx.globalAlpha = 1;
        const p0 = sh.pts[0]; ctx.fillStyle = col; ctx.beginPath(); ctx.arc(p0[1] + ox, p0[2] + oy, 5, 0, 7); ctx.fill();
        label(ctx, sh.team != null ? String(sh.team) : "?", p0[1] + ox + 6, p0[2] + oy - 4, col, px(9));
      }
    }
    // credit landing: for POP_S after a scoreboard jump, "+balls" rises above every box of each team it went to
    const grp = currentGroup(), n = jumpsBefore(vt), got = new Map();
    for (let k = n - 1; k >= 0 && vt - S.jumps[k].vt < POP_S; k--) {
      const j = S.jumps[k], age = (vt - j.vt) / POP_S;
      for (const [t, b] of j.split) if (b >= 0.3) { const g = got.get(t) || { b: 0, age }; g.b += b; g.age = Math.min(g.age, age); got.set(t, g); }
    }
    for (const [t, g] of got) {
      if (S.layers.solo && S.sel.team != null && t !== S.sel.team) continue;
      for (const i of grp) {
        const r = D.dets[i]; if (r[7] !== t) continue;
        const [, , x0, y0] = r;
        ctx.globalAlpha = Math.max(0, 1 - g.age * g.age);
        label(ctx, `+${g.b.toFixed(1)}`, x0 * sx, y0 * sy - px(19) - px(8) * g.age, S.color[t], px(10));
        ctx.globalAlpha = 1;
      }
    }
  }
}

// a bumper plate sitting on the box's top-left corner: the box colour as the fill, the number in near-black
function label(ctx, txt, x, y, col, size) {
  ctx.font = `700 ${size}px ${C.display}`; const w = ctx.measureText(txt).width;
  const ph = Math.round(size * 1.3), pw = w + size * 0.7;
  ctx.fillStyle = col; ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(x, y - ph, pw, ph, 4); else ctx.rect(x, y - ph, pw, ph);
  ctx.fill(); ctx.fillStyle = C.plateInk; ctx.textBaseline = "alphabetic";
  ctx.fillText(txt, x + size * 0.35, y - ph * 0.22);
}

// field map ------------------------------------------------------
function fieldXform(c) {
  const F = S.D.field, pad = 10 * (window.devicePixelRatio || 1);
  const s = Math.min((c.width - 2 * pad) / F.length, (c.height - 2 * pad) / F.width);
  const ox = (c.width - F.length * s) / 2, oy = (c.height - F.width * s) / 2;
  return { s, m2p: (x, y) => [ox + x * s, oy + (F.width - y) * s], p2m: (px, py) => [(px - ox) / s, F.width - (py - oy) / s] };
}
function posAt(team, mt) {
  const arr = S.D.result.paths?.[String(team)]; if (!arr || !arr.length) return null;
  const ts = arr._t || (arr._t = arr.map((p) => p[0]));
  const i = nearestIdx(ts, mt); if (i < 0 || Math.abs(ts[i] - mt) > 0.6) return null; return arr[i];
}
function drawField() {
  const c = $("#fieldmap"), ctx = c.getContext("2d"), F = S.D.field;
  const { s, m2p } = fieldXform(c); const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = C.surface; ctx.fillRect(0, 0, c.width, c.height);
  const [x0, y0] = m2p(0, F.width), [x1, y1] = m2p(F.length, 0);
  // the carpet is one tone up from the panel; the alliance zones are hatched like every inactive area
  ctx.fillStyle = "#1d1d1d"; ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
  ctx.fillStyle = "#171717";
  { const [za] = m2p(F.zone_depth, 0), [zb] = m2p(F.length - F.zone_depth, 0); ctx.fillRect(x0, y0, za - x0, y1 - y0); ctx.fillRect(zb, y0, x1 - zb, y1 - y0);
    ctx.fillStyle = hatch(ctx, dpr); ctx.fillRect(x0, y0, za - x0, y1 - y0); ctx.fillRect(zb, y0, x1 - zb, y1 - y0); }
  ctx.lineWidth = 2 * dpr;
  const vline = (x, col) => { const [a, b] = m2p(x, 0), [, d] = m2p(x, F.width); ctx.strokeStyle = col; ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(a, d); ctx.stroke(); };
  vline(F.length / 2, "rgba(255,255,255,.12)"); vline(F.zone_depth, C.blue); vline(F.length - F.zone_depth, C.red);
  for (const [hub, col] of [[F.blue_hub, C.blue], [F.red_hub, C.red]]) {
    const h = 0.597; const [a, b] = m2p(hub[0] - h, hub[1] + h); ctx.fillStyle = col; ctx.globalAlpha = 0.85;
    ctx.beginPath(); if (ctx.roundRect) ctx.roundRect(a, b, 2 * h * s, 2 * h * s, 2 * dpr); else ctx.rect(a, b, 2 * h * s, 2 * h * s); ctx.fill(); ctx.globalAlpha = 1;
  }
  const mt = toMT($("#video").currentTime);
  for (const t of Object.keys(S.color).map(Number)) {
    const arr = S.D.result.paths?.[String(t)]; if (!arr) continue;
    const col = S.color[t]; const dim = S.sel.team != null && S.sel.team !== t;
    ctx.globalAlpha = dim ? 0.3 : 1;
    const trail = arr.filter((p) => p[0] <= mt && p[0] >= mt - 3);
    if (trail.length > 1) { ctx.strokeStyle = col; ctx.lineWidth = 2 * dpr; ctx.beginPath(); trail.forEach((p, k) => { const [a, b] = m2p(p[1], p[2]); k ? ctx.lineTo(a, b) : ctx.moveTo(a, b); }); ctx.stroke(); }
    const p = posAt(t, mt);
    if (p) {
      const [a, b] = m2p(p[1], p[2]); ctx.fillStyle = col; ctx.beginPath(); ctx.arc(a, b, Math.max(5 * dpr, 0.42 * s), 0, 7); ctx.fill();
      if (S.sel.team === t) { ctx.strokeStyle = C.accent; ctx.lineWidth = 2.5 * dpr; ctx.stroke(); }
      ctx.font = `700 ${13 * dpr}px ${C.display}`; ctx.fillStyle = C.text; ctx.fillText(String(t), a + 9 * dpr, b - 6 * dpr);
    }
    ctx.globalAlpha = 1;
  }
}

// timeline -------------------------------------------------------
const TL = { x0: 76, mt0: -2, mt1: 168, rows: [] };
function tlX(c, mt) { const dpr = window.devicePixelRatio || 1; return TL.x0 * dpr + (mt - TL.mt0) / (TL.mt1 - TL.mt0) * (c.width - (TL.x0 + 8) * dpr); }
function tlMT(c, px) { const dpr = window.devicePixelRatio || 1; return TL.mt0 + (px - TL.x0 * dpr) / (c.width - (TL.x0 + 8) * dpr) * (TL.mt1 - TL.mt0); }
const SPANS = { auto: [0, 20], transition: [20, 30], shift1: [30, 55], shift2: [55, 80], shift3: [80, 105], shift4: [105, 130], endgame: [130, 160] };

function drawTimeline() {
  const c = $("#timeline"), ctx = c.getContext("2d"), D = S.D, dpr = window.devicePixelRatio || 1;
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, c.width, c.height);
  const font = (w, px, fam = C.font) => `${w} ${px * dpr}px ${fam}`;
  ctx.font = font(400, 13);
  TL.rows = []; let y = 30 * dpr;
  // periods are alternating bands down the whole chart, named in the top row; no grid lines
  const NAMES = { auto: "Auto", transition: "T", shift1: "Shift 1", shift2: "Shift 2", shift3: "Shift 3", shift4: "Shift 4", endgame: "Endgame" };
  Object.entries(SPANS).forEach(([name, [a, b]], k) => {
    const xa = tlX(c, a), xb = tlX(c, b);
    ctx.fillStyle = k % 2 ? "#161616" : "#101010"; ctx.fillRect(xa, 0, xb - xa, c.height);
    ctx.fillStyle = C.text3; ctx.fillText(NAMES[name], xa + 6 * dpr, 17 * dpr);
  });
  const hatchFill = hatch(ctx, dpr);
  const platePx = (t, col, yy, h) => {  // the team's bumper plate as the row label
    ctx.font = font(700, 13, C.display); ctx.fillStyle = col; ctx.beginPath();
    const w = (TL.x0 - 14) * dpr; if (ctx.roundRect) ctx.roundRect(2 * dpr, yy, w, h, 3 * dpr); else ctx.rect(2 * dpr, yy, w, h); ctx.fill();
    ctx.fillStyle = C.plateInk; ctx.textAlign = "center"; ctx.fillText(String(t), 2 * dpr + w / 2, yy + h * 0.72); ctx.textAlign = "start";
  };
  for (const al of ["red", "blue"]) {
    const base = al === "red" ? C.red : C.blue;
    const blockTop = y;
    // counter increments per second
    const h0 = 36 * dpr; const inc = new Map();
    let prev = 0; for (const [t, cnum] of S.counts[al]) { if (cnum > prev) { const k = Math.floor(toMT(t)); inc.set(k, (inc.get(k) || 0) + cnum - prev); } prev = Math.max(prev, cnum); }
    const mx = Math.max(4, ...inc.values());
    ctx.fillStyle = base;
    for (const [k, n] of inc) { const xa = tlX(c, k), xb = tlX(c, k + 1); const hh = (h0 - 4 * dpr) * n / mx; ctx.fillRect(xa, y + h0 - hh, Math.max(1, xb - xa - 1), hh); }
    ctx.font = font(600, 13); ctx.fillStyle = al === "red" ? css("--red-text") : css("--blue-text");
    ctx.fillText(al === "red" ? "Red" : "Blue", 4 * dpr, y + 16 * dpr);
    ctx.font = font(400, 13); ctx.fillStyle = C.text3; ctx.fillText("fuel/s", 4 * dpr, y + 32 * dpr);
    TL.rows.push({ kind: "counter", al, y0: y, y1: y + h0, inc });
    y += h0 + 6 * dpr;
    for (const t of D.tba[al] || []) {
      const lh = 26 * dpr, col = S.color[t];
      ctx.fillStyle = S.sel.team === t ? "rgba(255,203,31,.08)" : "rgba(255,255,255,.025)"; ctx.fillRect(TL.x0 * dpr, y, c.width - TL.x0 * dpr, lh);
      platePx(t, col, y + 2 * dpr, lh - 4 * dpr);
      // segments
      const segs = Object.values(S.segIndex).filter((s) => s.team === t);
      for (const s of segs) {
        const xa = tlX(c, toMT(s.t0)), xb = tlX(c, toMT(s.t1));
        ctx.fillStyle = col; ctx.globalAlpha = S.sel.seg === s.id ? 1 : 0.55;
        ctx.fillRect(xa, y + 9 * dpr, Math.max(2, xb - xa), 8 * dpr); ctx.globalAlpha = 1;
        if (S.contradicted.has(`${s.alliance}:${s.id}`)) { ctx.strokeStyle = C.neg; ctx.lineWidth = 2 * dpr; ctx.strokeRect(xa, y + 8 * dpr, Math.max(2, xb - xa), 10 * dpr); }
      }
      // launches
      ctx.strokeStyle = col; ctx.lineWidth = 1.5 * dpr;
      for (const sh of D.result.shots) if (sh.team === t) { const x = tlX(c, toMT(sh.t)); ctx.beginPath(); ctx.moveTo(x, y + 1 * dpr); ctx.lineTo(x, y + 7 * dpr); ctx.stroke(); }
      // credited fuel per second
      const per = new Map(); for (const [m, tm, n] of D.result.fuel_log || []) if (tm === t) { const k = Math.floor(m); per.set(k, (per.get(k) || 0) + n); }
      const pm = Math.max(2, ...per.values());
      ctx.fillStyle = col; ctx.globalAlpha = 0.9;
      for (const [k, n] of per) { const xa = tlX(c, k), xb = tlX(c, k + 1); const hh = 6 * dpr * n / pm + 1; ctx.fillRect(xa, y + lh - hh, Math.max(1, xb - xa - 1), hh); }
      ctx.globalAlpha = 1;
      TL.rows.push({ kind: "team", al, team: t, y0: y, y1: y + lh, segs, per });
      y += lh + 4 * dpr;
    }
    // inactive shifts for this alliance: dimmed and hatched
    for (const p of D.inactive?.[al] || []) {
      const sp = SPANS[p]; if (!sp) continue; const xa = tlX(c, sp[0]), xb = tlX(c, sp[1]), hh = y - blockTop - 4 * dpr;
      ctx.fillStyle = "rgba(10,10,10,.55)"; ctx.fillRect(xa, blockTop, xb - xa, hh);
      ctx.fillStyle = hatchFill; ctx.fillRect(xa, blockTop, xb - xa, hh);
    }
    y += 10 * dpr;
  }
  // playhead: a yellow line with the match time on a tab at the top
  const mt = toMT($("#video").currentTime), x = tlX(c, mt);
  ctx.strokeStyle = C.accent; ctx.lineWidth = 2 * dpr; ctx.beginPath(); ctx.moveTo(x, 24 * dpr); ctx.lineTo(x, c.height); ctx.stroke();
  const lab = `${mt.toFixed(1)} s`; ctx.font = font(600, 13); const tw = ctx.measureText(lab).width + 12 * dpr;
  const tx = Math.min(Math.max(x - tw / 2, TL.x0 * dpr), c.width - tw);
  ctx.fillStyle = C.accent; ctx.beginPath(); if (ctx.roundRect) ctx.roundRect(tx, 2 * dpr, tw, 22 * dpr, 3 * dpr); else ctx.rect(tx, 2 * dpr, tw, 22 * dpr); ctx.fill();
  ctx.fillStyle = css("--accent-ink"); ctx.fillText(lab, tx + 6 * dpr, 18 * dpr);
}

function updateClock() {
  const vt = $("#video").currentTime, mt = toMT(vt);
  $("#clock").innerHTML = `Video <b>${vt.toFixed(2)} s</b>`;
  const paused = $("#video").paused, pb = $("#play");
  if (pb.dataset.state !== String(paused)) { pb.dataset.state = String(paused); pb.innerHTML = paused ? ICON.play : ICON.pause; pb.setAttribute("aria-label", paused ? "Play" : "Pause"); }
}

// ---------------------------------------------------------------- selection
function selectDet(i) {
  const r = S.D.dets[i]; S.sel = { det: i, seg: r[8], team: r[7] }; renderSelection(); drawAll();
}
function selectTeam(t) { S.sel = { det: null, seg: null, team: t }; renderSelection(); drawAll(); }
function selectSeg(seg) { const s = S.segIndex[seg]; S.sel = { det: null, seg, team: s?.team ?? null }; renderSelection(); drawAll(); }

async function ensureOCR() {
  if (S.ocr || S.ocrLoading) return; S.ocrLoading = true;
  try { S.ocr = await (await fetch(`/api/match/${S.key}/ocr`)).json(); } catch (e) { S.ocr = {}; }
  S.ocrLoading = false; renderSelection();
}

function segInfo(seg) {
  const s = S.segIndex[seg]; if (!s) return "";
  const teams = S.D.tba[s.alliance] || [];
  const ev = s.ocr.map((x, k) => x + (s.held?.[k] || 0));
  const evs = teams.map((t, k) => `<span style="color:${S.color[t]}">${t}</span> ${fmt(ev[k], 1)}`).join(" · ");
  const bad = S.contradicted.has(`${s.alliance}:${seg}`);
  return `<div>Segment</div><div>s${seg} · slot ${s.slot} · ${fmt(toMT(s.t0), 1)}–${fmt(toMT(s.t1), 1)} s</div>
          <div>Bumper evidence</div><div>${evs || "none"}</div>
          <div>Labelled</div><div style="color:${bad ? C.neg : "inherit"}">${s.team ?? "?"}${bad ? " (contradicts its reads)" : ""}</div>`;
}

function renderSelection() {
  const b = $("#sel-body"); const { det, seg, team } = S.sel;
  if (det == null && seg == null && team == null) {
    // nothing picked yet: offer the six robots as plates
    b.innerHTML = `<p class="sel-empty">Click a box in the video, a dot on the field, or one of these robots.</p>
      <div class="m-plates">${["red", "blue"].flatMap((al) => (S.D?.tba[al] || []).map((t) => `<button class="pick" data-team="${t}" aria-label="Select ${t}">${plate(t, S.color[t])}</button>`)).join("")}</div>`;
    b.querySelectorAll(".pick").forEach((p) => (p.onclick = () => selectTeam(+p.dataset.team)));
    return;
  }
  let html = `<canvas id="zoom"></canvas><div class="kv">`;
  if (team != null) {
    const st = S.teamStats[team];
    html += `<div>Team</div><div>${plate(team, S.color[team])}${st.al}</div>
             <div>Fuel credited</div><div><b>${fmt(st.fuel)}</b> &nbsp;<span class="muted">OPR split ${fmt(st.oprShare)} · OPR ${fmt(st.opr)}</span></div>
             <div>Video only</div><div>${fmt(st.video)} &nbsp;<span class="muted">${Math.abs(st.fuel - st.video) >= 0.5 ? `the OPR pull moved it ${st.fuel > st.video ? "+" : "−"}${fmt(Math.abs(st.fuel - st.video))}` : "the OPR pull did not move it"}</span></div>
             <div>Launches seen</div><div>${st.shots}</div><div>Tracked</div><div>${fmt(st.tracked)} s</div>`;
  } else html += `<div>Team</div><div class="muted">unlabelled</div>`;
  if (seg != null) html += segInfo(seg);
  if (det != null) {
    const r = S.D.dets[det];
    html += `<div>Detection</div><div>${r[1]} view · t=${r[0].toFixed(2)} s</div>`;
    if (r[9] != null) html += `<div>Hopper yellow</div><div>${(100 * r[9]).toFixed(0)} %</div>`;
    if (S.ocr) {
      const rd = S.ocr[String(det)];
      html += `<div>Bumper OCR</div><div class="reads">${rd ? (rd.length ? rd.map(([t, c]) => `"${t}" ${c}`).join(", ") : "read, nothing found") : "not read (crop too small or not sampled)"}</div>`;
    } else { html += `<div>Bumper OCR</div><div class="muted">loading…</div>`; ensureOCR(); }
  }
  b.innerHTML = html + "</div>";
  drawZoom();
}

function drawZoom() {
  const z = document.getElementById("zoom"); if (!z) return;
  const ctx = z.getContext("2d"); const v = $("#video");
  z.width = z.clientWidth * (window.devicePixelRatio || 1); z.height = z.clientHeight * (window.devicePixelRatio || 1);
  ctx.fillStyle = "#0d0d0d"; ctx.fillRect(0, 0, z.width, z.height);
  // follow the selected segment / team in the current frame, else the selected detection
  let r = null;
  for (const i of currentGroup()) { const d = S.D.dets[i]; if ((S.sel.seg != null && d[8] === S.sel.seg) || (S.sel.seg == null && S.sel.team != null && d[7] === S.sel.team)) { if (!r || d[1] !== "main") r = d; } }
  if (!r && S.sel.det != null && Math.abs(S.D.dets[S.sel.det][0] - v.currentTime) < 0.3) r = S.D.dets[S.sel.det];
  if (!r) {
    const dpr = window.devicePixelRatio || 1; ctx.fillStyle = hatch(ctx, dpr); ctx.fillRect(0, 0, z.width, z.height);
    ctx.fillStyle = C.text2; ctx.font = `400 ${13 * dpr}px ${C.font}`; ctx.textAlign = "center";
    ctx.fillText("Not visible in this frame", z.width / 2, z.height / 2); ctx.textAlign = "start"; return;
  }
  const fx = (v.videoWidth || FRAME_W) / FRAME_W, fy = (v.videoHeight || FRAME_H) / FRAME_H;
  let [x0, y0, x1, y1] = [r[2] * fx, r[3] * fy, r[4] * fx, r[5] * fy];
  const w = x1 - x0, h = y1 - y0, pad = 0.25;
  x0 -= w * pad; x1 += w * pad; y0 -= h * pad; y1 += h * pad;
  const ar = z.width / z.height; let cw = x1 - x0, ch = y1 - y0;
  if (cw / ch > ar) { const nh = cw / ar; y0 -= (nh - ch) / 2; ch = nh; } else { const nw = ch * ar; x0 -= (nw - cw) / 2; cw = nw; }
  try { ctx.drawImage(v, x0, y0, cw, ch, 0, 0, z.width, z.height); } catch (e) { /* video not ready */ }
}

// ---------------------------------------------------------------- tables
function renderSub() {
  document.querySelectorAll(".sub").forEach((b) => { b.classList.toggle("active", b.dataset.sub === S.sub); b.setAttribute("aria-selected", b.dataset.sub === S.sub); });
  const body = $("#sub-body"); const D = S.D; if (!D) return;
  renderTabCounts();
  if (S.sub === "teams") {
    let h = `<table><thead><tr><th>Team</th><th>Alliance</th><th title="Best estimate: video split, pulled partly toward fuel OPR">Fuel</th><th title="Split from the video evidence alone">Video only</th><th>OPR-split</th><th>Fuel OPR</th><th>Auto</th><th>Teleop</th><th>Launches seen</th><th>Tracked</th><th>Tower (auto/end)</th><th>Points</th></tr></thead><tbody>`;
    for (const al of ["red", "blue"]) for (const t of D.tba[al] || []) {
      const st = S.teamStats[t]; const auto = st.per.auto || 0; const tele = Object.entries(st.per).filter(([k]) => k !== "auto").reduce((a, [, v]) => a + v, 0);
      h += `<tr data-team="${t}"><td>${plate(t, S.color[t])}</td><td>${al}</td><td>${fmt(st.fuel)}</td><td>${fmt(st.video)}</td><td>${fmt(st.oprShare)}</td><td>${fmt(st.opr)}</td><td>${fmt(auto)}</td><td>${fmt(tele)}</td><td>${st.shots}</td><td>${fmt(st.tracked)} s</td><td>${st.tower}</td><td>${fmt(st.points)}</td></tr>`;
    }
    body.innerHTML = h + "</tbody></table>";
    body.querySelectorAll("tr[data-team]").forEach((tr) => (tr.onclick = () => selectTeam(+tr.dataset.team)));
  } else if (S.sub === "segments") {
    const segs = Object.values(S.segIndex).sort((a, b) => a.t0 - b.t0).filter((s) => !S.segFilter || S.contradicted.has(`${s.alliance}:${s.id}`));
    let h = `<label class="tog"><input type="checkbox" id="segfilter" ${S.segFilter ? "checked" : ""}><span>Only segments labelled against their own bumper reads (${S.contradicted.size})</span></label>
             <table><thead><tr><th>Seg</th><th>Alliance</th><th>Slot</th><th>Start</th><th>End</th><th>Length</th><th>Bumper OCR evidence</th><th>Labelled</th></tr></thead><tbody>`;
    for (const s of segs) {
      const teams = D.tba[s.alliance] || []; const ev = s.ocr.map((x, k) => x + (s.held?.[k] || 0));
      const evs = teams.map((t, k) => ev[k] > 0 ? `<span style="color:${S.color[t]}">${t}:${fmt(ev[k], 1)}</span>` : "").filter(Boolean).join(" ") || '<span class="muted">none</span>';
      const bad = S.contradicted.has(`${s.alliance}:${s.id}`);
      h += `<tr data-seg="${s.id}" class="${bad ? "bad" : ""}"><td>s${s.id}</td><td>${s.alliance}</td><td>${s.slot}</td><td>${fmt(toMT(s.t0), 1)}</td><td>${fmt(toMT(s.t1), 1)}</td><td>${fmt(s.t1 - s.t0, 1)} s</td><td>${evs}</td><td>${s.team != null ? plate(s.team, S.color[s.team]) : "?"}</td></tr>`;
    }
    body.innerHTML = h + "</tbody></table>";
    $("#segfilter").onchange = (e) => { S.segFilter = e.target.checked; renderSub(); };
    body.querySelectorAll("tr[data-seg]").forEach((tr) => (tr.onclick = () => { const s = S.segIndex[tr.dataset.seg]; seekVT(s.t0 + 0.05); selectSeg(+tr.dataset.seg); }));
  } else if (S.sub === "shots") {
    let h = `<table><thead><tr><th>Match time</th><th>Team</th><th>View</th><th>Linked by</th><th>Toward own hub</th><th>Weight</th></tr></thead><tbody>`;
    for (const sh of [...D.result.shots].sort((a, b) => a.t - b.t)) {
      h += `<tr data-t="${sh.t}"><td>${fmt(toMT(sh.t), 2)}</td><td>${sh.team != null ? plate(sh.team, S.color[sh.team]) : "?"}</td><td>${sh.view}</td><td>${sh.how === "box" ? "robot box in same view" : sh.how === "proj" ? "robot tracked in another view" : sh.how}</td><td>${sh.toward == null ? "–" : sh.toward ? "yes" : "no (pass?)"}</td><td>${fmt(sh.conf, 2)}</td></tr>`;
    }
    body.innerHTML = h + "</tbody></table>";
    body.querySelectorAll("tr[data-t]").forEach((tr) => (tr.onclick = () => seekVT(+tr.dataset.t - 0.3)));
  } else if (S.sub === "labels") {
    renderLabelsTable(body);
  } else {
    const fl = D.flags;
    body.innerHTML = fl.length ? fl.map((f) => `<div class="flag ${f.level}"><b>${f.level === "bad" ? "Issue" : "Note"}</b><span>${f.text}</span></div>`).join("") : `<div class="none">Nothing flagged for this match.</div>`;
  }
}

// ---------------------------------------------------------------- shooter labelling
const LAB_HIDE = ["boxes", "unassigned", "shots", "credit", "segids", "solo"];

async function toggleLabel(on) {
  const L = S.label; L.on = on ?? !L.on;
  $("#label-toggle").classList.toggle("active", L.on);
  $("#label-toggle").textContent = L.on ? "Done labelling" : "Label shooters";
  $("#match").classList.toggle("labeling", L.on);
  $("#label-card").hidden = !L.on;
  if (L.on) {
    enter([$("#label-card")], { y: 16 });
    L.prevLayers = { ...S.layers };
    for (const k of LAB_HIDE) S.layers[k] = false;
    document.querySelectorAll("[data-layer]").forEach((cb) => (cb.checked = !!S.layers[cb.dataset.layer]));
    $("#lab-name").value = L.name;
    S.sub = "labels"; renderSub();
    if (S.D) await loadLabelData();
  } else {
    if (L.prevLayers) S.layers = L.prevLayers;
    document.querySelectorAll("[data-layer]").forEach((cb) => (cb.checked = !!S.layers[cb.dataset.layer]));
    L.stopAt = null; await flushSave();
    if (S.sub === "labels") { S.sub = "teams"; renderSub(); }
    resize(); renderHeader();
  }
  drawAll();
}

async function loadLabelData() {
  const L = S.label;
  const key = S.key;
  const B = await (await fetch(`/api/match/${key}/bursts`)).json();
  L.bursts = B.bursts || []; L.teams = B.teams || { red: [], blue: [] }; L.labels = {}; L.labelsKey = key;
  if (L.name) {
    const r = await (await fetch(`/api/labels/${key}?labeller=${encodeURIComponent(L.name)}`)).json();
    if (!r.error && L.labelsKey === key) L.labels = r.labels || {};
  }
  const first = L.bursts.findIndex((b) => !L.labels[b.id]);
  gotoBurst(first < 0 ? 0 : first, false);
  renderSub();
}

function gotoBurst(i, play = true) {
  const L = S.label; if (!L.bursts.length) { L.cur = null; renderLabelCard(); return; }
  L.i = Math.max(0, Math.min(L.bursts.length - 1, i));
  const b = L.bursts[L.i]; const prev = L.labels[b.id];
  L.cur = { shooters: new Set(prev?.shooters || []), status: prev?.status || null };
  renderLabelCard(); renderSub(); drawTimeline();
  if (play) playBurst(); else seekVT(b.play0);
}

function playBurst() {
  const L = S.label, b = L.bursts[L.i]; if (!b) return; const v = $("#video");
  v.playbackRate = parseFloat($("#lab-speed").value);
  v.currentTime = b.play0; L.stopAt = b.play1 + 0.25; v.play();
}

function renderLabelCard() {
  const L = S.label, b = L.bursts[L.i];
  const done = L.bursts.filter((x) => L.labels[x.id]).length;
  $("#lab-progress").innerHTML = `${done} / ${L.bursts.length} bursts labelled` +
    (L.bursts.length ? ` · ${L.bursts.filter((x) => L.labels[x.id]).reduce((a, x) => a + x.balls, 0)} of ${L.bursts.reduce((a, x) => a + x.balls, 0)} balls` : "");
  if (!b) { $("#lab-burst").innerHTML = `<span class="faint">No counter jumps found for this match.</span>`; $("#lab-teams").innerHTML = ""; return; }
  const col = b.alliance === "red" ? css("--red-text") : css("--blue-text");
  const st = L.labels[b.id];
  $("#lab-burst").innerHTML = `<div class="lb-top"><span class="lb-al" style="color:${col}">${b.alliance === "red" ? "Red" : "Blue"} alliance</span>
      <span class="lb-n">Burst ${L.i + 1} of ${L.bursts.length}</span><span class="lb-balls">+${b.balls}<small>fuel</small></span></div>
    <div class="faint">Counter rose at video ${b.t0.toFixed(1)} to ${b.t1.toFixed(1)} s. The balls entered the hub around match ${b.mt.toFixed(1)} s (${periodOf(b.mt)}).</div>
    <div class="lb-state">${st ? `Saved: ${labelText(st)}` : "Not labelled yet"}</div>`;
  const teams = L.teams[b.alliance] || [];
  $("#lab-teams").innerHTML = "";
  teams.forEach((t, k) => {
    const on = L.cur.shooters.has(t);
    const btn = el("button", { className: "shooter-btn" + (on ? " active" : "") },
      `${plate(t, S.color[t])}<kbd>${k + 1}</kbd>`);
    btn.onclick = () => toggleShooter(t);
    $("#lab-teams").append(btn);
  });
  $("#lab-other").classList.toggle("active", L.cur.status === "other");
  $("#lab-unsure").classList.toggle("active", L.cur.status === "unsure");
}

function labelText(st) {
  if (st.status === "labeled") return st.shooters.join(" + ");
  return { other: "human player / other", unsure: "can't tell", skip: "skipped" }[st.status] || st.status;
}

function toggleShooter(t) {
  const c = S.label.cur; if (!c) return;
  c.shooters.has(t) ? c.shooters.delete(t) : c.shooters.add(t);
  c.status = c.shooters.size ? "labeled" : null; renderLabelCard();
}
function setLabelStatus(st) {
  const c = S.label.cur; if (!c) return;
  c.status = c.status === st ? null : st; if (c.status) c.shooters.clear(); renderLabelCard();
}

function commitCurrent() {
  const L = S.label, b = L.bursts[L.i]; if (!b || !L.cur) return false;
  if (!L.cur.status) return false;
  L.labels[b.id] = { alliance: b.alliance, t0: b.t0, t1: b.t1, balls: b.balls, status: L.cur.status,
                     shooters: [...L.cur.shooters], at: new Date().toISOString().slice(0, 19) };
  scheduleSave(); return true;
}

function nextLabel() {
  const L = S.label;
  if (!L.name) { $("#lab-name").focus(); $("#lab-saved").textContent = "enter your name first"; return; }
  if (!commitCurrent()) { $("#lab-saved").textContent = "pick the shooter(s), Can't tell, or Other first"; return; }
  let j = L.bursts.findIndex((x, k) => k > L.i && !L.labels[x.id]);
  if (j < 0) j = Math.min(L.bursts.length - 1, L.i + 1);
  renderSub(); gotoBurst(j);
}

function scheduleSave() {
  // save right away (small payload); labels belong to the match they were loaded for
  S.label.dirty = true; $("#lab-saved").textContent = "saving…"; flushSave();
}
async function flushSave() {
  const L = S.label; clearTimeout(L.saveTimer); L.saveTimer = null;
  const key = L.labelsKey, payload = { labeller: L.name, labels: { ...L.labels } };
  if (!L.name || !key || !Object.keys(payload.labels).length) return;
  try {
    const r = await (await fetch(`/api/labels/${key}`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload) })).json();
    if (r.ok) L.dirty = false;
    if (L.labelsKey === key) $("#lab-saved").textContent = r.ok ? `saved (${r.saved})` : `not saved: ${r.error}`;
    const m = S.matches.find((x) => x.key === key); if (m && r.progress) { m.labels = r.progress; renderList(); }
  } catch (e) { $("#lab-saved").textContent = "not saved: " + e; }
}

function renderLabelsTable(body) {
  const L = S.label;
  if (!L.bursts.length) { body.innerHTML = `<div class="none">Turn on <b>Label shooters</b> under the video to label this match.</div>`; return; }
  let h = `<table><thead><tr><th>#</th><th>Alliance</th><th>Match time</th><th>Fuel</th><th>Label</th></tr></thead><tbody>`;
  L.bursts.forEach((b, k) => {
    const st = L.labels[b.id];
    h += `<tr data-k="${k}" class="${k === L.i ? "cur" : ""}"><td>${k + 1}</td><td class="${b.alliance === "red" ? "r" : "b"}">${b.alliance}</td>
          <td>${b.mt.toFixed(1)} s · ${periodOf(b.mt)}</td><td>+${b.balls}</td><td>${st ? labelText(st) : '<span class="faint">–</span>'}</td></tr>`;
  });
  body.innerHTML = h + "</tbody></table>";
  body.querySelectorAll("tr[data-k]").forEach((tr) => (tr.onclick = () => gotoBurst(+tr.dataset.k)));
}

function labelKey(e) {
  const L = S.label; if (!L.on) return false;
  const k = e.key.toLowerCase();
  const teams = L.bursts[L.i] ? L.teams[L.bursts[L.i].alliance] || [] : [];
  if (["1", "2", "3"].includes(k) && teams[+k - 1] != null) toggleShooter(teams[+k - 1]);
  else if (k === "h") setLabelStatus("other");
  else if (k === "u") setLabelStatus("unsure");
  else if (k === "r") playBurst();
  else if (k === "p") gotoBurst(L.i - 1);
  else if (k === "n") gotoBurst(L.i + 1);
  else if (e.key === "Enter") nextLabel();
  else return false;
  e.preventDefault(); return true;
}

// ---------------------------------------------------------------- interaction
function seekVT(t) { const v = $("#video"); v.currentTime = Math.max(0, t); drawAll(); }

function bind() {
  $("#filter").oninput = renderList; $("#event-filter").onchange = renderList; $("#flagged-only").onchange = renderList;
  const v = $("#video");
  $("#play").onclick = () => (v.paused ? v.play() : v.pause());
  document.querySelectorAll("[data-step]").forEach((b) => (b.onclick = () => seekVT(v.currentTime + parseFloat(b.dataset.step))));
  document.querySelectorAll("[data-jump]").forEach((b) => (b.onclick = () => seekVT(toVT(parseFloat(b.dataset.jump)))));
  $("#speed").onchange = (e) => (v.playbackRate = parseFloat(e.target.value));
  document.querySelectorAll("[data-layer]").forEach((cb) => (cb.onchange = () => { S.layers[cb.dataset.layer] = cb.checked; drawAll(); }));
  document.querySelectorAll(".sub").forEach((b) => (b.onclick = () => { S.sub = b.dataset.sub; renderSub(); }));
  v.addEventListener("play", tick); v.addEventListener("pause", drawAll); v.addEventListener("seeked", drawAll);
  v.addEventListener("timeupdate", () => { if (v.paused) drawAll(); });
  window.addEventListener("resize", () => { resize(); drawAll(); });

  $("#overlay").onclick = (e) => {
    const c = e.target, r = c.getBoundingClientRect();
    const fx = (e.clientX - r.left) / r.width * FRAME_W, fy = (e.clientY - r.top) / r.height * FRAME_H;
    let best = null, area = Infinity;
    for (const i of currentGroup()) { const d = S.D.dets[i]; if (!visibleDet(d)) continue;
      if (fx >= d[2] && fx <= d[4] && fy >= d[3] && fy <= d[5]) { const a = (d[4] - d[2]) * (d[5] - d[3]); if (a < area) { area = a; best = i; } } }
    if (best != null) selectDet(best); else { S.sel = { det: null, seg: null, team: null }; renderSelection(); drawAll(); }
  };
  $("#fieldmap").onclick = (e) => {
    const c = e.target, r = c.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    const [mx, my] = fieldXform(c).p2m((e.clientX - r.left) * dpr, (e.clientY - r.top) * dpr);
    const mt = toMT(v.currentTime); let best = null, bd = 1.2;
    for (const t of Object.keys(S.color).map(Number)) { const p = posAt(t, mt); if (!p) continue; const d = Math.hypot(p[1] - mx, p[2] - my); if (d < bd) { bd = d; best = t; } }
    if (best != null) selectTeam(best);
  };
  const tl = $("#timeline"), tip = $("#tl-tip");
  tl.onclick = (e) => {
    const r = tl.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    const px = (e.clientX - r.left) * dpr, py = (e.clientY - r.top) * dpr;
    const mt = tlMT(tl, px); seekVT(toVT(mt));
    const row = TL.rows.find((rw) => py >= rw.y0 && py <= rw.y1);
    if (row && row.kind === "team") { const s = row.segs.find((s) => toMT(s.t0) <= mt && mt <= toMT(s.t1)); if (s) selectSeg(s.id); else selectTeam(row.team); }
  };
  tl.onmousemove = (e) => {
    if (!S.D) return; const r = tl.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    const px = (e.clientX - r.left) * dpr, py = (e.clientY - r.top) * dpr; const mt = tlMT(tl, px);
    const row = TL.rows.find((rw) => py >= rw.y0 && py <= rw.y1); let txt = `${mt.toFixed(1)} s · ${periodOf(mt)}`;
    if (row?.kind === "counter") txt += ` · ${row.al} +${row.inc.get(Math.floor(mt)) || 0} fuel this second`;
    if (row?.kind === "team") {
      const s = row.segs.find((s) => toMT(s.t0) <= mt && mt <= toMT(s.t1)); const cr = row.per.get(Math.floor(mt)) || 0;
      txt += ` · ${row.team}` + (s ? ` · segment s${s.id}` : " · not tracked") + (cr ? ` · credited ${cr.toFixed(1)}` : "");
    }
    tip.hidden = false; tip.textContent = txt; tip.style.left = (e.clientX - tl.closest(".card").getBoundingClientRect().left + 12) + "px";
    tip.style.top = (e.clientY - tl.closest(".card").getBoundingClientRect().top + 14) + "px";
  };
  tl.onmouseleave = () => (tip.hidden = true);

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || !S.D) return;
    if (labelKey(e)) return;
    if (e.code === "Space") { e.preventDefault(); v.paused ? v.play() : v.pause(); }
    else if (e.key === "ArrowLeft") seekVT(v.currentTime - 1);
    else if (e.key === "ArrowRight") seekVT(v.currentTime + 1);
    else if (e.key === ",") seekVT(v.currentTime - 1 / 60);
    else if (e.key === ".") seekVT(v.currentTime + 1 / 60);
    else if (e.key === "Escape" && S.selJump != null) { S.selJump = null; drawAll(); }
  });
  document.querySelectorAll(".tab-btn").forEach((b) => (b.onclick = () => showView(b.dataset.view)));
  // shooter labelling
  $("#label-toggle").onclick = () => toggleLabel();
  $("#lab-other").onclick = () => setLabelStatus("other");
  $("#lab-unsure").onclick = () => setLabelStatus("unsure");
  $("#lab-prev").onclick = () => gotoBurst(S.label.i - 1);
  $("#lab-next").onclick = () => nextLabel();
  $("#lab-replay").onclick = () => playBurst();
  $("#lab-name").onchange = async (e) => {
    const name = e.target.value.trim(); S.label.name = name;
    try { localStorage.setItem("as_labeller", name); } catch (err) { /* storage blocked */ }
    if (S.D && S.label.on) await loadLabelData();
  };
  v.addEventListener("timeupdate", () => {
    const L = S.label;
    if (L.on && L.stopAt != null && v.currentTime >= L.stopAt) { v.pause(); L.stopAt = null; v.playbackRate = parseFloat($("#speed").value); }
  });
  window.addEventListener("beforeunload", () => {
    const L = S.label;
    if (L.dirty && L.name && L.labelsKey && navigator.sendBeacon)
      navigator.sendBeacon(`/api/labels/${L.labelsKey}`, new Blob([JSON.stringify({ labeller: L.name, labels: L.labels })], { type: "application/json" }));
  });
  window.addEventListener("beforeunload", () => { if (S.key) history.replaceState(null, "", `#match=${S.key}&t=${v.currentTime.toFixed(2)}`); });
}

function tick() {
  const v = $("#video"); drawAll();
  const L = S.label;
  if (L.on && L.stopAt != null && v.currentTime >= L.stopAt) { v.pause(); L.stopAt = null; v.playbackRate = parseFloat($("#speed").value); return; }
  if (v.paused || v.ended) return;
  if (v.requestVideoFrameCallback) v.requestVideoFrameCallback(tick); else requestAnimationFrame(tick);
}

// ---------------------------------------------------------------- system diagram view
let docLoaded = false;
async function showView(name) {
  document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
  $("#inspect-view").hidden = name !== "inspect"; $("#system-view").hidden = name !== "system";
  if (name === "inspect") { resize(); drawAll(); return; }
  if (docLoaded) return; docLoaded = true;
  const doc = $("#doc");
  try {
    const md = await (await fetch("/docs/ARCHITECTURE.md")).text();
    if (window.marked) doc.innerHTML = marked.parse(md); else doc.textContent = md;
    const blocks = [...doc.querySelectorAll("pre code.language-mermaid")];
    for (const b of blocks) { const d = el("div", { className: "mermaid" }); d.textContent = b.textContent; b.parentElement.replaceWith(d); }
    if (window.mermaid) {
      mermaid.initialize({
        startOnLoad: false, theme: "base", securityLevel: "loose", flowchart: { useMaxWidth: true }, sequence: { useMaxWidth: true },
        themeVariables: {  // the inspector's near-black palette instead of mermaid's light default
          darkMode: true, fontFamily: C.font, fontSize: "13px", background: C.surface,
          primaryColor: "#1c1c1c", primaryTextColor: C.text, primaryBorderColor: "#3a3a3a",
          secondaryColor: "#161616", tertiaryColor: "#141414", lineColor: "#8a8a8a", textColor: C.text2,
          clusterBkg: "#161616", clusterBorder: "#2e2e2e", edgeLabelBackground: C.surface, titleColor: C.text2,
          nodeBorder: "#3a3a3a", mainBkg: "#1c1c1c", noteBkgColor: "#2a2412", noteTextColor: C.text, noteBorderColor: "#5c4b12",
          actorBkg: "#1c1c1c", actorBorder: "#3a3a3a", actorTextColor: C.text, actorLineColor: "#4a4a4a",
          signalColor: "#bdbdbd", signalTextColor: C.text, labelBoxBkgColor: "#1c1c1c", labelBoxBorderColor: "#3a3a3a",
          labelTextColor: C.text, loopTextColor: C.text2, activationBkgColor: "#2a2a2a", activationBorderColor: "#4a4a4a",
          sequenceNumberColor: C.surface, classText: C.text,
        },
      });
      await mermaid.run({ nodes: doc.querySelectorAll(".mermaid") });
      // a download button under each diagram, named after the section it sits in
      doc.querySelectorAll(".mermaid").forEach((d, i) => {
        let h = d.previousElementSibling; while (h && !/^H[1-3]$/.test(h.tagName)) h = h.previousElementSibling;
        const name = (h ? h.textContent : `diagram-${i + 1}`).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
        const btn = el("button", { textContent: "Download SVG", className: "ghost" });
        btn.onclick = () => {
          const svg = d.querySelector("svg"); if (!svg) return;
          const blob = new Blob([new XMLSerializer().serializeToString(svg)], { type: "image/svg+xml" });
          const a = el("a", { href: URL.createObjectURL(blob), download: `${name}.svg` }); a.click();
        };
        d.after(btn);
      });
    }
  } catch (e) { doc.textContent = "Could not load docs/ARCHITECTURE.md: " + e; }
}

window.addEventListener("hashchange", () => {
  const h = new URLSearchParams(location.hash.slice(1)); const k = h.get("match");
  const t = parseFloat(h.get("t") || "NaN");
  if (k && k !== S.key) openMatch(k, t); else if (k && Number.isFinite(t)) seekVT(t);
});

bind();
loadMatches();
// canvases are drawn with the web fonts; redraw once they arrive
document.fonts?.ready.then(() => { if (S.D) { resize(); drawAll(); } });
