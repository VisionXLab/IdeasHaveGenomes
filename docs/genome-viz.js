/*
 * Interactive Idea-Genome / GenomeDiff lineage graph for the #genome section.
 * Renders one lineage trace (Transformer -> ViT -> LLaVA) as Idea Genomes aligned by
 * GenomeDiffs, colored by unit fate, with the inherited-driver backbone highlighted and
 * each transition tagged with its evolutionary dynamics (Table 2). Palette matches styles.css.
 * Requires d3 v7 (loaded before this file). No-op if the container / d3 are absent.
 */
(function () {
  "use strict";
  if (typeof d3 === "undefined") return;
  const svgEl = document.getElementById("genomeGraph");
  if (!svgEl) return;

  // ---- palette (mirrors styles.css :root) ----
  const INK = "#172033", MUTED = "#5b6576", LINE = "#e7dfd2", PAPER = "#fffaf0";
  const TYPE = { niche: "#5b6576", mechanism: "#3d84c5", observation: "#2f9a9a",
                 limitation: "#c7594b", delta: "#db8b2b", claim: "#8662b5" };
  const FATE = { Inherited: "#4f9a4d", Mutated: "#db8b2b", Lost: "#c7594b",
                 Novel: "#3d84c5", External: "#8662b5" };
  const SHAPE = { niche: d3.symbolSquare, mechanism: d3.symbolStar, observation: d3.symbolCircle,
                  limitation: d3.symbolTriangle, delta: d3.symbolDiamond, claim: d3.symbolWye };
  const ROLE_DESC = { niche: "setting", mechanism: "driver", observation: "finding",
                      limitation: "weakness", delta: "change", claim: "contribution" };

  // ---- data ----
  const PAPERS = [
    { id: "P0", title: "Transformer", year: "2017 · t₀", col: 0 },
    { id: "P1", title: "ViT",         year: "2020 · t₁", col: 1 },
    { id: "P2", title: "LLaVA",       year: "2023 · t₂", col: 2 },
  ];
  // [uid, paper, type, content, driver]
  const UNITS = [
    ["A_niche", "P0", "niche", "seq-to-seq NLP", false],
    ["A_mech",  "P0", "mechanism", "self-attention", true],
    ["A_obs",   "P0", "observation", "long-range deps", false],
    ["A_lim",   "P0", "limitation", "token-only; O(n²)", false],
    ["A_claim", "P0", "claim", "attn is all you need", false],
    ["B_niche", "P1", "niche", "image classification", false],
    ["B_mech",  "P1", "mechanism", "self-attn on patches", true],
    ["B_delta", "P1", "delta", "image → patch tokens", false],
    ["B_obs",   "P1", "observation", "scales with data", false],
    ["B_lim",   "P1", "limitation", "data-hungry w/o pretrain", false],
    ["C_niche", "P2", "niche", "multimodal chat", false],
    ["C_mech",  "P2", "mechanism", "ViT/CLIP encoder", true],
    ["C_ext",   "P2", "mechanism", "instruction-tuned LLM", false],
    ["C_delta", "P2", "delta", "vision–language proj.", false],
    ["C_claim", "P2", "claim", "visual instruction tuning", false],
  ];
  const DIFFS = [
    { dynamics: "Adaptive Radiation", edges: [
      ["A_niche", "B_niche", "Mutated"], ["A_mech", "B_mech", "Inherited"],
      ["A_obs", "B_obs", "Inherited"], ["A_lim", "B_lim", "Mutated"],
      ["A_claim", null, "Lost"], [null, "B_delta", "Novel"] ] },
    { dynamics: "Hybridization", edges: [
      ["B_niche", "C_niche", "Mutated"], ["B_mech", "C_mech", "Inherited"],
      ["EXT", "C_ext", "External"], [null, "C_delta", "Novel"],
      [null, "C_claim", "Novel"], ["B_lim", null, "Lost"] ] },
  ];

  // ---- layout ----
  const W = 1200, H = 560, R_CAP = 112, R_RING = 80;
  const colX = (c) => 0.235 * W + c * 0.265 * W;
  const hubY = H * 0.36;
  PAPERS.forEach((p) => { p.x = colX(p.col); p.y = hubY; p.fx = p.x; p.fy = p.y; p.kind = "paper"; });
  const unitNodes = UNITS.map((u) => ({ id: u[0], paper: u[1], type: u[2], text: u[3], driver: u[4], kind: "unit" }));
  const nodeById = {};
  [...PAPERS, ...unitNodes].forEach((n) => (nodeById[n.id] = n));

  const memberLinks = unitNodes.map((u) => ({ source: u.paper, target: u.id }));
  const diffLinks = [], halfMarks = [];
  DIFFS.forEach((d, di) => d.edges.forEach(([s, t, fate]) => {
    if (fate === "Lost") return halfMarks.push({ unitId: s, fate });
    if (fate === "Novel") return halfMarks.push({ unitId: t, fate });
    if (fate === "External") return halfMarks.push({ unitId: t, fate });
    diffLinks.push({ source: s, target: t, fate, di });
  }));
  const markedIds = new Set(halfMarks.map((m) => m.unitId));
  const extIds = new Set(halfMarks.filter((m) => m.fate === "External").map((m) => m.unitId));
  diffLinks.forEach((l) => { const s = nodeById[l.source], t = nodeById[l.target];
    l.driverEdge = !!(s && t && s.driver && t.driver && l.fate === "Inherited"); });

  const perPaper = {};
  unitNodes.forEach((u) => (perPaper[u.paper] = perPaper[u.paper] || []).push(u));
  Object.values(perPaper).forEach((arr) => arr.forEach((u, i) => {
    const a = -Math.PI / 2 + (i / arr.length) * 2 * Math.PI;
    u.x = nodeById[u.paper].x + Math.cos(a) * 80; u.y = nodeById[u.paper].y + Math.sin(a) * 80;
  }));
  function pinExternal() { const c = nodeById["C_ext"], p = nodeById["P2"];
    if (c && p) { c.fx = p.x + 60; c.fy = p.y + 94; c.x = c.fx; c.y = c.fy; } }
  pinExternal();

  // ---- render ----
  const svg = d3.select(svgEl).attr("viewBox", `0 0 ${W} ${H}`).attr("preserveAspectRatio", "xMidYMid meet");
  svg.selectAll("*").remove();
  const defs = svg.append("defs");
  const hg = defs.append("radialGradient").attr("id", "gv-hub").attr("cx", "35%").attr("cy", "28%");
  hg.append("stop").attr("offset", "0%").attr("stop-color", "#39485f");
  hg.append("stop").attr("offset", "100%").attr("stop-color", "#1b2536");
  const glow = defs.append("filter").attr("id", "gv-glow").attr("x", "-50%").attr("y", "-50%").attr("width", "200%").attr("height", "200%");
  glow.append("feGaussianBlur").attr("stdDeviation", "3").attr("result", "b");
  const fm = glow.append("feMerge"); fm.append("feMergeNode").attr("in", "b"); fm.append("feMergeNode").attr("in", "SourceGraphic");
  Object.entries(FATE).forEach(([k, c]) => defs.append("marker").attr("id", "gv-arw-" + k)
    .attr("viewBox", "0 0 10 10").attr("refX", 9).attr("refY", 5).attr("markerWidth", 7).attr("markerHeight", 7)
    .attr("orient", "auto-start-reverse").append("path").attr("d", "M0,0 L10,5 L0,10 z").attr("fill", c));

  const root = svg.append("g");
  const gHull = root.append("g"), gMember = root.append("g"), gSpine = root.append("g"),
        gDiff = root.append("g"), gHalf = root.append("g"), gNode = root.append("g"), gLabel = root.append("g");
  svg.call(d3.zoom().scaleExtent([0.6, 2.5]).on("zoom", (e) => root.attr("transform", e.transform)));

  gHull.selectAll("circle").data(PAPERS).join("circle").attr("r", R_CAP)
    .attr("fill", "#000").attr("fill-opacity", 0.03).attr("stroke", "#b8bec9")
    .attr("stroke-opacity", 0.35).attr("stroke-width", 1.2).attr("stroke-dasharray", "2 5");
  const memSel = gMember.selectAll("line").data(memberLinks).join("line")
    .attr("stroke", "#cdcabf").attr("stroke-opacity", 0.6).attr("stroke-width", 1);

  const _cv = document.createElement("canvas").getContext("2d");
  const measure = (s, fs, fw) => { _cv.font = `${fw || 400} ${fs}px Inter, Arial`; return _cv.measureText(s).width; };

  const spineSel = gSpine.selectAll("path").data(diffLinks.filter((d) => d.driverEdge)).join("path")
    .attr("fill", "none").attr("stroke", FATE.Inherited).attr("stroke-opacity", 0.22)
    .attr("stroke-width", 15).attr("stroke-linecap", "round");
  const diffSel = gDiff.selectAll("path").data(diffLinks).join("path").attr("fill", "none")
    .attr("stroke", (d) => FATE[d.fate]).attr("stroke-width", (d) => (d.driverEdge ? 3.6 : 2.2)).attr("stroke-opacity", 0.92)
    .attr("stroke-dasharray", (d) => (d.fate === "Mutated" ? "7 4" : "none"))
    .attr("marker-end", (d) => "url(#gv-arw-" + d.fate + ")");

  const halfSel = gHalf.selectAll("g").data(halfMarks).join("g");
  halfSel.each(function (m) {
    const g = d3.select(this);
    if (m.fate === "External") {
      g.append("circle").attr("r", 9).attr("fill", FATE.External).attr("stroke", "#fff").attr("stroke-width", 1.6);
      g.append("path").attr("d", "M-3,-3 L2.6,0 L-3,3 Z").attr("fill", "#fff");
      return;
    }
    const c = m.fate === "Lost" ? FATE.Lost : FATE.Novel, sym = m.fate === "Lost" ? "✕" : "+";
    g.append("circle").attr("r", 8.5).attr("fill", "#fff").attr("stroke", c).attr("stroke-width", 1.8);
    g.append("text").attr("text-anchor", "middle").attr("dy", "0.34em").attr("font-size", 12).attr("font-weight", 800).attr("fill", c).text(sym);
  });

  const hubSel = gNode.selectAll("g.hub").data(PAPERS).join("g");
  hubSel.each(function (d) {
    const g = d3.select(this), w = Math.max(104, measure(d.title, 15, 800) + 40), h = 46;
    d.chipW = w;
    g.append("rect").attr("x", -w / 2).attr("y", -h / 2).attr("width", w).attr("height", h).attr("rx", 12)
      .attr("fill", "url(#gv-hub)").attr("filter", "url(#gv-glow)").attr("stroke", "#fff").attr("stroke-width", 2.2);
    g.append("text").attr("text-anchor", "middle").attr("dy", "-2").attr("fill", "#fff").attr("font-size", 15).attr("font-weight", 800).text(d.title);
    g.append("text").attr("text-anchor", "middle").attr("dy", "14").attr("fill", "#c3cdda").attr("font-size", 10).text(d.year);
  });

  const symGen = (type, area) => d3.symbol().type(SHAPE[type]).size(area)();
  const uR = (d) => (d.driver ? 15 : 10);
  const uSel = gNode.selectAll("g.unit").data(unitNodes).join("g").style("cursor", "grab");
  uSel.append("path").attr("d", (d) => symGen(d.type, d.driver ? 400 : 170)).attr("fill", (d) => TYPE[d.type])
    .attr("stroke", "#fff").attr("stroke-width", (d) => (d.driver ? 2.4 : 1.7)).attr("filter", (d) => (d.driver ? "url(#gv-glow)" : null));

  const uLab = gLabel.selectAll("text").data(unitNodes).join("text")
    .attr("paint-order", "stroke").attr("stroke", "#fffdf7").attr("stroke-width", 3.2);
  uLab.append("tspan").attr("class", "c").attr("font-size", (d) => (d.driver ? 12 : 11)).attr("font-weight", 800).attr("fill", INK).text((d) => d.text);
  uLab.append("tspan").attr("class", "t").attr("font-size", 8.6).attr("font-weight", 700)
    .attr("fill", (d) => (extIds.has(d.id) ? FATE.External : TYPE[d.type])).text((d) => (extIds.has(d.id) ? d.type + " · external" : d.type));

  const plate = DIFFS.map((d, i) => ({ dynamics: d.dynamics, x: (colX(i) + colX(i + 1)) / 2, y: hubY + 66 }));
  const plateSel = gLabel.selectAll("g.plate").data(plate).join("g").attr("transform", (d) => `translate(${d.x},${d.y})`);
  plateSel.each(function (d) {
    const g = d3.select(this), label = "⟳ " + d.dynamics, w = measure(label, 12.5, 800) + 26;
    g.append("rect").attr("x", -w / 2).attr("y", -14).attr("width", w).attr("height", 28).attr("rx", 14)
      .attr("fill", "#eef7ee").attr("stroke", FATE.Inherited).attr("stroke-width", 1.6);
    g.append("text").attr("text-anchor", "middle").attr("dy", "0.34em").attr("font-size", 12.5).attr("font-weight", 800).attr("fill", "#3f7d3d").text(label);
  });

  // ---- forces ----
  function diffPath(d) {
    const s = nodeById[d.source.id || d.source], t = nodeById[d.target.id || d.target];
    const mx = (s.x + t.x) / 2, my = (s.y + t.y) / 2 - Math.min(90, Math.abs(t.x - s.x) * 0.3);
    const dS = Math.hypot(mx - s.x, my - s.y) || 1, dT = Math.hypot(mx - t.x, my - t.y) || 1;
    const sx = s.x + ((mx - s.x) / dS) * (uR(s) + 5), sy = s.y + ((my - s.y) / dS) * (uR(s) + 5);
    const ex = t.x + ((mx - t.x) / dT) * (uR(t) + 8), ey = t.y + ((my - t.y) / dT) * (uR(t) + 8);
    return `M${sx},${sy} Q${mx},${my} ${ex},${ey}`;
  }
  function ring(alpha) {
    unitNodes.forEach((u) => { const h = nodeById[u.paper]; const dx = u.x - h.x, dy = u.y - h.y, d = Math.hypot(dx, dy) || 1;
      const k = ((d - R_RING) / d) * alpha * 0.35; u.vx -= dx * k; u.vy -= dy * k; });
  }
  const sim = d3.forceSimulation([...PAPERS, ...unitNodes])
    .force("diff", d3.forceLink(diffLinks.map((l) => ({ source: l.source, target: l.target }))).id((d) => d.id).distance(220).strength(0.015))
    .force("charge", d3.forceManyBody().strength((d) => (d.kind === "unit" ? -110 : -40)))
    .force("collide", d3.forceCollide().radius((d) => (d.kind === "paper" ? (d.chipW ? d.chipW / 2 : 56) + 8 : (d.driver ? 18 : 13) + 24)).strength(0.9))
    .on("tick", ticked);

  function ticked() {
    ring(sim.alpha());
    gHull.selectAll("circle").attr("cx", (d) => d.x).attr("cy", (d) => d.y);
    memSel.attr("x1", (d) => nodeById[d.source.id || d.source].x).attr("y1", (d) => nodeById[d.source.id || d.source].y)
      .attr("x2", (d) => nodeById[d.target.id || d.target].x).attr("y2", (d) => nodeById[d.target.id || d.target].y);
    diffSel.attr("d", diffPath); spineSel.attr("d", diffPath);
    halfSel.attr("transform", (m) => { const u = nodeById[m.unitId], h = nodeById[u.paper];
      const dx = u.x - h.x, dy = u.y - h.y, d = Math.hypot(dx, dy) || 1; return `translate(${u.x + (dx / d) * 13},${u.y + (dy / d) * 13})`; });
    hubSel.attr("transform", (d) => `translate(${d.x},${d.y})`);
    uSel.attr("transform", (d) => `translate(${d.x},${d.y})`);
    uLab.each(function (d) { const h = nodeById[d.paper], dx = d.x - h.x, dy = d.y - h.y, dd = Math.hypot(dx, dy) || 1;
      const off = markedIds.has(d.id) ? 30 : uR(d) + 12, lx = d.x + (dx / dd) * off, ly = d.y + (dy / dd) * off, up = dy < -4;
      const s = d3.select(this).attr("text-anchor", d.x < h.x - 6 ? "end" : d.x > h.x + 6 ? "start" : "middle");
      s.select("tspan.c").attr("x", lx).attr("y", up ? ly - 12 : ly);
      s.select("tspan.t").attr("x", lx).attr("y", up ? ly - 0.5 : ly + 12); });
  }

  // ---- interaction ----
  let tip = document.getElementById("gv-tip");
  if (!tip) { tip = document.createElement("div"); tip.id = "gv-tip"; tip.className = "gv-tip"; document.body.appendChild(tip); }
  uSel.on("mousemove", (e, d) => {
    const fates = [];
    diffLinks.forEach((l) => { if ((l.source.id || l.source) === d.id) fates.push(l.fate + " → next"); if ((l.target.id || l.target) === d.id) fates.push("prev → " + l.fate); });
    halfMarks.forEach((m) => { if (m.unitId === d.id) fates.push(m.fate); });
    tip.style.display = "block"; tip.style.left = e.clientX + 14 + "px"; tip.style.top = e.clientY + 14 + "px";
    tip.innerHTML = `<span class="gv-ty" style="background:${TYPE[d.type]}">${d.type}${d.driver ? " ★ driver" : ""}</span><b>${d.text}</b>${fates.length ? `<div class="gv-fate">fate: ${[...new Set(fates)].join(" · ")}</div>` : ""}`;
  }).on("mouseleave", () => (tip.style.display = "none"));

  uSel.call(d3.drag()
    .on("start", (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
    .on("drag", (e, d) => { d.fx = e.x; d.fy = e.y; })
    .on("end", (e, d) => { if (!e.active) sim.alphaTarget(0); if (d.id !== "C_ext") { d.fx = null; d.fy = null; } }));

  // ---- legends ----
  const NS = "http://www.w3.org/2000/svg";
  const mk = (t, a, p) => { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); if (p) p.appendChild(e); return e; };
  const mini = (w, h) => { const s = mk("svg", { width: w, height: h, viewBox: `0 0 ${w} ${h}` }); s.style.flex = "0 0 auto"; return s; };
  const legend = document.getElementById("genomeLegend");
  if (legend) {
    legend.innerHTML = "";
    const roles = document.createElement("div"); roles.className = "gv-legsec";
    roles.innerHTML = "<h4>Idea Genome — unit role (shape)</h4>"; const ri = document.createElement("div"); ri.className = "gv-items";
    ["niche", "mechanism", "observation", "limitation", "delta", "claim"].forEach((role) => {
      const row = document.createElement("div"); row.className = "gv-li"; const s = mini(20, 20);
      mk("path", { d: d3.symbol().type(SHAPE[role]).size(role === "mechanism" ? 140 : 125)(), transform: "translate(10,10)", fill: TYPE[role], stroke: "#fff", "stroke-width": 1.2 }, s);
      row.appendChild(s); const sp = document.createElement("span");
      sp.innerHTML = `<b style="color:${TYPE[role]}">${role}</b> <span class="gv-mut">— ${ROLE_DESC[role]}</span>`; row.appendChild(sp); ri.appendChild(row);
    });
    roles.appendChild(ri); legend.appendChild(roles);

    const fates = document.createElement("div"); fates.className = "gv-legsec";
    fates.innerHTML = "<h4>GenomeDiff — unit fate (edge)</h4>"; const fi = document.createElement("div"); fi.className = "gv-items";
    [["Inherited", "solid", "arrow"], ["Mutated", "dash", "arrow"], ["Lost", "dot", "cross"], ["Novel", "solid", "plus"], ["External", "solid", "arrow"]].forEach(([name, style, end]) => {
      const row = document.createElement("div"); row.className = "gv-li"; const s = mini(38, 14), c = FATE[name];
      mk("line", { x1: 3, y1: 7, x2: 27, y2: 7, stroke: c, "stroke-width": name === "Inherited" ? 3 : 2.2, "stroke-dasharray": style === "dash" ? "6 3" : style === "dot" ? "2 3" : "none", "stroke-linecap": "round" }, s);
      if (end === "arrow") mk("path", { d: "M27,3 L35,7 L27,11 Z", fill: c }, s);
      else { const g = mk("g", { transform: "translate(32,7)" }, s); mk("circle", { r: 6, fill: "#fff", stroke: c, "stroke-width": 1.6 }, g);
        if (end === "cross") mk("path", { d: "M-2.4,-2.4 L2.4,2.4 M2.4,-2.4 L-2.4,2.4", stroke: c, "stroke-width": 1.6 }, g);
        else mk("path", { d: "M0,-3 L0,3 M-3,0 L3,0", stroke: c, "stroke-width": 1.8 }, g); }
      row.appendChild(s); const sp = document.createElement("span"); sp.innerHTML = `<b style="color:${c}">${name}</b>`; row.appendChild(sp); fi.appendChild(row);
    });
    fates.appendChild(fi); legend.appendChild(fates);

    const dyn = document.createElement("div"); dyn.className = "gv-legsec";
    dyn.innerHTML = "<h4>GenomeDiff — dynamics (Table 2)</h4>" +
      `<div class="gv-dyn"><span class="gv-dg">lineage — driver inherited:</span><br>Mutation · <span class="gv-hl">Adaptive Radiation ★</span> · <span class="gv-hl">Hybridization ★</span> · Speciation</div>` +
      `<div class="gv-dyn"><span class="gv-dn">co-location — no inheritance:</span><br>Niche Competition · Isolation</div>` +
      `<div class="gv-note">★ the two transitions shown above</div>`;
    legend.appendChild(dyn);
  }

  sim.alpha(1).restart();
})();
