/*
 * Interactive Idea-Genome / GenomeDiff explorer for the #genome section.
 * Shows the six evolutionary dynamics of Table 2 as selectable canonical examples: each is a
 * GenomeDiff between two (or three) Idea Genomes, with units colored/shaped by role and edges
 * colored by fate. Lineage cases highlight the inherited-driver "backbone"; co-location cases
 * make the absence of descent explicit. Palette matches styles.css. Requires d3 v7.
 */
(function () {
  "use strict";
  if (typeof d3 === "undefined") return;
  const svgEl = document.getElementById("genomeGraph");
  if (!svgEl) return;

  const INK = "#172033", MUTED = "#5b6576";
  const TYPE = { niche: "#5b6576", mechanism: "#3d84c5", observation: "#2f9a9a",
                 limitation: "#c7594b", delta: "#db8b2b", claim: "#8662b5" };
  const FATE = { Inherited: "#4f9a4d", Mutated: "#db8b2b", Lost: "#c7594b", Novel: "#3d84c5", External: "#8662b5" };
  const SHAPE = { niche: d3.symbolSquare, mechanism: d3.symbolStar, observation: d3.symbolCircle,
                  limitation: d3.symbolTriangle, delta: d3.symbolDiamond, claim: d3.symbolWye };
  const ROLE_DESC = { niche: "setting", mechanism: "driver", observation: "finding",
                      limitation: "weakness", delta: "change", claim: "contribution" };

  // ---- Table 2 examples. Each genome unit = [uid, type, content, isDriver]. ----
  const EXAMPLES = {
    mutation: {
      dynamics: "Mutation", lineage: true, layout: "chain2",
      pair: "YOLO → YOLOv2",
      crit: "Driver mechanism is inherited or locally mutated; niche stays the same or nearby.",
      genomes: [
        { id: "g0", title: "YOLO", year: "2016", units: [
          ["m_a_n", "niche", "real-time detection", 0], ["m_a_m", "mechanism", "one-stage grid regression", 1],
          ["m_a_l", "limitation", "coarse localization", 0], ["m_a_c", "claim", "unified real-time detector", 0] ] },
        { id: "g1", title: "YOLOv2", year: "2017", units: [
          ["m_b_n", "niche", "real-time detection", 0], ["m_b_m", "mechanism", "one-stage detection", 1],
          ["m_b_d", "delta", "anchors · BN · multi-scale", 0], ["m_b_o", "observation", "higher recall & mAP", 0] ] },
      ],
      edges: [["m_a_n", "m_b_n", "Inherited"], ["m_a_m", "m_b_m", "Inherited"],
              ["m_a_l", null, "Lost"], [null, "m_b_d", "Novel"], ["m_a_c", null, "Lost"], [null, "m_b_o", "Novel"]],
    },
    adaptiveRadiation: {
      dynamics: "Adaptive Radiation", lineage: true, layout: "chain2",
      pair: "Transformer → ViT",
      crit: "Driver mechanism persists, but moves into a new task, domain, or evaluation ecology.",
      genomes: [
        { id: "g0", title: "Transformer", year: "2017", units: [
          ["r_a_n", "niche", "seq-to-seq NLP", 0], ["r_a_m", "mechanism", "self-attention", 1],
          ["r_a_l", "limitation", "token-only; O(n²)", 0], ["r_a_c", "claim", "attention is all you need", 0] ] },
        { id: "g1", title: "ViT", year: "2020", units: [
          ["r_b_n", "niche", "image classification", 0], ["r_b_m", "mechanism", "self-attn on patches", 1],
          ["r_b_d", "delta", "image → patch tokens", 0], ["r_b_o", "observation", "scales with data", 0] ] },
      ],
      edges: [["r_a_n", "r_b_n", "Mutated"], ["r_a_m", "r_b_m", "Inherited"],
              ["r_a_l", null, "Lost"], [null, "r_b_d", "Novel"], ["r_a_c", null, "Lost"], [null, "r_b_o", "Novel"]],
    },
    hybridization: {
      dynamics: "Hybridization", lineage: true, layout: "hybrid",
      pair: "CLIP encoder + instruction-tuned LLM → LLaVA",
      crit: "Successor imports driver units from two or more distinct lineages.",
      genomes: [
        { id: "g0", title: "CLIP", year: "2021", units: [
          ["h_a_n", "niche", "image–text alignment", 0], ["h_a_m", "mechanism", "contrastive visual encoder", 1],
          ["h_a_c", "claim", "transferable features", 0] ] },
        { id: "g1", title: "Vicuna / LLM", year: "2023", units: [
          ["h_b_n", "niche", "instruction following", 0], ["h_b_m", "mechanism", "instruction-tuned LLM", 1],
          ["h_b_c", "claim", "chat assistant", 0] ] },
        { id: "g2", title: "LLaVA", year: "2023", units: [
          ["h_c_n", "niche", "multimodal chat", 0], ["h_c_m", "mechanism", "visual encoder", 1],
          ["h_c_e", "mechanism", "instruction-tuned LLM", 0], ["h_c_d", "delta", "vision–language projection", 0] ] },
      ],
      edges: [["h_a_m", "h_c_m", "Inherited"], ["h_b_m", "h_c_e", "External"],
              ["h_a_n", "h_c_n", "Mutated"], [null, "h_c_d", "Novel"]],
    },
    speciation: {
      dynamics: "Speciation", lineage: true, layout: "chain2",
      pair: "Faster R-CNN → DETR",
      crit: "Same or nearby niche, but the predecessor's driver mechanism is replaced by a new lineage-forming mechanism.",
      genomes: [
        { id: "g0", title: "Faster R-CNN", year: "2015", units: [
          ["s_a_n", "niche", "object detection", 0], ["s_a_m", "mechanism", "CNN region proposals (RPN)", 1],
          ["s_a_l", "limitation", "hand-designed anchors / NMS", 0], ["s_a_o", "observation", "strong accuracy", 0] ] },
        { id: "g1", title: "DETR", year: "2020", units: [
          ["s_b_n", "niche", "object detection", 0], ["s_b_m", "mechanism", "Transformer set prediction", 1],
          ["s_b_d", "delta", "bipartite matching, no NMS", 0], ["s_b_c", "claim", "end-to-end detection", 0] ] },
      ],
      edges: [["s_a_n", "s_b_n", "Inherited"], ["s_a_m", null, "Lost"], [null, "s_b_m", "Novel"],
              ["s_a_l", null, "Lost"], [null, "s_b_d", "Novel"]],
      driverReplaced: true,
    },
    nicheCompetition: {
      dynamics: "Niche Competition", lineage: false, layout: "coloc",
      pair: "Faster R-CNN  vs.  YOLO",
      crit: "Same ecology or problem niche, but no driver inheritance — competing, not descent.",
      genomes: [
        { id: "g0", title: "Faster R-CNN", year: "2015", units: [
          ["n_a_n", "niche", "object detection", 0], ["n_a_m", "mechanism", "two-stage region proposals", 1],
          ["n_a_o", "observation", "high accuracy", 0] ] },
        { id: "g1", title: "YOLO", year: "2016", units: [
          ["n_b_n", "niche", "object detection", 0], ["n_b_m", "mechanism", "one-stage regression", 1],
          ["n_b_o", "observation", "real-time speed", 0] ] },
      ],
      edges: [], sharedNiche: ["n_a_n", "n_b_n"],
    },
    isolation: {
      dynamics: "Isolation", lineage: false, layout: "coloc",
      pair: "BERT  vs.  YOLO",
      crit: "Neither shared ecology nor driver inheritance.",
      genomes: [
        { id: "g0", title: "BERT", year: "2018", units: [
          ["i_a_n", "niche", "language understanding", 0], ["i_a_m", "mechanism", "masked-LM Transformer", 1],
          ["i_a_c", "claim", "pretrain + fine-tune", 0] ] },
        { id: "g1", title: "YOLO", year: "2016", units: [
          ["i_b_n", "niche", "object detection", 0], ["i_b_m", "mechanism", "one-stage detection", 1],
          ["i_b_c", "claim", "real-time detector", 0] ] },
      ],
      edges: [],
    },
  };
  const ORDER = ["mutation", "adaptiveRadiation", "hybridization", "speciation", "nicheCompetition", "isolation"];

  const W = 1160, H = 520, R_RING = 70;
  // canonical role angle for a predecessor genome (0 = toward the partner / gap on the right);
  // a successor genome is the horizontal mirror (a -> π - a), so partner units line up.
  const ROLE_ANG = { mechanism: 0, niche: -0.92, observation: 0.92, delta: 1.62, limitation: 2.42, claim: -2.42 };
  const LAYOUT = {
    chain2: [ { cx: 0.28, cy: 0.52, mirror: false }, { cx: 0.72, cy: 0.52, mirror: true } ],
    hybrid: [ { cx: 0.24, cy: 0.31, mirror: false }, { cx: 0.24, cy: 0.73, mirror: false }, { cx: 0.75, cy: 0.52, mirror: true } ],
    coloc:  [ { cx: 0.28, cy: 0.52, mirror: false }, { cx: 0.72, cy: 0.52, mirror: true } ],
  };

  const svg = d3.select(svgEl).attr("viewBox", `0 0 ${W} ${H}`).attr("preserveAspectRatio", "xMidYMid meet");
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
  const gPlot = svg.append("g");

  const _cv = document.createElement("canvas").getContext("2d");
  const measure = (s, fs, fw) => { _cv.font = `${fw || 400} ${fs}px Inter, Arial`; return _cv.measureText(s).width; };
  const uR = (d) => (d.driver ? 14 : 9);

  let tip = document.getElementById("gv-tip");
  if (!tip) { tip = document.createElement("div"); tip.id = "gv-tip"; tip.className = "gv-tip"; document.body.appendChild(tip); }

  function render(key) {
    const ex = EXAMPLES[key];
    const pos = LAYOUT[ex.layout];
    const byId = {};
    // place units by role angle (successor mirrored) so partner units align across the gap
    ex.genomes.forEach((g, gi) => {
      const p = pos[gi], cx = p.cx * W, cy = p.cy * H, seen = {};
      g._x = cx; g._y = cy;
      g.units.forEach((u) => {
        let a = ROLE_ANG[u[1]] != null ? ROLE_ANG[u[1]] : 0;
        const k = (seen[u[1]] = (seen[u[1]] || 0) + 1); // fan out duplicate roles (e.g. two mechanisms)
        if (k > 1) a += (k - 1) * 0.62 * (p.mirror ? -1 : 1);
        if (p.mirror) a = Math.PI - a;
        byId[u[0]] = { id: u[0], type: u[1], text: u[2], driver: !!u[3], x: cx + Math.cos(a) * R_RING, y: cy + Math.sin(a) * R_RING, hub: { x: cx, y: cy } };
      });
    });
    const halfMarks = [], diffLinks = [];
    ex.edges.forEach(([s, t, fate]) => {
      if (fate === "Lost") return halfMarks.push({ id: s, fate });
      if (fate === "Novel") return halfMarks.push({ id: t, fate });
      if (fate === "External") return halfMarks.push({ id: t, fate, edge: [s, t] });
      diffLinks.push({ s, t, fate });
    });
    if (ex.driverReplaced) { const dv = ex.genomes.at(-1).units.find((u) => u[3]); if (dv) halfMarks.push({ id: dv[0], fate: "NewDriver" }); }
    const extIds = new Set(halfMarks.filter((m) => m.fate === "External").map((m) => m.id));

    gPlot.selectAll("*").remove();
    const gHull = gPlot.append("g"), gShare = gPlot.append("g"), gMember = gPlot.append("g"),
          gSpine = gPlot.append("g"), gDiff = gPlot.append("g"), gHalf = gPlot.append("g"),
          gNode = gPlot.append("g"), gLabel = gPlot.append("g");

    // genome capsules
    gHull.selectAll("circle").data(ex.genomes).join("circle").attr("cx", (d) => d._x).attr("cy", (d) => d._y)
      .attr("r", R_RING + 34).attr("fill", "#000").attr("fill-opacity", 0.03).attr("stroke", "#b8bec9")
      .attr("stroke-opacity", 0.35).attr("stroke-width", 1.2).attr("stroke-dasharray", "2 5");

    // shared-ecology link (co-location, e.g. Niche Competition)
    if (ex.sharedNiche) {
      const a = byId[ex.sharedNiche[0]], b = byId[ex.sharedNiche[1]];
      gShare.append("line").attr("x1", a.x).attr("y1", a.y).attr("x2", b.x).attr("y2", b.y)
        .attr("stroke", MUTED).attr("stroke-width", 1.6).attr("stroke-dasharray", "4 5").attr("stroke-opacity", 0.7);
      gShare.append("text").attr("x", (a.x + b.x) / 2).attr("y", (a.y + b.y) / 2 - 8).attr("text-anchor", "middle")
        .attr("font-size", 11).attr("font-weight", 700).attr("fill", MUTED).attr("paint-order", "stroke").attr("stroke", "#fffdf7").attr("stroke-width", 3).text("shared ecology, no descent");
    }

    // member spokes
    const members = [];
    ex.genomes.forEach((g) => g.units.forEach((u) => members.push({ a: byId[u[0]], h: g })));
    gMember.selectAll("line").data(members).join("line").attr("x1", (d) => d.h._x).attr("y1", (d) => d.h._y)
      .attr("x2", (d) => d.a.x).attr("y2", (d) => d.a.y).attr("stroke", "#cdcabf").attr("stroke-opacity", 0.6).attr("stroke-width", 1);

    function pathD(s, t) {
      const a = byId[s], b = byId[t];
      const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2 - Math.min(70, Math.abs(b.x - a.x) * 0.24);
      const dS = Math.hypot(mx - a.x, my - a.y) || 1, dT = Math.hypot(mx - b.x, my - b.y) || 1;
      const sx = a.x + ((mx - a.x) / dS) * (uR(a) + 5), sy = a.y + ((my - a.y) / dS) * (uR(a) + 5);
      const ex2 = b.x + ((mx - b.x) / dT) * (uR(b) + 8), ey = b.y + ((my - b.y) / dT) * (uR(b) + 8);
      return `M${sx},${sy} Q${mx},${my} ${ex2},${ey}`;
    }
    // driver backbone glow (inherited driver→driver)
    const backbone = diffLinks.filter((l) => l.fate === "Inherited" && byId[l.s].driver && byId[l.t].driver);
    gSpine.selectAll("path").data(backbone).join("path").attr("d", (d) => pathD(d.s, d.t)).attr("fill", "none")
      .attr("stroke", FATE.Inherited).attr("stroke-opacity", 0.22).attr("stroke-width", 14).attr("stroke-linecap", "round");
    // diff edges
    gDiff.selectAll("path").data(diffLinks).join("path").attr("d", (d) => pathD(d.s, d.t)).attr("fill", "none")
      .attr("stroke", (d) => FATE[d.fate]).attr("stroke-width", (d) => (backbone.includes(d) ? 3.4 : 2.2)).attr("stroke-opacity", 0.92)
      .attr("stroke-dasharray", (d) => (d.fate === "Mutated" ? "7 4" : "none")).attr("marker-end", (d) => "url(#gv-arw-" + d.fate + ")");
    // external edge (import into a driver unit)
    halfMarks.filter((m) => m.fate === "External" && m.edge).forEach((m) => {
      gDiff.append("path").attr("d", pathD(m.edge[0], m.edge[1])).attr("fill", "none")
        .attr("stroke", FATE.External).attr("stroke-width", 2.2).attr("stroke-opacity", 0.92).attr("marker-end", "url(#gv-arw-External)");
    });

    // half marks
    const hs = gHalf.selectAll("g").data(halfMarks.filter((m) => m.fate !== "External")).join("g")
      .attr("transform", (m) => { const u = byId[m.id]; const dx = u.x - u.hub.x, dy = u.y - u.hub.y, d = Math.hypot(dx, dy) || 1; return `translate(${u.x + (dx / d) * 13},${u.y + (dy / d) * 13})`; });
    hs.each(function (m) {
      const g = d3.select(this);
      if (m.fate === "NewDriver") {
        g.append("circle").attr("r", 9).attr("fill", "none").attr("stroke", FATE.Novel).attr("stroke-width", 2).attr("stroke-dasharray", "2 2");
        return;
      }
      const c = m.fate === "Lost" ? FATE.Lost : FATE.Novel, sym = m.fate === "Lost" ? "✕" : "+";
      g.append("circle").attr("r", 8.5).attr("fill", "#fff").attr("stroke", c).attr("stroke-width", 1.8);
      g.append("text").attr("text-anchor", "middle").attr("dy", "0.34em").attr("font-size", 12).attr("font-weight", 800).attr("fill", c).text(sym);
    });

    // hubs
    const hub = gNode.selectAll("g.hub").data(ex.genomes).join("g").attr("transform", (d) => `translate(${d._x},${d._y})`);
    hub.each(function (d) {
      const g = d3.select(this), w = Math.max(96, measure(d.title, 14, 800) + 36), h = 42;
      g.append("rect").attr("x", -w / 2).attr("y", -h / 2).attr("width", w).attr("height", h).attr("rx", 11)
        .attr("fill", "url(#gv-hub)").attr("filter", "url(#gv-glow)").attr("stroke", "#fff").attr("stroke-width", 2);
      g.append("text").attr("text-anchor", "middle").attr("dy", "-1").attr("fill", "#fff").attr("font-size", 14).attr("font-weight", 800).text(d.title);
      g.append("text").attr("text-anchor", "middle").attr("dy", "13").attr("fill", "#c3cdda").attr("font-size", 9.5).text(d.year);
    });

    // unit nodes
    const symGen = (t, a) => d3.symbol().type(SHAPE[t]).size(a)();
    const allUnits = Object.values(byId);
    const uSel = gNode.selectAll("g.unit").data(allUnits, (d) => d.id).join("g").attr("transform", (d) => `translate(${d.x},${d.y})`).style("cursor", "default");
    uSel.append("path").attr("d", (d) => symGen(d.type, d.driver ? 360 : 150)).attr("fill", (d) => TYPE[d.type])
      .attr("stroke", "#fff").attr("stroke-width", (d) => (d.driver ? 2.2 : 1.6)).attr("filter", (d) => (d.driver ? "url(#gv-glow)" : null));

    // labels (content + role)
    const marked = new Set(halfMarks.map((m) => m.id));
    gLabel.selectAll("text.u").data(allUnits, (d) => d.id).join("text").attr("class", "u")
      .attr("paint-order", "stroke").attr("stroke", "#fffdf7").attr("stroke-width", 3).each(function (d) {
        const s = d3.select(this);
        const roleTxt = extIds.has(d.id) ? d.type + " · external" : d.type;
        const roleFill = extIds.has(d.id) ? FATE.External : TYPE[d.type];
        // driver labels sit above the node (centered) so opposing drivers never collide in the gap
        if (d.driver) {
          s.attr("text-anchor", "middle");
          s.append("tspan").attr("x", d.x).attr("y", d.y - uR(d) - 16).attr("font-size", 12).attr("font-weight", 800).attr("fill", INK).text(d.text);
          s.append("tspan").attr("x", d.x).attr("y", d.y - uR(d) - 4).attr("font-size", 8.4).attr("font-weight", 700).attr("fill", roleFill).text(roleTxt);
          return;
        }
        const dx = d.x - d.hub.x, dy = d.y - d.hub.y, dd = Math.hypot(dx, dy) || 1;
        const off = marked.has(d.id) ? 28 : uR(d) + 11, lx = d.x + (dx / dd) * off, ly = d.y + (dy / dd) * off, up = dy < -4;
        s.attr("text-anchor", d.x < d.hub.x - 6 ? "end" : d.x > d.hub.x + 6 ? "start" : "middle");
        s.append("tspan").attr("x", lx).attr("y", up ? ly - 11 : ly).attr("font-size", 11).attr("font-weight", 800).attr("fill", INK).text(d.text);
        s.append("tspan").attr("x", lx).attr("y", up ? ly - 0.5 : ly + 12).attr("font-size", 8.4).attr("font-weight", 700).attr("fill", roleFill).text(roleTxt);
      });

    // tooltip
    uSel.on("mousemove", (e, d) => {
      tip.style.display = "block"; tip.style.left = e.clientX + 14 + "px"; tip.style.top = e.clientY + 14 + "px";
      tip.innerHTML = `<span class="gv-ty" style="background:${TYPE[d.type]}">${d.type}${d.driver ? " ★ driver" : ""}</span><b>${d.text}</b>`;
    }).on("mouseleave", () => (tip.style.display = "none"));

    // info panel
    const info = document.getElementById("genomeInfo");
    if (info) {
      const lin = ex.lineage
        ? '<span class="gv-badge gv-yes">lineage · descent</span>'
        : '<span class="gv-badge gv-no">co-location · no descent</span>';
      info.innerHTML = `<div class="gv-info-top"><b>${ex.dynamics}</b> ${lin} <span class="gv-pair">${ex.pair}</span></div><p>${ex.crit}</p>`;
    }
  }

  // ---- selector ----
  const sel = document.getElementById("genomeSelect");
  if (sel) {
    sel.innerHTML = "";
    ORDER.forEach((key, i) => {
      const ex = EXAMPLES[key];
      const b = document.createElement("button");
      b.type = "button"; b.className = "gv-tab" + (ex.lineage ? " gv-tab-lin" : " gv-tab-col");
      b.dataset.key = key; b.textContent = ex.dynamics;
      b.addEventListener("click", () => {
        sel.querySelectorAll(".gv-tab").forEach((t) => t.classList.remove("is-active"));
        b.classList.add("is-active"); render(key);
      });
      sel.appendChild(b);
    });
  }

  // ---- static legend ----
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
  }

  // initial render + active tab
  render("adaptiveRadiation");
  if (sel) { const b = sel.querySelector('[data-key="adaptiveRadiation"]'); if (b) b.classList.add("is-active"); }
})();
