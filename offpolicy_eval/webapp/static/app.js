/* 离线策略评估台前端：原生 JS，无外部依赖，图表为手写 SVG。 */
"use strict";

const $ = (sel) => document.querySelector(sel);
const api = async (path, opts) => {
  const r = await fetch(path, opts);
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
  return data;
};
const post = (path, body) => api(path, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});
const fmt = (x, d = 3) =>
  x === null || x === undefined || Number.isNaN(x) ? "—" : Number(x).toFixed(d);
const pct = (x) => (x === null || x === undefined ? "—" : (x * 100).toFixed(1) + "%");
const esc = (s) => String(s ?? "").replace(/[&<>"]/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const METHOD_LABELS = {
  ips: "IPS 倾向加权", cips: "截断 IPS", snips: "SNIPS",
  replay_all: "集合匹配重放", replay_any: "主动作重放",
  naive_zero: "错误对照(未曝光记零)",
};
const METHOD_COLORS = {
  ips: "#2b5cd7", cips: "#7c3aed", snips: "#0e7490",
  replay_all: "#b45309", replay_any: "#a16207", naive_zero: "#b91c1c",
};
const state = { tasks: [], strategies: [], versions: [] };

/* ---------------- SVG 小图表 ---------------- */
function barChart(series, labels, opts = {}) {
  // series: [{name, color, values:[]}]；labels: x 轴类别
  const W = opts.width || 760, H = 260, pad = { l: 52, r: 12, t: 16, b: 64 };
  const all = series.flatMap((s) => s.values.filter((v) => !Number.isNaN(v)));
  const vmax = Math.max(...all, 0) * 1.1;
  const vmin = Math.min(...all, 0);
  const n = labels.length, ns = series.length;
  const gw = (W - pad.l - pad.r) / n;
  const bw = Math.min(26, (gw - 8) / ns);
  const y = (v) => pad.t + (H - pad.t - pad.b) * (1 - (v - vmin) / (vmax - vmin || 1));
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => vmin + t * (vmax - vmin));
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img">`;
  for (const tv of ticks) {
    svg += `<line x1="${pad.l}" x2="${W - pad.r}" y1="${y(tv)}" y2="${y(tv)}"
      stroke="#e3e6eb"/><text x="${pad.l - 6}" y="${y(tv) + 4}"
      text-anchor="end">${tv.toFixed(2)}</text>`;
  }
  labels.forEach((lab, i) => {
    const gx = pad.l + i * gw + gw / 2;
    series.forEach((s, j) => {
      const v = s.values[i];
      if (v === undefined || Number.isNaN(v)) return;
      const x = gx - (ns * bw) / 2 + j * bw;
      svg += `<rect x="${x}" y="${y(Math.max(v, 0))}" width="${bw - 2}"
        height="${Math.abs(y(v) - y(0))}" fill="${s.color}">
        <title>${esc(lab)} · ${s.name}: ${fmt(v)}</title></rect>`;
    });
    svg += `<text x="${gx}" y="${H - pad.b + 16}" text-anchor="middle"
      transform="rotate(-24 ${gx} ${H - pad.b + 16})">${esc(lab)}</text>`;
  });
  svg += "</svg>";
  const legend = `<div class="legend">${series.map((s) =>
    `<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join("")}</div>`;
  return legend + svg;
}

function histogram(values, bins = 20, color = "#2b5cd7") {
  if (!values.length) return "<p class='muted'>无数据</p>";
  const lo = Math.min(...values), hi = Math.max(...values);
  const width = (hi - lo) / bins || 1;
  const counts = Array(bins).fill(0);
  values.forEach((v) => {
    const b = Math.min(bins - 1, Math.floor((v - lo) / width));
    counts[b]++;
  });
  const W = 760, H = 220, pad = { l: 48, r: 12, t: 14, b: 40 };
  const cmax = Math.max(...counts);
  const bw = (W - pad.l - pad.r) / bins;
  const y = (c) => pad.t + (H - pad.t - pad.b) * (1 - c / cmax);
  let svg = `<svg viewBox="0 0 ${W} ${H}">`;
  counts.forEach((c, i) => {
    const x = pad.l + i * bw;
    svg += `<rect x="${x + 1}" y="${y(c)}" width="${bw - 2}"
      height="${H - pad.b - y(c)}" fill="${color}">
      <title>区间 ${fmt(lo + i * width, 2)} ~ ${fmt(lo + (i + 1) * width, 2)}：${c} 个</title></rect>`;
  });
  svg += `<text x="${pad.l}" y="${H - 8}">${fmt(lo, 2)}</text>
    <text x="${W - pad.r}" y="${H - 8}" text-anchor="end">${fmt(hi, 2)}</text></svg>`;
  return svg;
}

/* ---------------- 页签切换 ---------------- */
document.querySelectorAll(".tabs button").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((b) =>
      b.classList.toggle("active", b === btn));
    document.querySelectorAll(".tab").forEach((t) =>
      t.classList.toggle("active", t.id === "tab-" + btn.dataset.tab));
    if (btn.dataset.tab === "compare") loadCompare();
    if (btn.dataset.tab === "reports") loadReports();
    if (btn.dataset.tab === "tasks") loadTasks();
    if (btn.dataset.tab === "contrib") initContribMethods();
    if (btn.dataset.tab === "support" && !$("#sp-task").innerHTML) loadTasks();
  });
});

async function initContribMethods() {
  if (!state.tasks.length) await loadTasks();
  const sel = $("#ct-task");
  if (!sel.value && sel.options.length) sel.value = sel.options[0].value;
  if (sel.value) {
    const t = state.tasks.find((x) => x.task_id === sel.value);
    const methods = t ? t.config.methods : Object.keys(METHOD_LABELS);
    $("#ct-method").innerHTML = methods.map((m) =>
      `<option value="${m}">${METHOD_LABELS[m] || m}</option>`).join("");
  }
}

/* ---------------- ① 总览 ---------------- */
async function loadOverview() {
  const ov = await api("/api/overview");
  const c = ov.counts;
  const kpis = [
    ["查询数", c.queries], ["候选曝光行", c.exposures],
    ["实际曝光", c.exposed], ["点击", c.clicks],
    ["成交", c.orders], ["退款", c.refunds],
  ];
  $("#ov-kpis").innerHTML = kpis.map(([l, v]) =>
    `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`
  ).join("") + `<div class="kpi"><div class="v">${esc(ov.scenario)}</div>
    <div class="l">合成场景</div></div>`;
  $("#ov-versions").innerHTML = `<tr><th>版本</th><th>cutoff</th><th class="num">查询</th>
    <th class="num">曝光</th><th class="num">点击</th><th class="num">成交</th>
    <th class="num">退款</th><th>data_hash</th></tr>` +
    ov.data_versions.map((d) => `<tr><td>v${d.data_version}</td>
      <td class="mono">${esc(d.cutoff)}</td><td class="num">${d.n_queries}</td>
      <td class="num">${d.n_exposures}</td><td class="num">${d.n_clicks}</td>
      <td class="num">${d.n_orders}</td><td class="num">${d.n_refunds}</td>
      <td class="mono">${esc(d.data_hash)}</td></tr>`).join("");
  state.strategies = await api("/api/strategies");
  state.versions = ov.data_versions;
  $("#ov-strategies").innerHTML = `<tr><th>策略</th><th>版本</th><th>名称</th><th>说明</th></tr>` +
    state.strategies.map((s) => `<tr><td class="mono">${esc(s.strategy_id)}</td>
      <td class="mono">${esc(s.version)}</td><td>${esc(s.label)}</td>
      <td class="muted">${esc(s.description)}</td></tr>`).join("");
  fillSelects();
}

function fillSelects() {
  const stratOpts = state.strategies.map((s) =>
    `<option value="${s.strategy_id}@${s.version}">${esc(s.label)}
      (${s.strategy_id}@${s.version})</option>`).join("");
  $("#f-strategy").innerHTML = stratOpts;
  $("#ct-task").innerHTML = $("#sp-task").innerHTML =
    $("#rp-task").innerHTML = state.tasks.map((t) =>
      `<option value="${t.task_id}">${esc(t.name)} [${t.status}]</option>`).join("");
  $("#f-dv").innerHTML = state.versions.map((d) =>
    `<option value="${d.data_version}">v${d.data_version} · ${esc(d.cutoff)}</option>`)
    .join("");
  $("#f-methods").innerHTML = Object.keys(METHOD_LABELS).map((m) =>
    `<label><input type="checkbox" name="method" value="${m}" checked>
     ${METHOD_LABELS[m]}</label>`).join("");
}

/* ---------------- ② 任务 ---------------- */
async function loadTasks() {
  state.tasks = await api("/api/tasks");
  // 仅刷新任务相关下拉，保留 fillSelects 中已填充的策略/数据版本/方法选项
  const taskOpts = state.tasks.map((t) =>
    `<option value="${t.task_id}">${esc(t.name)} [${t.status}]</option>`).join("");
  ["#ct-task", "#sp-task", "#rp-task"].forEach((sel) => {
    $(sel).innerHTML = taskOpts;
  });
  const table = $("#task-table");
  const total = (t) => t.total_queries || "—";
  $("#task-table").innerHTML = `<tr><th>任务</th><th>状态</th><th>策略</th>
    <th>数据版本</th><th>口径</th><th class="num">进度</th>
    <th>comparability_key</th><th>操作</th></tr>` +
    state.tasks.map((t) => {
      const s = t.config.strategies[0];
      const cal = `${t.config.attribution_window_hours}h / ${t.config.refund_policy}
        / ${t.config.reward_metric} / clip=${t.config.clip_threshold}`;
      return `<tr><td class="mono">${esc(t.task_id)}<br>
          <span class="muted">${esc(t.name)}</span></td>
        <td><span class="pill ${t.status}">${t.status}</span></td>
        <td class="mono">${esc(s.strategy_id)}@${esc(s.version)}</td>
        <td>v${t.data_version}</td><td class="mono small">${esc(cal)}</td>
        <td class="num">${t.n_done}/${total(t)}</td>
        <td class="mono">${esc(t.comparability_key)}</td>
        <td>${t.status !== "completed"
          ? `<button class="action" onclick="runTask('${t.task_id}')">执行/恢复</button>`
          : `<button class="ghost" onclick="viewTask('${t.task_id}')">查看结果</button>`}
        </td></tr>`;
    }).join("");
}

window.runTask = async (id) => {
  try {
    const r = await post(`/api/tasks/${id}/run`, {});
    alert(`任务 ${id} → ${r.status}`);
    loadTasks();
  } catch (e) { alert(e.message); }
};

window.viewTask = (id) => {
  $("#ct-task").value = id;
  document.querySelector('[data-tab="contrib"]').click();
};

$("#f-create").addEventListener("click", async () => {
  const [sid, ver] = $("#f-strategy").value.split("@");
  const config = {
    strategies: [{ strategy_id: sid, version: ver }],
    methods: [...document.querySelectorAll('input[name=method]:checked')].map((e) => e.value),
    clip_threshold: Number($("#f-clip").value),
    n_bootstrap: Number($("#f-boot").value),
    bootstrap_seed: 7,
    attribution_window_hours: Number($("#f-window").value),
    refund_policy: $("#f-refund").value,
    reward_metric: $("#f-metric").value,
    top_k: 10,
    replay_k: Number($("#f-replayk").value),
  };
  try {
    const r = await post("/api/tasks", {
      name: $("#f-name").value || `${sid}@${ver}`,
      data_version: Number($("#f-dv").value), config,
    });
    alert("已创建 " + r.task_id + "，口径已冻结");
    loadTasks();
  } catch (e) { alert(e.message); }
});

/* ---------------- ③ 同口径对比 ---------------- */
async function loadCompare() {
  state.tasks = state.tasks.length ? state.tasks : await api("/api/tasks");
  const groups = [...new Set(state.tasks.map((t) => t.comparability_key))];
  $("#cmp-group").innerHTML = groups.map((g) =>
    `<option value="${g}">${g}</option>`).join("") ||
    "<option value=''>暂无任务</option>";
  if (!groups.length) {
    $("#cmp-table").innerHTML = "<p class='muted'>请先在②创建并完成任务</p>";
    return;
  }
  await renderCompare();
}
$("#cmp-refresh").addEventListener("click", renderCompare);
$("#cmp-group").addEventListener("change", renderCompare);

async function renderCompare() {
  const group = $("#cmp-group").value;
  if (!group) return;
  const data = await api(`/api/compare?group=${encodeURIComponent(group)}`);
  const tasks = data.tasks;
  const cfg = tasks[0].config;
  $("#cmp-meta").innerHTML = `<div class="card mono">
    口径组 ${esc(group)}<br>
    归因窗口=${cfg.attribution_window_hours}h · 退款=${esc(cfg.refund_policy)} ·
    指标=${esc(cfg.reward_metric)} · 截断=${cfg.clip_threshold} ·
    bootstrap=${cfg.n_bootstrap} · 数据版本=v${tasks[0].data_version}</div>`;

  const methods = [...new Set(tasks.flatMap((t) => t.results.map((r) => r.method)))];
  const labels = tasks.map((t) => t.config.strategies[0].strategy_id +
    "@" + t.config.strategies[0].version);
  const series = methods.map((m) => ({
    name: METHOD_LABELS[m] || m, color: METHOD_COLORS[m] || "#666",
    values: tasks.map((t) => {
      const r = t.results.find((x) => x.method === m);
      return r ? r.estimate : NaN;
    }),
  }));
  // ground truth 部署值作为参照线（仅合成数据可得）
  const gt = tasks.map((t) => {
    const ips = t.results.find((x) => x.method === "ips");
    return ips ? ips.diagnostics.gt_deploy : NaN;
  });
  series.push({ name: "Ground Truth 部署值(仅合成数据)", color: "#15803d", values: gt });
  $("#cmp-chart").innerHTML = barChart(series, labels);

  $("#cmp-table").innerHTML = `<tr><th>策略</th>${methods.map((m) =>
    `<th class="num">${METHOD_LABELS[m] || m}</th>`).join("")}
    <th class="num">GT 部署值</th><th class="num">GT 可识别值</th></tr>` +
    tasks.map((t) => {
      const cell = (m) => {
        const r = t.results.find((x) => x.method === m);
        if (!r) return "<td class='num muted'>—</td>";
        return `<td class="num">${fmt(r.estimate, 2)}<br>
          <span class="muted">SE ${fmt(Math.sqrt(r.variance || 0), 2)}</span></td>`;
      };
      const ips = t.results.find((x) => x.method === "ips");
      return `<tr><td class="mono">${esc(t.config.strategies[0].strategy_id)}@${
        esc(t.config.strategies[0].version)}</td>
        ${methods.map(cell).join("")}
        <td class="num">${ips ? fmt(ips.diagnostics.gt_deploy, 2) : "—"}</td>
        <td class="num">${ips ? fmt(ips.diagnostics.gt_identifiable, 2) : "—"}</td></tr>`;
    }).join("");

  // 前提与诊断告警
  const caveats = [];
  const anyUnsupported = tasks.some((t) => t.results.some((r) =>
    r.diagnostics.n_unsupported_queries > 0));
  if (anyUnsupported) {
    const rows = tasks.map((t) => {
      const r = t.results.find((x) => x.method === "ips");
      return `${t.config.strategies[0].strategy_id}@${t.config.strategies[0].version}: ` +
        `${r.diagnostics.n_unsupported_queries} 个查询无支持`;
    });
    caveats.push(["bad", "存在无支持区域：目标策略选择了日志中曝光概率为 0 的商品，" +
      "IPS 只能估计可识别值，与部署值的缺口不可由日志修复。", rows]);
  }
  const highW = tasks.map((t) => t.results.find((x) => x.method === "ips"))
    .filter(Boolean).find((r) => r.diagnostics.weight_p99 > 50);
  if (highW) {
    caveats.push(["", `权重不稳定：p99 IPS 权重=${fmt(highW.diagnostics.weight_p99, 1)}，` +
      `有效样本量 ESS=${fmt(highW.diagnostics.ess, 1)}，估计方差可能主导结论。`]);
  }
  const replay = tasks.map((t) => t.results.find((x) => x.method === "replay_all"))
    .find(Boolean);
  if (replay && replay.diagnostics.coverage < 0.1) {
    caveats.push(["", `匹配重放覆盖率仅 ${pct(replay.diagnostics.coverage)}，` +
      "可用查询太少，重放均值方差大，不适合作为主要结论。"]);
  }
  const naive = tasks.map((t) => t.results.find((x) => x.method === "naive_zero"))
    .find(Boolean);
  if (naive) {
    caveats.push(["bad", "错误对照（未曝光记零）系统性低于 IPS 与 GT——" +
      "未曝光不代表用户不会点击，该口径禁止用于决策。"]);
  }
  caveats.push(["ok", "本对比仅为离线日志估计；上线收益需通过 A/B 实验验证，" +
    "本页面不提供上线效果承诺。"]);
  $("#cmp-caveats").innerHTML = caveats.map(([cls, title, rows]) =>
    `<div class="caveat ${cls}"><strong>${esc(title)}</strong>${
      rows ? "<br>" + rows.map(esc).join("<br>") : ""}</div>`).join("");
}

/* ---------------- ④ 逐查询贡献 ---------------- */
$("#ct-task").addEventListener("change", async (e) => {
  const t = state.tasks.find((x) => x.task_id === e.target.value);
  $("#ct-method").innerHTML = (t ? t.config.methods : Object.keys(METHOD_LABELS))
    .map((m) => `<option value="${m}">${METHOD_LABELS[m] || m}</option>`).join("");
});
$("#ct-load").addEventListener("click", async () => {
  const tid = $("#ct-task").value, method = $("#ct-method").value;
  if (!tid) return;
  const rows = await api(`/api/tasks/${tid}/contribs?method=${method}&limit=300`);
  const unsupported = rows.filter((r) => !r.supported).length;
  $("#ct-summary").innerHTML = `<div class="kpi-row">
    <div class="kpi ${unsupported ? "warn" : ""}"><div class="v">${unsupported}</div>
      <div class="l">无支持查询数（前300行内）</div></div>
    <div class="kpi"><div class="v">${fmt(rows.reduce((s, r) => s + r.contribution, 0) /
      Math.max(1, rows.length), 3)}</div><div class="l">平均贡献</div></div>
    <div class="kpi"><div class="v">${fmt(Math.max(...rows.map((r) =>
      Math.abs(r.contribution)), 0), 2)}</div><div class="l">最大绝对贡献（高影响查询）</div></div>
  </div>`;
  $("#ct-chart").innerHTML = histogram(rows.map((r) => r.contribution), 24,
    METHOD_COLORS[method] || "#2b5cd7");
  $("#ct-table").innerHTML = `<tr><th>查询</th><th>查询文本</th><th>时间</th>
    <th class="num">${METHOD_LABELS[method] || method} 贡献</th><th>支持性</th></tr>` +
    rows.map((r) => `<tr><td class="mono">${esc(r.query_id)}</td>
      <td>${esc(r.query_text)}</td><td class="mono">${esc(r.ts_log)}</td>
      <td class="num">${fmt(r.contribution, 3)}</td>
      <td class="${r.supported ? "tag-supported" : "tag-unsupported"}">
      ${r.supported ? "有支持" : "无支持(p=0)"}</td></tr>`).join("");
});

/* ---------------- ⑤ 支持区域 / 权重 ---------------- */
$("#sp-load").addEventListener("click", async () => {
  const tid = $("#sp-task").value;
  if (!tid) return;
  const [detail, weights] = await Promise.all([
    api(`/api/tasks/${tid}`),
    api(`/api/tasks/${tid}/weights?limit=5000`),
  ]);
  const ips = detail.results.find((r) => r.method === "ips");
  const d = ips ? ips.diagnostics : {};
  const kpis = [
    ["无支持查询", d.n_unsupported_queries, d.unsupported_fraction > 0 ? "bad" : ""],
    ["无支持占比", pct(d.unsupported_fraction), ""],
    ["权重 p99", fmt(d.weight_p99, 1), d.weight_p99 > 50 ? "warn" : ""],
    ["最大权重", fmt(d.weight_max, 1), d.weight_max > 100 ? "warn" : ""],
    ["ESS", fmt(d.ess, 1), d.ess < detail.total_queries * 0.3 ? "warn" : ""],
    ["被截断权重占比", pct(d.clipped_fraction), ""],
  ];
  $("#sp-kpis").innerHTML = kpis.map(([l, v, cls]) =>
    `<div class="kpi ${cls}"><div class="v">${v}</div><div class="l">${l}</div></div>`
  ).join("");
  const wvals = weights.map((w) => w.ips_weight).filter((v) => v != null);
  $("#sp-dist").innerHTML = "<h3>IPS 权重分布（1/p_recorded）</h3>" +
    histogram(wvals, 25, "#7c3aed");
  const top = weights.filter((w) => w.ips_weight != null)
    .sort((a, b) => b.ips_weight - a.ips_weight).slice(0, 50);
  $("#sp-table").innerHTML = `<tr><th>查询</th><th>商品</th>
    <th class="num">记录倾向 p</th><th class="num">IPS 权重</th></tr>` +
    top.map((w) => `<tr><td class="mono">${esc(w.query_id)}</td>
      <td class="mono">${esc(w.item_id)}</td>
      <td class="num">${fmt(w.p_recorded, 4)}</td>
      <td class="num ${w.ips_weight > 50 ? "tag-unsupported" : ""}">
      ${fmt(w.ips_weight, 1)}</td></tr>`).join("");
});

/* ---------------- ⑥ 报告 ---------------- */
async function loadReports() {
  const state2 = await api("/api/tasks");
  $("#rp-task").innerHTML = state2.map((t) =>
    `<option value="${t.task_id}">${esc(t.name)} [${t.status}]</option>`).join("");
  const reports = await api("/api/reports");
  $("#rp-table").innerHTML = `<tr><th>报告</th><th>任务</th><th>保存时间</th>
    <th>备注</th><th class="num">data_hash</th><th class="num">config_hash</th></tr>` +
    reports.map((r) => `<tr><td class="mono">${esc(r.report_id)}</td>
      <td class="mono">${esc(r.task_id)}</td><td>${esc(r.created_at)}</td>
      <td>${esc(r.note)}</td><td class="mono">${esc(r.data_hash)}</td>
      <td class="mono">${esc(r.config_hash)}</td></tr>`).join("") ||
    "<p class='muted'>暂无保存的报告</p>";
}
$("#rp-save").addEventListener("click", async () => {
  try {
    const r = await post("/api/reports", {
      task_id: $("#rp-task").value, note: $("#rp-note").value,
    });
    alert("报告已保存（不可修改）：" + r.report_id);
    loadReports();
  } catch (e) { alert(e.message); }
});

/* ---------------- 启动 ---------------- */
loadOverview().then(loadTasks).catch((e) =>
  $("#ov-kpis").innerHTML = `<div class="caveat bad">加载失败：${esc(e.message)}。
    请先运行 <code>python -m offpolicy_eval.cli demo --db data/app.db --scenario base</code></div>`);
