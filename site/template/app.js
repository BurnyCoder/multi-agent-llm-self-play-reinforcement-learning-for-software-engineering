/* Saved artifacts are untrusted display data: textContent is used for every value. */
"use strict";
const byId = id => document.getElementById(id);
const number = value => typeof value === "number" ? new Intl.NumberFormat("en", {maximumFractionDigits: 1}).format(value) : "—";
const percent = value => typeof value === "number" ? `${(100 * value).toFixed(1)}%` : "—";
let dataset;
const sessionEpisodes = new Map();

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function renderEvidence() {
  const human = dataset.analysis.conditions.human;
  for (const name of ["base", "trained"]) {
    byId(`${name}-score`).textContent = percent(human[name].rate);
    byId(`${name}-coverage`).textContent = human[name].completed
      ? `${human[name].successes} / ${human[name].completed} scored · ${number(human[name].planned)} planned · ${human[name].unscored + (human[name].unrecorded || 0)} missing`
      : `No scored measurements · ${number(human[name].planned)} planned`;
  }
  const pair = dataset.analysis.comparisons.human;
  if (pair.difference !== null) {
    byId("effect").textContent = `${pair.difference >= 0 ? "+" : ""}${(pair.difference * 100).toFixed(1)} pp`;
    byId("effect-ci").textContent = `95% interval [${(pair.ci95[0] * 100).toFixed(1)}, ${(pair.ci95[1] * 100).toFixed(1)}] · ${pair.paired_tasks} paired tasks`;
    const messages = {
      positive_task_level_evidence: "The recorded task comparison favors the trained policy. This does not establish seed robustness or the causal benefit of subagents.",
      nonpositive_task_level_evidence: "The recorded task comparison does not support the predicted improvement. Negative results remain part of the research record.",
      no_observed_difference: "No difference was observed on the paired tasks. The preregistered interval rule contradicts the predicted improvement, while this degenerate interval does not establish equivalence or rule out an effect beyond this sample.",
      inconclusive: "The paired interval includes zero. This pilot is inconclusive about an improvement in held-out human-bug repair."
    };
    byId("interpretation").textContent = messages[pair.interpretation] || "The paired comparison is not yet available.";
    if (pair.interval_degenerate && pair.interpretation !== "no_observed_difference")
      byId("interpretation").textContent += " The interval is degenerate and should not be interpreted as precise population uncertainty.";
  }
  renderTable();
}

function renderTable() {
  const source = byId("source-filter").value;
  const outcome = byId("outcome-filter").value;
  const search = byId("task-filter").value.trim().toLowerCase();
  const rows = dataset.analysis.records.filter(row => {
    const scored = row.status === "completed" && typeof row.success === "boolean";
    return (source === "all" || row.source === source) && String(row.task_id).toLowerCase().includes(search)
      && (outcome === "all" || (outcome === "resolved" && scored && row.success)
        || (outcome === "unresolved" && scored && !row.success) || (outcome === "unscored" && !scored));
  });
  const body = byId("evaluation-table"); body.replaceChildren();
  for (const row of rows) {
    const tr = element("tr");
    tr.append(element("td", row.task_id), element("td", row.source), element("td", row.condition));
    const scored = row.status === "completed" && typeof row.success === "boolean";
    const td = element("td");
    const badge = element("span", scored ? (row.success ? "Resolved" : "Not resolved") : row.status || "Unscored",
      `outcome ${scored ? (row.success ? "success" : "failure") : "unknown"}`);
    badge.title = row.reason || ""; td.append(badge);
    tr.append(td, element("td", number(row.generated_tokens)), element("td", number(row.delegations)));
    body.append(tr);
  }
  if (!rows.length) {
    const tr = element("tr"), td = element("td", "No matching evaluation records.", "empty"); td.colSpan = 6; tr.append(td); body.append(tr);
  }
  byId("visible-count").textContent = `${rows.length} records`;
}

function eventValue(event, name) { return event[name] ?? event.data?.[name]; }
function sessionId(event) { return eventValue(event, "session_id"); }
function episodeId(event) { return eventValue(event, "episode_id") || sessionEpisodes.get(sessionId(event)); }
function eventKind(event) { return event.kind || event.event || event.type || "event"; }

function renderTraces() {
  const selectedEpisode = byId("episode-filter").value;
  const events = dataset.events.filter(event => selectedEpisode === "all" || episodeId(event) === selectedEpisode);
  const sessions = new Map();
  for (const event of events) {
    const id = sessionId(event);
    if (!id) continue;
    if (!sessions.has(id)) sessions.set(id, {id, parent: eventValue(event, "parent_session_id"),
      role: eventValue(event, "role") || eventValue(event, "agent") || "agent", events: []});
    sessions.get(id).events.push(event);
    if (eventValue(event, "parent_session_id")) sessions.get(id).parent = eventValue(event, "parent_session_id");
  }
  const tree = byId("session-tree"); tree.replaceChildren();
  byId("trace-count").textContent = `${sessions.size} sessions · ${events.length} events`;
  for (const session of sessions.values()) {
    const button = element("button", undefined, `session-button ${session.parent ? "child" : ""}`);
    button.append(element("strong", `${session.parent ? "↳ " : ""}${session.role}`), element("span", session.id));
    button.onclick = () => {
      tree.querySelectorAll("button").forEach(item => item.classList.remove("active")); button.classList.add("active");
      const panel = byId("event-panel"); panel.replaceChildren(element("h3", `${session.role} · ${session.id}`));
      for (const event of session.events) {
        const article = element("article", undefined, "event");
        const header = element("div", undefined, "event-head");
        header.append(element("strong", eventKind(event)), element("time", event.timestamp || event.time || ""));
        const detail = {...event};
        for (const key of ["timestamp", "time", "session_id", "parent_session_id", "episode_id", "kind", "event", "type"]) delete detail[key];
        article.append(header, element("pre", JSON.stringify(detail, null, 2))); panel.append(article);
      }
    };
    tree.append(button);
  }
  if (!sessions.size) {
    tree.append(element("p", "No agent sessions recorded.", "empty"));
    byId("event-panel").replaceChildren(element("p", "No trace is available for this selection.", "empty"));
  } else tree.querySelector("button").click();
}

function renderPatches() {
  const patches = dataset.patches || [];
  if (!patches.length) return;
  const select = byId("patch-filter"); select.replaceChildren();
  patches.forEach((patch, index) => {
    const option = element("option", patch.path); option.value = String(index); select.append(option);
  });
  const show = () => {
    const patch = patches[Number(select.value)];
    byId("patch-kind").textContent = patch.kind;
    byId("patch-panel").textContent = patch.text || "Empty patch: no source changes were submitted.";
  };
  select.addEventListener("change", show); show();
}

function renderDiagnostics() {
  const diagnostics = dataset.diagnostics || {};
  const labels = {generated_action_tokens: "Generated action tokens", delegation_calls: "Delegation tool calls",
    child_sessions: "Child sessions", optimizer_steps: "Optimizer steps", validated_artifacts: "Validated artifacts"};
  const list = byId("diagnostic-list");
  for (const [key, label] of Object.entries(labels)) list.append(element("dt", label), element("dd", number(diagnostics[key])));
  byId("diagnostic-data").textContent = JSON.stringify(diagnostics, null, 2);
  for (const control of dataset.controls || []) byId("control-list").append(element("li", `${control.condition} — Not run. ${control.purpose}.`));
  if (dataset.failure && Object.keys(dataset.failure).length) {
    byId("run-failure").hidden = false;
    byId("run-failure").textContent = `Recorded ${dataset.failure.phase || "run"} failure: ${dataset.failure.error || "Unspecified"}. This record is retained even if a later phase completes.`;
  }
}

async function start() {
  const response = await fetch("data.json", {cache: "no-cache"});
  if (!response.ok) throw new Error(`Could not load run data (${response.status}).`);
  dataset = await response.json();
  byId("run-id").textContent = dataset.run_id;
  byId("run-status").textContent = String(dataset.status).replaceAll("_", " ");
  byId("generated-at").textContent = `Generated ${new Date(dataset.generated_at).toISOString().slice(0, 16).replace("T", " ")} UTC`;
  byId("summary-link").href = "data.json";
  byId("run-metadata").textContent = JSON.stringify({manifest: dataset.manifest, preflight: dataset.preflight,
    failure: dataset.failure, preparation: dataset.preparation, budget: dataset.budget, run_summary: dataset.run_summary}, null, 2);
  for (const source of dataset.sources) {
    const li = element("li"), a = element("a", source.title); a.href = source.url; a.rel = "noopener"; li.append(a); byId("source-list").append(li);
  }
  for (const id of ["source-filter", "outcome-filter", "task-filter"]) byId(id).addEventListener("input", renderTable);
  for (const event of dataset.events) {
    if (sessionId(event) && eventValue(event, "episode_id")) sessionEpisodes.set(sessionId(event), eventValue(event, "episode_id"));
  }
  const episodes = [...new Set(dataset.events.map(episodeId).filter(Boolean))];
  for (const id of episodes) { const option = element("option", id); option.value = id; byId("episode-filter").append(option); }
  byId("episode-filter").addEventListener("change", renderTraces);
  renderEvidence(); renderTraces(); renderPatches(); renderDiagnostics();
}

start().catch(error => {
  const notice = byId("load-error"); notice.hidden = false;
  notice.textContent = `${error.message} Serve this directory over HTTP or use the published GitHub Pages site.`;
  byId("run-status").textContent = "Data unavailable";
});
