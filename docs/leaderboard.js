/**
 * GNN Challenge — Interactive Leaderboard
 *
 * Data is embedded between the LEADERBOARD_DATA markers so that
 * `competition/render_leaderboard.py --update-js` can refresh it
 * automatically after each CI scoring run.
 *
 * Features (§13.3):
 *   - Free-text search over team, model, notes, and date
 *   - Filtering by submission type (human, llm, human+llm, baseline)
 *   - Date-range filtering
 *   - Sorting by any column (default: macro_f1 descending)
 *   - Toggling visibility of columns
 */

// --- LEADERBOARD_DATA_START ---
const LEADERBOARD_DATA = [];
// --- LEADERBOARD_DATA_END ---

/* ----------------------------------------------------------------
   Helpers
   ---------------------------------------------------------------- */

const MEDALS = { 1: "🥇", 2: "🥈", 3: "🥉" };

const TOGGLEABLE_COLS = [
  "efficiency", "params", "time_ms", "cliff_accuracy",
  "submission_type", "submitted_at"
];

function fmtNum(v, decimals = 4) {
  return v == null ? "-" : v.toFixed(decimals);
}

function fmtParams(p) {
  if (p == null) return "-";
  if (p >= 1e6) return (p / 1e6).toFixed(1) + "M";
  if (p >= 1e3) return (p / 1e3).toFixed(1) + "K";
  return String(p);
}

function fmtType(t) {
  if (!t) return "-";
  const cls = "type-" + t.replace("+", "-");
  return `<span class="type-badge ${cls}">${t}</span>`;
}

/* ----------------------------------------------------------------
   Column visibility
   ---------------------------------------------------------------- */

function getHiddenCols() {
  const hidden = new Set();
  document.querySelectorAll(".column-toggles input[type=checkbox]").forEach((cb) => {
    if (!cb.checked) hidden.add(cb.dataset.col);
  });
  return hidden;
}

function applyColumnVisibility(hidden) {
  document.querySelectorAll("[data-col]").forEach((el) => {
    el.classList.toggle("hidden-col", hidden.has(el.dataset.col));
  });
}

/* ----------------------------------------------------------------
   Rendering
   ---------------------------------------------------------------- */

function renderTable(data) {
  const tbody = document.getElementById("leaderboard-body");
  const table = document.getElementById("leaderboard-table");
  const emptyMsg = document.getElementById("empty-message");
  const hidden = getHiddenCols();
  applyColumnVisibility(hidden);
  tbody.innerHTML = "";

  if (data.length === 0) {
    table.style.display = "none";
    emptyMsg.style.display = "block";
    document.getElementById("row-count").textContent = "";
    return;
  }

  table.style.display = "table";
  emptyMsg.style.display = "none";

  data.forEach((row, idx) => {
    const rank = idx + 1;
    const medal = MEDALS[rank] || "";
    const tr = document.createElement("tr");
    const rc = rank <= 3 ? ` class="rank-${rank}"` : "";

    tr.innerHTML = `
      <td${rc} data-col="rank">${medal} ${rank}</td>
      <td data-col="team">${row.team}</td>
      <td data-col="macro_f1" class="primary-metric">${fmtNum(row.macro_f1)}</td>
      <td data-col="efficiency"${hidden.has("efficiency") ? ' class="hidden-col"' : ""}>${fmtNum(row.efficiency)}</td>
      <td data-col="params"${hidden.has("params") ? ' class="hidden-col"' : ""}>${fmtParams(row.params)}</td>
      <td data-col="time_ms"${hidden.has("time_ms") ? ' class="hidden-col"' : ""}>${row.time_ms != null ? row.time_ms.toFixed(1) : "-"}</td>
      <td data-col="cliff_accuracy"${hidden.has("cliff_accuracy") ? ' class="hidden-col"' : ""}>${fmtNum(row.cliff_accuracy)}</td>
      <td data-col="submission_type"${hidden.has("submission_type") ? ' class="hidden-col"' : ""}>${fmtType(row.submission_type)}</td>
      <td data-col="submitted_at"${hidden.has("submitted_at") ? ' class="hidden-col"' : ""}>${row.submitted_at || "-"}</td>
    `;
    tbody.appendChild(tr);
  });

  document.getElementById("row-count").textContent =
    `Showing ${data.length} of ${LEADERBOARD_DATA.length} entries`;
}

/* ----------------------------------------------------------------
   Sorting
   ---------------------------------------------------------------- */

function sortData(data, key) {
  const ascending = key.endsWith("_asc");
  const realKey = key.replace(/_asc$/, "");

  return [...data].sort((a, b) => {
    const va = a[realKey] ?? (ascending ? Infinity : -Infinity);
    const vb = b[realKey] ?? (ascending ? Infinity : -Infinity);
    return ascending ? va - vb : vb - va;
  });
}

/* ----------------------------------------------------------------
   Filtering
   ---------------------------------------------------------------- */

function filterByText(data, query) {
  if (!query) return data;
  const q = query.toLowerCase();
  return data.filter((r) => {
    const searchable = [
      r.team,
      r.submission_type,
      r.submitted_at,
      r.notes,
      r.model_name,
    ]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
    return searchable.includes(q);
  });
}

function filterByType(data, typeValue) {
  if (!typeValue || typeValue === "all") return data;
  return data.filter(
    (r) => r.submission_type && r.submission_type.toLowerCase() === typeValue.toLowerCase()
  );
}

function filterByDateRange(data, from, to) {
  return data.filter((r) => {
    if (!r.submitted_at) return true; // show entries without dates
    if (from && r.submitted_at < from) return false;
    if (to && r.submitted_at > to) return false;
    return true;
  });
}

/* ----------------------------------------------------------------
   Refresh (central pipeline)
   ---------------------------------------------------------------- */

function refresh() {
  const sortKey = document.getElementById("sort-select").value;
  const query = document.getElementById("search-input").value;
  const typeFilter = document.getElementById("type-filter").value;
  const dateFrom = document.getElementById("date-from").value;
  const dateTo = document.getElementById("date-to").value;

  let data = LEADERBOARD_DATA;
  data = filterByText(data, query);
  data = filterByType(data, typeFilter);
  data = filterByDateRange(data, dateFrom, dateTo);
  data = sortData(data, sortKey);
  renderTable(data);
}

/* ----------------------------------------------------------------
   Init
   ---------------------------------------------------------------- */

document.getElementById("sort-select").addEventListener("change", refresh);
document.getElementById("search-input").addEventListener("input", refresh);
document.getElementById("type-filter").addEventListener("change", refresh);
document.getElementById("date-from").addEventListener("change", refresh);
document.getElementById("date-to").addEventListener("change", refresh);

document.querySelectorAll(".column-toggles input[type=checkbox]").forEach((cb) => {
  cb.addEventListener("change", refresh);
});

// Initial render
refresh();
