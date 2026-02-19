// 1) Configure once
const API_BASE =
  (new URLSearchParams(window.location.search)).get("api") ||
  "http://localhost:8000"; // default for local dev

async function apiGet(path) {
  const r = await fetch(`${API_BASE}${path}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function apiPost(path, body) {
  const r = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

// 2) Calls your backend
export async function ping() {
  return apiGet("/health");
}

export async function runBaseline(payload) {
  return apiPost("/baseline", payload);
}

export async function runIntervened(payload) {
  return apiPost("/intervene", payload);
}

export async function loadConcepts() {
  return apiGet("/concepts");
}
// Deprecated module retained for compatibility. No mock data loading.
