/* payTrace web app. No framework, no build step. All text is inserted as text nodes (never innerHTML from data). */
"use strict";

// ---------------- helpers ----------------
// replaceChildren() turns null into the text "null"; optional parts of a view are often null, so drop them
const _replaceChildren = Element.prototype.replaceChildren;
Element.prototype.replaceChildren = function (...nodes) { return _replaceChildren.apply(this, nodes.filter(n => n != null && n !== false)); };
const $ = (s, r = document) => r.querySelector(s);
const app = $("#app");
const DEVICE_KEY = "fs_device_id";
const deviceId = localStorage.getItem(DEVICE_KEY) || (() => {
  const id = (crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2)).replace(/-/g, "").slice(0, 20);
  localStorage.setItem(DEVICE_KEY, id); return id;
})();
const state = { user: null, options: null, profile: null, tab: null, selectedCase: null, llm: null };
const statusText = s => (state.user && state.user.role !== "customer" && s === "verification_required") ? "Awaiting customer decision" : (STATUS_TEXT[s] || s);
const rupee = v => v == null ? "–" : "₹" + Number(v).toLocaleString("en-IN", { maximumFractionDigits: 0 });
const pct = (v, d = 1) => v == null ? "–" : (v * 100).toFixed(d) + "%";
const STATUS_TEXT = { completed: "Sent", verification_required: "Waiting for your choice", held: "Blocked",
  cancelled: "Cancelled", released: "Released by fraud team", confirmed_fraud: "Stopped: confirmed fraud",
  review: "Needs review", cleared: "Cleared", release_requested: "Release requested", blocked_final: "Blocked after review" };
const CHOICES = { verification_required: ["proceed", "cancel"] };   // high-risk (blocked) payments have no customer choice

function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "html") throw new Error("no html");
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}
const svg = (tag, attrs) => { const e = document.createElementNS("http://www.w3.org/2000/svg", tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); return e; };

async function api(path, opts = {}) {
  const res = await fetch(path, {
    method: opts.method || "GET", credentials: "same-origin",
    headers: { "Content-Type": "application/json", "X-FS-Request": "1", "X-FS-Device": deviceId },
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  let data = null;
  try { data = await res.json(); } catch { /* non-JSON */ }
  if (!res.ok) { const e = new Error((data && data.error) || res.statusText); e.status = res.status; throw e; }
  return data;
}
function toast(msg, isErr) {
  const t = $("#toast"); t.textContent = msg; t.className = "toast on" + (isErr ? " err" : "");
  clearTimeout(t._h); t._h = setTimeout(() => t.className = "toast", 3500);
}
const chip = (cls, text) => h("span", { class: "chip " + cls }, text || cls);
const kpi = (l, v, d) => h("div", { class: "card kpi" }, h("div", { class: "l" }, l), h("div", { class: "v" }, v), d ? h("div", { class: "d" }, d) : null);
function mount(...nodes) { app.replaceChildren(...nodes); }

/* tiny, safe markdown: escapes everything, then supports headings, bold, bullets, numbered lists */
function md(text) {
  const box = h("div", { class: "md" });
  let list = null;
  const inline = (line, el) => {
    // italics only for _words_ that start and end at a word boundary, so names like SIM_new_payee stay intact
    line.split(/(\*\*[^*]+\*\*|(?<![\w])_[^_\s][^_]*_(?![\w]))/).forEach(part => {
      if (/^\*\*[^*]+\*\*$/.test(part)) el.append(h("strong", {}, part.slice(2, -2)));
      else if (/^_[^_]+_$/.test(part)) el.append(h("em", {}, part.slice(1, -1)));
      else el.append(document.createTextNode(part));
    });
    return el;
  };
  for (const raw of String(text).split("\n")) {
    const line = raw.trimEnd();
    let m;
    if ((m = line.match(/^\s*[-*]\s+(.*)/))) { if (!list || list.tagName !== "UL") { list = h("ul"); box.append(list); } list.append(inline(m[1], h("li"))); continue; }
    if ((m = line.match(/^\s*\d+[.)]\s+(.*)/))) { if (!list || list.tagName !== "OL") { list = h("ol"); box.append(list); } list.append(inline(m[1], h("li"))); continue; }
    list = null;
    if (!line.trim()) continue;
    if ((m = line.match(/^#{1,4}\s+(.*)/))) { box.append(inline(m[1], h("h4"))); continue; }
    box.append(inline(line, h("p")));
  }
  return box;
}

// ---------------- boot ----------------
async function boot() {
  const hl = await api("/api/health").catch(() => null);
  if (hl) { state.llm = hl.llm; setLlmChip(); }
  try {
    const me = await api("/api/me");
    state.user = me.user; state.options = me.options; state.profile = me.profile || null;
    if (hl && !hl.ready) return warmup();
    renderShell();
  } catch (e) {
    if (e.status === 401) return renderLogin(hl);
    mount(h("div", { class: "card" }, "Cannot reach the server: " + e.message));
  }
}
function setLlmChip() {
  const c = $("#llmChip"); const s = state.llm || {};
  c.textContent = s.configured ? `AI: ${s.provider}` : "AI: template";
  c.className = "chip" + (s.configured ? " llm-on" : "");
  c.title = s.configured ? `Explanations written by ${s.provider} (${s.model}) from verified evidence` : "No LLM key configured: explanations use templates";
}
async function warmup() {
  $("#topbar").hidden = true;
  const msg = h("p", { class: "muted" }, "Starting…");
  mount(h("div", { class: "warmup card" }, h("div", { class: "spin" }), h("h2", {}, "Starting the fraud engine"), msg,
    h("p", { class: "small muted" }, "Loading models and rebuilding customer behaviour from past payments. This takes about 20–40 seconds, once.")));
  for (;;) {
    const hl = await api("/api/health").catch(() => null);
    if (hl && hl.error) { msg.textContent = "Failed: " + hl.error; return; }
    if (hl && hl.ready) return boot();
    if (hl) msg.textContent = hl.progress;
    await new Promise(r => setTimeout(r, 1500));
  }
}

// ---------------- login ----------------
async function renderLogin(hl) {
  $("#topbar").hidden = true;
  const user = h("input", { id: "u", autocomplete: "username", required: true });
  const pw = h("input", { id: "p", type: "password", autocomplete: "current-password", required: true });
  const errBox = h("p", { class: "small err-text" });
  const signinForm = h("form", { class: "card", onsubmit: async ev => {
    ev.preventDefault(); errBox.textContent = "";
    try { await api("/api/login", { method: "POST", body: { username: user.value, password: pw.value } }); boot(); }
    catch (e) { errBox.textContent = e.message; }
  } }, h("h2", {}, "Sign in"), h("div", { class: "field" }, h("label", { for: "u" }, "Username"), user),
    h("div", { class: "field" }, h("label", { for: "p" }, "Password"), pw), errBox, h("button", { class: "btn block cta" }, "Sign in"),
    h("p", { class: "small muted center" }, "New customer? ", h("a", { href: "#", onclick: ev => { ev.preventDefault(); slot.replaceChildren(signupForm); } }, "Create an account")));

  const su = { name: h("input", { id: "sn", autocomplete: "name", required: true, maxlength: "60" }),
    user: h("input", { id: "su", autocomplete: "username", required: true, pattern: "[a-z0-9_.]{3,32}", maxlength: "32" }),
    pw: h("input", { id: "sp", type: "password", autocomplete: "new-password", required: true, minlength: "8" }),
    pw2: h("input", { id: "sp2", type: "password", autocomplete: "new-password", required: true, minlength: "8" }) };
  su.user.addEventListener("input", () => { su.user.value = su.user.value.toLowerCase(); });
  const suErr = h("p", { class: "small err-text" });
  const signupForm = h("form", { class: "card", onsubmit: async ev => {
    ev.preventDefault(); suErr.textContent = "";
    if (su.pw.value !== su.pw2.value) { suErr.textContent = "The passwords don't match."; return; }
    try { await api("/api/signup", { method: "POST", body: { display_name: su.name.value, username: su.user.value, password: su.pw.value } });
      toast("Account created. Welcome!"); boot(); }
    catch (e) { suErr.textContent = e.message; }
  } }, h("h2", {}, "Create your account"),
    h("p", { class: "small muted" }, "Opens a new customer account. Staff accounts are created by an administrator."),
    h("div", { class: "field" }, h("label", { for: "sn" }, "Full name"), su.name),
    h("div", { class: "field" }, h("label", { for: "su" }, "Choose a username"), su.user, h("div", { class: "hint" }, "3–32 characters: lowercase letters, numbers, dot or underscore")),
    h("div", { class: "field" }, h("label", { for: "sp" }, "Password"), su.pw, h("div", { class: "hint" }, "At least 8 characters, with letters and numbers")),
    h("div", { class: "field" }, h("label", { for: "sp2" }, "Confirm password"), su.pw2),
    suErr, h("button", { class: "btn block cta" }, "Create account"),
    h("p", { class: "small muted center" }, "Already have an account? ", h("a", { href: "#", onclick: ev => { ev.preventDefault(); slot.replaceChildren(signinForm); } }, "Sign in")));
  const slot = h("div", {}, signinForm);
  const demo = h("div", { class: "card flat", style: "margin-top:16px" });
  try {
    const d = await api("/api/demo-accounts");
    if (d.accounts.length) {
      demo.append(h("h3", {}, "Demo accounts"), h("p", { class: "small muted" }, `Password for all: ${d.password}. Click one to fill the form.`));
      const groups = { customer: "Customers", analyst: "Fraud analysts", admin: "Administrators" };
      for (const role of ["customer", "analyst", "admin"]) {
        demo.append(h("div", { class: "small muted", style: "margin-top:10px" }, groups[role]));
        d.accounts.filter(a => a.role === role).slice(0, role === "customer" ? 6 : 3).forEach(a =>
          demo.append(h("div", { class: "acct", onclick: () => { slot.replaceChildren(signinForm); user.value = a.username; pw.value = d.password; } },
            h("span", {}, a.display_name), h("span", { class: "mono small muted" }, a.username))));
      }
    } else demo.append(h("p", { class: "muted" }, "Ask your administrator for an account."));
  } catch { demo.hidden = true; }
  mount(h("div", { class: "login" },
    h("div", { class: "hero" }, h("h1", {}, "Every payment checked in milliseconds."),
      h("p", {}, "payTrace checks each payment against the customer's own behaviour, the payment network and patterns of known fraud, then explains its decision in plain language."),
      h("p", { class: "small muted" }, "Customers send money and confirm unusual payments. Analysts investigate flagged cases with an evidence-based AI copilot. Admins run the system."),
      demo), slot));
}

// ---------------- shell / tabs ----------------
const TABS = {
  customer: [["pay", "Send money"], ["activity", "My activity"], ["queries", "My queries"]],
  analyst: [["cases", "Case queue"], ["rings", "Fraud rings"], ["check", "Check a payment"], ["lookup", "Customer lookup"]],
  admin: [["overview", "Overview"], ["scenarios", "Scenario lab"], ["cases", "Cases"], ["queries", "Queries"], ["rings", "Fraud rings"], ["check", "Check payment"],
          ["traffic", "Traffic"], ["replay", "Real data"], ["model", "Model"], ["users", "Users"], ["system", "System"]],
};
function renderShell() {
  $("#topbar").hidden = false;
  $("#who").textContent = `${state.user.display_name} · ${state.user.role}`;
  $("#logoutBtn").onclick = async () => { await api("/api/logout", { method: "POST" }).catch(() => {}); location.reload(); };
  const tabs = TABS[state.user.role];
  if (!state.tab || !(tabs.find(t => t[0] === state.tab) || state.tab === "account")) state.tab = tabs[0][0];
  const who = $("#who"); who.title = "My account: change password"; who.classList.toggle("on", state.tab === "account");
  who.onclick = () => { state.tab = "account"; renderShell(); };
  const nav = $("#tabs"); nav.replaceChildren(...tabs.map(([k, label]) =>
    h("button", { class: k === state.tab ? "on" : "", onclick: () => { state.tab = k; renderShell(); } }, label)));
  tickClock();
  ({ pay: viewPay, activity: viewActivity, cases: viewCases, check: viewCheck, lookup: viewLookup, overview: viewOverview,
     traffic: viewTraffic, model: viewModel, system: viewSystem, users: viewUsers, account: viewAccount, replay: viewReplay,
     rings: viewRings, scenarios: viewScenarios, queries: state.user.role === "admin" ? viewAdminQueries : viewMyQueries })[state.tab]();
}
async function tickClock() {
  const hl = await api("/api/health").catch(() => null);
  if (hl) { $("#clock").textContent = hl.bank_time ? hl.bank_time.slice(0, 16) : ""; state.llm = hl.llm; setLlmChip(); }
}

// ---------------- customer: send money ----------------
const chLabel = c => ((state.options && state.options.channel_labels) || {})[c] || c;
const catLabel = c => String(c).replace(/_/g, " ").replace(/\bpos\b/, "(in store)").replace(/\bnet\b/, "(online)");
const catSelect = () => h("select", {}, h("option", { value: "" }, "Automatic (merchant's own category)"),
  state.options.categories.map(c => h("option", { value: c }, catLabel(c))));
const chSelect = () => h("select", {}, h("option", { value: "" }, "Automatic"), state.options.channels.map(c => h("option", { value: c }, chLabel(c))));
let cityListN = 0;
const cityInput = value => { const id = `cities${++cityListN}`;
  return [h("input", { list: id, value: value || "", placeholder: "City, state" }), h("datalist", { id }, state.options.cities.map(c => h("option", { value: c })))]; };

// search-as-you-type suggestions from the bank's records (staff forms)
let suggestN = 0;
function suggest(input, kind) {
  const id = `sg${++suggestN}`, dl = h("datalist", { id });
  input.setAttribute("list", id); input.setAttribute("autocomplete", "off");
  let t, last = null;
  const load = () => { clearTimeout(t); t = setTimeout(async () => {
    const q = input.value.trim(); if (q === last) return; last = q;
    try { const r = await api(`/api/directory?kind=${kind}&q=${encodeURIComponent(q)}`);
      dl.replaceChildren(...r.results.map(x => h("option", { value: x.value, label: x.label }, x.label))); } catch { /* ignore */ }
  }, 200); };
  input.addEventListener("input", load); input.addEventListener("focus", load);
  return dl;
}

function viewPay() {
  const p = state.profile, o = state.options;
  const ex = (p.saved_payees || [])[0] || (o.examples || {}).merchant || "";
  const payee = h("input", { id: "payee", list: "payees", placeholder: ex ? `e.g. ${ex}` : "merchant name", required: true });
  const dl = h("datalist", { id: "payees" }, (p.saved_payees || []).map(x => h("option", { value: x })));
  const amount = h("input", { id: "amount", type: "number", min: "1", step: "0.01", required: true, placeholder: "0.00" });
  const cat = catSelect(); cat.id = "cat";
  const ch = chSelect(); ch.id = "ch";
  const dev = h("select", { id: "dev" }, h("option", { value: "this_browser" }, "This browser (a device the bank has never seen)"),
    p.usual_device ? h("option", { value: "usual", selected: true }, `My usual phone (${p.usual_device})`) : null,
    h("option", { value: "custom" }, "Another device id…"));
  const custom = h("input", { id: "cdev", placeholder: "device id, e.g. D_SHARED_1", hidden: true });
  dev.onchange = () => custom.hidden = dev.value !== "custom";
  const [city, cityDl] = cityInput(p.usual_city); city.id = "city";
  const out = h("div");
  const btn = h("button", { class: "btn block cta" }, "Pay now");
  const form = h("form", { class: "card", onsubmit: async ev => {
    ev.preventDefault(); btn.disabled = true; btn.textContent = "Checking payment…";
    try {
      const r = await api("/api/customer/pay", { method: "POST", body: { payee: payee.value, amount: amount.value, category: cat.value,
        channel: ch.value, device: dev.value, custom_device: custom.value, city: city.value } });
      out.replaceChildren(resultCard(r));
    } catch (e) { toast(e.message, true); }
    finally { btn.disabled = false; btn.textContent = "Pay now"; }
  } },
    h("h2", {}, "Send money"),
    h("div", { class: "field" }, h("label", { for: "payee" }, "To (merchant)"), payee, dl,
      h("div", { class: "hint" }, "Your usual merchants are suggested as you type. A known merchant's category and channel are filled in by the bank.")),
    h("div", { class: "grid g2" }, h("div", { class: "field" }, h("label", { for: "amount" }, "Amount (₹)"), amount),
      h("div", { class: "field" }, h("label", { for: "ch" }, "Pay with"), ch)),
    h("div", { class: "field" }, h("label", { for: "cat" }, "Category (only used for a merchant the bank doesn't know)"), cat),
    h("details", { class: "sim" }, h("summary", {}, "Demo controls: simulate the device and location of this payment"),
      h("div", { class: "field", style: "margin-top:10px" }, h("label", { for: "dev" }, "Device"), dev, custom),
      h("div", { class: "field" }, h("label", { for: "city" }, "Location"), city, cityDl),
      o.has_devices === false ? h("p", { class: "small muted" }, "This model was trained on card data without device information, so the device only affects the bank's policy rules, not the model score.") : null,
      h("p", { class: "small muted" }, "In a real banking app these come from the phone automatically. They are selectable here so you can test how the engine reacts.")),
    btn);
  mount(h("h1", {}, `Hello, ${state.user.display_name.split(" (")[0]}`),
    h("p", { class: "sub" }, p.past_payments ? `Account ${p.customer_id} · ${p.past_payments} past payments · usually ${rupee(p.usual_amount)} · ${p.usual_city || "–"}`
      : `Account ${p.customer_id} · new account. Your first payments may ask you to confirm while we learn how you pay.`),
    h("div", { class: "split" }, form, h("div", {}, out.childNodes.length ? null : h("div", { class: "card flat" },
      h("h3", {}, "How your payments are protected"),
      h("p", { class: "muted" }, "Every payment is checked instantly against how you normally pay: your usual devices, places, amounts and payees, plus patterns seen across the bank."),
      h("p", { class: "muted" }, "Normal payments go straight through. Unusual ones (moderate risk) let you decide: send it after a quick security check, or cancel. High-risk payments are blocked outright so no money is lost.")), out)));
}

function resultCard(r) {
  const titles = { SENT: "Payment sent", VERIFY: "Check this payment before it goes", HELD: "Payment blocked" };
  const icons = { SENT: "SENT", VERIFY: "CHECK", HELD: "BLOCKED" };   // rubber-stamp marks
  const levels = { low: "Low", elevated: "Elevated", high: "High" };
  const c = r.checked || {};
  const box = h("div", { class: "result " + r.action },
    h("div", { class: "row", style: "align-items:flex-start" }, h("div", { class: "icon" }, icons[r.action]),
      h("div", { class: "spacer" }), r.action !== "SENT" ? chip(r.risk_level === "high" ? "BLOCK" : "REVIEW", `Risk: ${levels[r.risk_level] || r.risk_level}`) : null),
    h("h2", {}, titles[r.action]),
    md(r.message),
    r.action !== "SENT" ? h("details", { class: "how" }, h("summary", {}, "How we checked this payment"),
      h("ul", {},
        h("li", {}, c.past_payments ? `Compared it with your ${Number(c.past_payments).toLocaleString("en-IN")} earlier payments: amounts, merchants and times.` : "Your account has no earlier payments to compare with."),
        h("li", {}, !c.payee_known && c.confirmed_before ? `The bank doesn't list this payee, but you've already confirmed ${c.confirmed_before} payment${c.confirmed_before > 1 ? "s" : ""} to it here.` : c.payee_known ? `Checked ${r.category ? "the merchant and its category" : "the merchant"} against the bank's records${c.paid_before ? ` (you have paid them ${c.paid_before} times)` : " (a merchant you haven't paid before)"}.` : "Looked the payee up in the bank's merchant records and couldn't find it."),
        h("li", {}, "Ran fraud-detection models trained on 1.85 million card payments, plus checks for unusual activity."),
        h("li", {}, "Applied the bank's safety rules: payment limits, unknown payees and unusually large amounts."))) : null,
    r.message_source && r.message_source !== "template" ? h("p", { class: "small muted" }, `Explanation written by ${r.message_source} AI from the bank's checks.`) : null,
    h("p", { class: "small muted" }, `Reference ${r.txn_id} · ${r.ts}${r.latency_ms != null ? ` · checked in ${r.latency_ms.toFixed(1)} ms` : ""}`));
  const status = r.status;
  if (CHOICES[status]) box.append(choiceCard(r.txn_id, status));
  if (status === "held") box.append(queryCard(r.txn_id));
  return box;
}

/* A blocked payment can't be sent by the customer, but they can ask the bank to review it. */
function queryCard(txnId) {
  const reason = h("textarea", { rows: 3, maxlength: 1000, placeholder: "Explain why this payment is genuine, e.g. who the payee is and why you're paying them." });
  const btn = h("button", { class: "btn", type: "submit" }, "Raise a query");
  return h("form", { class: "card flat", style: "margin-top:12px", onsubmit: async ev => {
      ev.preventDefault(); btn.disabled = true;
      try { await api("/api/customer/queries/new", { method: "POST", body: { txn_id: txnId, reason: reason.value } });
        toast("Query sent. You can follow it under My queries."); state.tab = "queries"; renderShell(); }
      catch (e) { toast(e.message, true); btn.disabled = false; }
    } },
    h("h3", {}, "Think this is a mistake?"),
    h("p", { class: "small muted" }, "Raise a query and the bank will review this payment together with your reasons. Nothing is sent until they decide; you'll see the decision under My queries."),
    reason, h("div", { style: "margin-top:8px" }, btn));
}

const Q_CHIP = { open: "REVIEW", approved: "ALLOW", rejected: "BLOCK" };
async function viewMyQueries() {
  mount(h("h1", {}, "My queries"), h("p", { class: "sub" }, "Queries you raised about blocked payments, and the bank's decision on each."), h("div", { class: "card" }, "Loading…"));
  const d = await api("/api/customer/queries");
  mount(h("h1", {}, "My queries"), h("p", { class: "sub" }, "Queries you raised about blocked payments, and the bank's decision on each. To raise one, open a blocked payment under My activity."),
    d.queries.length ? h("div", {}, d.queries.map(q => h("div", { class: "card", style: "margin-bottom:12px" },
      h("div", { class: "row" }, h("h3", { style: "margin:0" }, `${rupee(q.amount)} to ${q.payee}`), h("div", { class: "spacer" }), chip(Q_CHIP[q.status], q.status_text)),
      h("p", { class: "small muted" }, `Query #${q.id} · raised ${q.created}${q.status !== "open" ? ` · decided ${q.updated}` : ""} · payment ${q.payment_ts}`),
      h("p", {}, h("b", {}, "Your reason: "), q.reason),
      q.response ? h("p", {}, h("b", {}, "Bank's response: "), q.response) : h("p", { class: "small muted" }, "The bank hasn't decided yet. This page updates automatically.")))) 
      : h("div", { class: "card muted" }, "No queries yet."));
  if (d.queries.some(q => q.status === "open")) setTimeout(() => { if (state.tab === "queries") viewMyQueries(); }, 10000);
}

async function viewAdminQueries() {
  mount(h("h1", {}, "Customer queries"), h("div", { class: "card" }, "Loading…"));
  const d = await api("/api/admin/queries");
  const card = q => {
    const resp = h("textarea", { rows: 2, maxlength: 1000, placeholder: "Response to the customer (required)" });
    const act = async decision => {
      try { const r = await api(`/api/admin/queries/${q.id}`, { method: "POST", body: { decision, response: resp.value } });
        toast(decision === "approve" ? `Approved: payment ${statusText(r.payment_status)}` : "Rejected: payment stays blocked"); viewAdminQueries(); }
      catch (e) { toast(e.message, true); }
    };
    return h("div", { class: "card", style: "margin-bottom:12px" },
      h("div", { class: "row" }, h("h3", { style: "margin:0" }, `#${q.id} · ${q.username} (${q.user_id}) · ${rupee(q.amount)} to ${q.payee}`), h("div", { class: "spacer" }), chip(Q_CHIP[q.status], q.status_text)),
      h("p", { class: "small muted" }, `Raised ${q.created} · payment ${q.payment_ts} · ${q.txn_id} · decision ${q.decision} at risk ${pct(q.risk, 1)} · payment now: ${statusText(q.payment_status)}`),
      h("p", {}, h("b", {}, "Customer's reason: "), q.reason),
      h("button", { class: "btn ghost sm", onclick: () => { state.selectedCase = q.txn_id; state.tab = "cases"; renderShell(); } }, "Open the payment's evidence (case)"),
      q.status === "open" ? h("div", { style: "margin-top:10px" }, resp, h("div", { class: "row", style: "margin-top:8px;gap:8px" },
          h("button", { class: "btn good", onclick: () => act("approve") }, "Approve and release payment"),
          h("button", { class: "btn bad-btn", onclick: () => act("reject") }, "Reject, keep blocked")))
        : h("p", {}, h("b", {}, `Decided by ${q.admin} (${q.updated}): `), q.response));
  };
  const c = d.counts || {};
  mount(h("h1", {}, "Customer queries"), h("p", { class: "sub" }, "Customers' requests to review a BLOCKED payment. Check the evidence, then approve (the payment is released and recorded as genuine) or reject (it stays blocked for good). The customer sees your response under My queries."),
    h("div", { class: "grid g3" }, kpi("Open", c.open || 0), kpi("Approved", c.approved || 0), kpi("Rejected", c.rejected || 0)),
    h("div", { style: "margin-top:16px" }, d.queries.length ? d.queries.map(card) : h("div", { class: "card muted" }, "No queries yet.")));
}

/* The customer's own decision on a flagged payment, behind a security check (CAPTCHA) for "send" / "release". */
function choiceCard(txnId, status) {
  const wrap = h("div", { class: "choice" });
  const img = h("img", { class: "captcha", alt: "Security check: type the characters you see", width: "220", height: "70" });
  let n = 0;
  const reload = () => { img.src = `/api/customer/payments/${encodeURIComponent(txnId)}/captcha?n=${++n}`; ans.value = ""; };
  const ans = h("input", { class: "otp", maxlength: "5", autocomplete: "off", autocapitalize: "off", autocorrect: "off", spellcheck: "false", placeholder: "Type the characters (case matters)", "aria-label": "Security check answer" });
  const done = h("div");
  const send = async choice => {
    try {
      const res = await api(`/api/customer/payments/${encodeURIComponent(txnId)}/respond`, { method: "POST", body: { choice, captcha: ans.value } });
      const said = { completed: "Done. Your payment has been sent.", cancelled: "Cancelled. No money has left your account.",
        release_requested: "Thanks. Our fraud team will look at it first and let you know. No money has left your account yet." };
      wrap.replaceChildren(h("p", {}, said[res.status] || "Updated."), chip(res.status, STATUS_TEXT[res.status] || res.status));
    } catch (e) { toast(e.message, true); if (/expired|new image|too many/.test(e.message)) reload(); }
  };
  const ok = h("button", { class: "btn good", onclick: () => send("proceed") }, "Yes, send it");
  wrap.append(
    h("h3", { style: "margin-top:4px" }, "Your choice"),
    h("p", { class: "small muted" }, "It's your decision. To send it, prove you're a person (not an automated script) by typing the characters below exactly as shown: capital and small letters are different."),
    h("div", { class: "row", style: "margin:8px 0" }, img, h("button", { class: "btn ghost sm", type: "button", onclick: reload }, "New image")),
    h("div", { class: "field", style: "max-width:260px" }, ans),
    h("div", { class: "row" }, ok, h("button", { class: "btn ghost", onclick: () => send("cancel") }, "Cancel, I didn't make it")),
    done);
  reload();
  return wrap;
}

// ---------------- customer: activity ----------------
async function viewActivity() {
  mount(h("h1", {}, "My activity"), h("p", { class: "sub" }, "Payments you made in this app, with the decision on each."), h("div", { class: "card" }, "Loading…"));
  const d = await api("/api/customer/payments");
  const panel = h("div");
  const rows = d.payments.map(p => h("tr", {}, h("td", { class: "mono small" }, p.ts), h("td", {}, p.payee), h("td", { class: "num" }, rupee(p.amount)),
    h("td", {}, chip(p.status, STATUS_TEXT[p.status] || p.status),
      CHOICES[p.status] ? h("div", {}, h("button", { class: "btn sm", style: "margin-top:6px", onclick: () => { panel.replaceChildren(h("div", { class: "card", style: "margin-top:16px" }, h("h3", {}, `${rupee(p.amount)} to ${p.payee}`), md(p.customer_message || ""), choiceCard(p.txn_id, p.status))); panel.scrollIntoView({ behavior: "smooth" }); } }, "Decide")) : null,
      p.status === "held" ? h("div", {}, h("button", { class: "btn ghost sm", style: "margin-top:6px", onclick: () => { panel.replaceChildren(h("div", { class: "card", style: "margin-top:16px" }, h("h3", {}, `${rupee(p.amount)} to ${p.payee}`), md(p.customer_message || ""), queryCard(p.txn_id))); panel.scrollIntoView({ behavior: "smooth" }); } }, "Raise a query")) : null),
    h("td", { class: "small muted" }, String(p.customer_message || "").split("\n")[0])));
  const earlier = d.earlier_activity.map(p => h("tr", {}, h("td", { class: "mono small" }, p.ts), h("td", {}, p.payee), h("td", { class: "num" }, rupee(p.amount)),
    h("td", { class: "small muted" }, `${p.category} · ${p.channel} · ${p.city}`)));
  mount(h("h1", {}, "My activity"), h("p", { class: "sub" }, "Payments you made in this app, with the decision on each."),
    h("div", { class: "card" }, rows.length ? h("table", {}, h("thead", {}, h("tr", {}, ["Time", "To", "Amount", "Status", "Message"].map(x => h("th", {}, x)))), h("tbody", {}, rows))
      : h("p", { class: "muted" }, "No payments yet. Use Send money to make one.")), panel,
    h("div", { class: "card", style: "margin-top:16px" }, h("h3", {}, "Earlier activity on your account (bank records)"),
      h("table", {}, h("thead", {}, h("tr", {}, ["Time", "To", "Amount", "Details"].map(x => h("th", {}, x)))), h("tbody", {}, earlier))));
}

// ---------------- analyst: case queue ----------------
async function viewCases() {
  const status = h("select", {}, h("option", { value: "open" }, "Open cases"), h("option", { value: "all" }, "All flagged"));
  const source = h("select", {}, ["all", "customer", "analyst_check", "api", "simulator"].map(s => h("option", { value: s }, s === "all" ? "All sources" : s)));
  const list = h("div", { class: "card scroll" }, "Loading…");
  const detail = h("div", {}, h("div", { class: "card flat muted" }, "Select a case to investigate."));
  const load = async () => {
    const d = await api(`/api/cases?status=${status.value}&source=${source.value}`);
    const counts = Object.entries(d.counts).map(([k, v]) => chip(k, `${statusText(k)}: ${v}`));
    if (!d.cases.length) { list.replaceChildren(h("div", { class: "row" }, counts), h("p", { class: "muted" }, "No cases. Make a payment as a customer, or start simulated traffic (admin).")); return; }
    list.replaceChildren(h("div", { class: "row", style: "margin-bottom:8px" }, counts),
      h("table", {}, h("thead", {}, h("tr", {}, ["Risk", "Amount", "Customer → payee", "Status"].map(x => h("th", {}, x)))),
        h("tbody", {}, d.cases.map(c => {
          const tr = h("tr", { class: "click" + (c.txn_id === state.selectedCase ? " sel" : ""), onclick: () => { state.selectedCase = c.txn_id; list.querySelectorAll("tr").forEach(x => x.classList.remove("sel")); tr.classList.add("sel"); openCase(c.txn_id, detail, load); } },
            h("td", {}, chip(c.decision, pct(c.risk, 0))), h("td", { class: "num" }, rupee(c.amount)),
            h("td", { class: "small" }, `${c.user_id} → ${c.payee}`), h("td", {}, chip(c.status, statusText(c.status)), c.source !== "simulator" ? h("div", { class: "small muted" }, c.source) : null));
          return tr;
        }))));
  };
  status.onchange = load; source.onchange = load;
  mount(h("div", { class: "row" }, h("div", {}, h("h1", {}, "Case queue"), h("p", { class: "sub" }, "Flagged payments, customer-initiated first, then by risk.")),
    h("div", { class: "spacer" }), status, source, h("button", { class: "btn ghost sm", onclick: load }, "Refresh")),
    h("div", { class: "split" }, list, detail));
  await load();
  if (state.selectedCase) openCase(state.selectedCase, detail, load);
}

const isPolicy = r => String(r.feature || "").startsWith("policy:");
function policyBox(reasons, modelDecision, finalDecision) {
  const pol = reasons.filter(isPolicy);
  if (!pol.length) return null;
  return h("div", { class: "policy" }, h("strong", {}, "Bank policy rules"),
    modelDecision && modelDecision !== finalDecision ? h("p", { class: "small", style: "margin:4px 0" }, `The model alone said ${modelDecision}; policy raised it to ${finalDecision}.`) : null,
    h("ul", {}, pol.map(r => h("li", {}, r.reason.replace(/^Policy: /, "")))));
}
function bars(items, valueKey, labelKey, signed) {
  const max = Math.max(...items.map(i => Math.abs(i[valueKey] || 0)), 1e-9);
  return h("div", { class: "bars" }, items.map(i => {
    const v = i[valueKey] || 0;
    return h("div", { class: "bar" }, h("span", {}, i[labelKey]),
      h("div", { class: "track" }, h("div", { class: "fill " + (signed ? (v >= 0 ? "pos" : "neg") : ""), style: `width:${(Math.abs(v) / max * 100).toFixed(1)}%` })),
      h("span", { class: "num small muted" }, (signed && v > 0 ? "+" : "") + v.toFixed(2)));
  }));
}

function network(net, onPick) {
  const W = 640, H = 380, pad = 30;
  const box = svg("svg", { class: "net", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Payment network around this case" });
  if (!net.nodes.length) return box;
  const xs = net.nodes.map(n => n.x), ys = net.nodes.map(n => n.y);
  const sx = x => pad + (x - Math.min(...xs)) / ((Math.max(...xs) - Math.min(...xs)) || 1) * (W - 2 * pad);
  const sy = y => pad + (y - Math.min(...ys)) / ((Math.max(...ys) - Math.min(...ys)) || 1) * (H - 2 * pad);
  const pos = Object.fromEntries(net.nodes.map(n => [n.id, [sx(n.x), sy(n.y)]]));
  // confirmed fraud (analyst label) = solid red; suspicious only (model flagged) = dashed amber; normal = grey
  net.edges.forEach(e => box.append(svg("line", { x1: pos[e.a][0], y1: pos[e.a][1], x2: pos[e.b][0], y2: pos[e.b][1],
    stroke: e.confirmed ? "#B91C1C" : e.flagged ? "#B45309" : "#8DB3C4", "stroke-width": e.confirmed ? 2.4 : e.flagged ? 1.6 : 1,
    "stroke-dasharray": !e.confirmed && e.flagged ? "5 3" : "", "stroke-opacity": e.flagged || e.confirmed ? .9 : .5 })));
  const color = r => r > .5 ? "#B91C1C" : r > .1 ? "#D97706" : "#7CC4E4";
  net.nodes.forEach(n => {
    const [x, y] = pos[n.id], r = n.seed ? 10 : 6 + Math.min(n.flagged, 6);
    const shape = n.kind === "device" ? svg("rect", { x: x - r, y: y - r, width: 2 * r, height: 2 * r, rx: 2 })
      : n.kind === "payee" ? svg("polygon", { points: `${x},${y - r - 2} ${x + r + 2},${y} ${x},${y + r + 2} ${x - r - 2},${y}` })
      : svg("circle", { cx: x, cy: y, r });
    shape.setAttribute("fill", color(n.risk));
    shape.setAttribute("stroke", n.confirmed ? "#B91C1C" : n.seed ? "#082F49" : "#FFFFFF");
    shape.setAttribute("stroke-width", n.confirmed ? 3.5 : n.seed ? 2.5 : 1.5);
    shape.style.cursor = "pointer"; shape.addEventListener("click", () => onPick(n));
    const t = svg("title", {}); t.textContent = `${n.id} · ${n.confirmed ? "CONFIRMED fraud link · " : ""}${n.flagged} flagged · max risk ${(n.risk * 100).toFixed(0)}%`; shape.append(t);
    box.append(shape);
    if (n.seed || n.flagged > 1) { const lab = svg("text", { x, y: y - r - 5, "text-anchor": "middle" }); lab.textContent = n.id.split(":")[1].slice(0, 14); box.append(lab); }
  });
  return box;
}

const NET_LEGEND = () => h("div", { class: "legend" }, h("span", {}, "● customer"), h("span", {}, "■ device"), h("span", {}, "◆ payee"),
  h("span", {}, "fill: red high risk, amber medium, sky low"), h("span", { style: "color:#B91C1C" }, "red outline or solid red line: confirmed fraud (analyst)"),
  h("span", { style: "color:#B45309" }, "dashed amber line: flagged by the model, unconfirmed"), h("span", {}, "dark outline: selected"));

/* Interactive graph explorer: click an entity to see its findings and rings, then expand the graph around it. */
function graphExplorer(initial, title) {
  const wrap = h("div"), info = h("div", { style: "margin-top:10px" }, h("span", { class: "small muted" }, "Click a node to see its evidence; expand to move the graph to it."));
  const trail = [];
  const draw = (net, label) => {
    const back = trail.length ? h("button", { class: "btn ghost sm", onclick: () => { trail.pop(); const prev = trail.pop(); draw(prev.net, prev.label); } }, "← Back") : null;
    trail.push({ net, label });
    wrap.replaceChildren(h("div", { class: "row small muted", style: "margin-bottom:6px" }, back, h("span", {}, label)),
      network(net, n => pickNode(n)), NET_LEGEND(), info);
  };
  const pickNode = n => {
    const expand = h("button", { class: "btn sm", onclick: async () => {
      expand.disabled = true;
      try { const g = await api(`/api/graph/node?id=${encodeURIComponent(n.id)}`); draw(g, `Around ${n.id}`); showRings(g.rings, n); }
      catch (e) { toast(e.message, true); expand.disabled = false; }
    } }, "Expand around this entity");
    info.replaceChildren(h("div", { class: "finding" }, h("b", {}, n.id, n.confirmed ? h("span", { class: "chip BLOCK", style: "margin-left:8px" }, "confirmed fraud link") : n.flagged ? h("span", { class: "chip REVIEW", style: "margin-left:8px" }, "suspicious") : null),
      h("div", {}, n.findings && n.findings.length ? n.findings.join(" · ") : "No suspicious findings for this entity."), h("div", { style: "margin-top:6px" }, expand)));
  };
  const showRings = (rings, n) => { if (rings && rings.length) info.append(...rings.map(r => ringCard(r))); };
  draw(initial, title || "Around this case");
  return wrap;
}

function ringCard(r) {
  return h("div", { class: "finding" },
    h("b", {}, `Ring ${r.id} · score ${r.score} · ${r.size} entities `, h("span", { class: "chip " + (r.status.startsWith("confirmed") ? "BLOCK" : "REVIEW") }, r.status)),
    r.customers != null ? h("div", { class: "small muted" }, `${r.customers} customers · ${r.devices} devices · ${r.payees} payees · ${r.flagged_payments} flagged payments`) : null,
    (r.confirmed_entities || []).length ? h("div", { class: "small" }, "Confirmed fraud entities: " + r.confirmed_entities.join(", ")) : null,
    h("ul", { class: "small", style: "margin:4px 0 0 16px" }, (r.evidence || []).map(e => h("li", {}, e))),
    r.hubs ? h("div", { class: "small muted" }, "Hubs (PageRank + betweenness): " + r.hubs.join(", ")) : null);
}

const ASK = { why: "Why was this flagged?", behaviour: "What changed in the customer's behaviour?", beneficiary: "Is the beneficiary linked to other suspicious accounts?",
  chain: "Show the transaction chain", evidence: "What evidence supports fraud?", similar: "Similar historical cases?", next: "What should I investigate next?" };
function askBox(tid) {
  const out = h("div", { style: "margin-top:10px" });
  const q = h("input", { placeholder: "Or type a question about this case…" });
  const ask = async question => {
    out.replaceChildren(h("p", { class: "muted small" }, "Looking it up in the case data…"));
    try { const a = await api(`/api/cases/${encodeURIComponent(tid)}/ask`, { method: "POST", body: { question } });
      out.replaceChildren(h("p", { class: "small muted" }, a.source === "template" ? "Answer assembled from the case data." : `Worded by ${a.source} from the case data only.`), md(a.answer));
    } catch (e) { out.replaceChildren(h("p", { class: "bad" }, e.message)); }
  };
  return h("div", { style: "margin-top:14px" }, h("div", { class: "small muted", style: "margin-bottom:6px" }, "Ask the copilot (answers use only this application's records; facts and hypotheses are kept apart; it never decides)"),
    h("div", { class: "row", style: "flex-wrap:wrap;gap:6px" }, Object.entries(ASK).map(([k, l]) => h("button", { class: "btn ghost sm", onclick: () => ask(k) }, l))),
    h("form", { class: "row", style: "margin-top:8px", onsubmit: ev => { ev.preventDefault(); if (q.value.trim()) ask(q.value.trim()); } }, q, h("button", { class: "btn sm" }, "Ask")), out);
}

function timelineCard(items) {
  return h("div", { class: "card", style: "margin-top:16px" }, h("h3", {}, "Evidence timeline (48 h before the case)"),
    items.length ? h("table", {}, h("thead", {}, h("tr", {}, ["Time", "Link", "Payment", "Amount", "Decision"].map(x => h("th", {}, x)))),
      h("tbody", {}, items.map(i => h("tr", { class: i.kind === "this case" ? "hl" : "" }, h("td", { class: "mono small" }, (i.ts || "").slice(5, 16) + (i.wall_clock ? " (real time)" : "")),
        h("td", { class: "small" }, i.kind), h("td", { class: "small" }, i.text), h("td", { class: "num small" }, i.amount != null ? rupee(i.amount) : ""),
        h("td", {}, i.decision ? chip(i.decision) : "")))))
      : h("p", { class: "muted" }, "No related payments in the 48 hours before this case."));
}

/* AI analysis of one payment from its evidence pack: automatic for flagged payments, on request otherwise. */
function aiCard(txnId, auto) {
  const body = h("div", { class: "muted small" }, auto ? "Writing the analysis from the evidence…" : "The model allowed this payment. You can still ask for an explanation.");
  const btn = h("button", { class: "btn sm ghost" }, auto ? "Regenerate" : "Explain with AI");
  const run = async () => {
    btn.disabled = true; body.replaceChildren(h("span", { class: "muted small" }, "Writing the analysis from the evidence…"));
    try {
      const s = await api(`/api/cases/${encodeURIComponent(txnId)}/summary`, { method: "POST" });
      body.replaceChildren(h("p", { class: "small muted" }, s.source === "template"
        ? (state.llm && state.llm.configured ? "Template analysis: the AI provider didn't answer (see the note at the end)." : "Template analysis (no AI key configured).")
        : `Written by ${s.source} (${(state.llm && state.llm.model) || "AI"}) from the case evidence only. The model and bank rules decided; the AI explains.`), md(s.summary));
    } catch (e) { body.replaceChildren(h("p", { class: "err-text" }, e.message)); }
    finally { btn.disabled = false; }
  };
  btn.onclick = run;
  const card = h("div", { class: "card ai-analysis" }, h("div", { class: "row" }, h("h3", { style: "margin:0" }, "AI analysis"), h("div", { class: "spacer" }), btn), h("div", { style: "margin-top:10px" }, body));
  if (auto) setTimeout(run, 0);
  return card;
}

async function openCase(tid, el, reload) {
  el.replaceChildren(h("div", { class: "card" }, "Loading case…"));
  let d;
  try { d = await api(`/api/cases/${encodeURIComponent(tid)}`); } catch (e) { el.replaceChildren(h("div", { class: "card" }, e.message)); return; }
  const { pack, payment } = d, t = pack.transaction, m = pack.model, c = pack.customer_history;
  const copilot = h("div", { class: "muted" }, "Generate an evidence-based summary of this case.");
  const genBtn = h("button", { class: "btn sm", onclick: async () => {
    genBtn.disabled = true; genBtn.textContent = "Writing…";
    try { const s = await api(`/api/cases/${encodeURIComponent(tid)}/summary`, { method: "POST" });
      copilot.replaceChildren(h("p", { class: "small muted" }, s.source === "template" ? (state.llm && state.llm.configured ? "Template summary: the AI provider didn't answer, see the note at the end." : "Template summary (no AI key configured).") : `Written by ${s.source} (${(state.llm && state.llm.model) || "AI"}) using only the case evidence.`), md(s.summary));
    } catch (e) { toast(e.message, true); }
    finally { genBtn.disabled = false; genBtn.textContent = "Regenerate"; }
  } }, "Generate summary");
  const note = h("textarea", { placeholder: "Note (required if you overturn a BLOCK)" });
  const decide = async label => {
    try { const r = await api(`/api/cases/${encodeURIComponent(tid)}/decision`, { method: "POST", body: { label, note: note.value } });
      toast(`Recorded: ${label}. Label ${r.label_status}${r.payment_status ? ", payment " + (STATUS_TEXT[r.payment_status] || r.payment_status) : ""}.`);
      reload(); openCase(tid, el, reload);
    } catch (e) { toast(e.message, true); }
  };
  el.replaceChildren(
    h("div", { class: "card" },
      h("div", { class: "row" }, chip(m.decision), h("strong", { style: "font-size:18px" }, rupee(t.amount)), h("span", {}, `${t.user_id} → ${t.merchant_id}`),
        h("div", { class: "spacer" }), chip(payment.status, statusText(payment.status))),
      h("p", { class: "small muted" }, `${t.ts.slice(0, 19)} · ${t.category} · ${t.channel} · device ${t.device_id} · ${t.city} · source ${payment.source} · ${t.txn_id}`),
      h("div", { style: "margin:14px 0 6px" }, h("div", { class: "meter" }, h("i", { style: `left:calc(${(m.risk * 100).toFixed(1)}% - 1px)` }))),
      h("div", { class: "row small muted" }, `Fused risk ${pct(m.risk)}`, `· supervised model ${pct(m.supervised_probability)}`,
        m.tripwires.length ? `· novelty detectors: ${m.tripwires.join(", ")}` : "· no novelty detector fired")),
    h("div", { class: "grid g2", style: "margin-top:16px" },
      h("div", { class: "card" }, h("h3", {}, "Why it was flagged"), policyBox(m.reason_codes, null, m.decision),
        m.reason_codes.filter(r => !isPolicy(r)).length ? bars(m.reason_codes.filter(r => !isPolicy(r)).map(r => ({ ...r, impact: r.impact == null ? 0 : r.impact })), "impact", "reason") : h("p", { class: "muted" }, "No model reason codes."),
        h("p", { class: "small muted", style: "margin-top:8px" }, "SHAP impact on the supervised model (log-odds). Detector-only reasons show 0.")),
      h("div", { class: "card" }, h("h3", {}, "How the risk score was built"), bars(m.risk_breakdown, "contribution", "component", true),
        h("p", { class: "small muted", style: "margin-top:8px" }, "Red pushes risk up, green pulls it down."))),
    h("div", { class: "grid g2", style: "margin-top:16px" },
      h("div", { class: "card" }, h("h3", {}, "Customer history"),
        h("p", {}, `${c.payments_last_30d} payments in 30 days · median ${rupee(c.median_amount)} · this payment ${c.this_amount_vs_median ? c.this_amount_vs_median.toFixed(1) + "× the median" : "–"}`),
        h("p", { class: "small muted" }, `Device seen before: ${c.device_seen_before ? "yes" : "no"} · payee paid before: ${c.payee_paid_before ? "yes" : "no"} · devices used: ${c.devices_seen}`),
        h("table", {}, h("tbody", {}, d.customer_recent.slice(0, 6).map(p => h("tr", {}, h("td", { class: "mono small" }, p.ts.slice(5, 16)), h("td", { class: "small" }, p.payee), h("td", { class: "num small" }, rupee(p.amount)), h("td", { class: "small muted" }, p.device_id)))))),
      h("div", { class: "card" }, h("h3", {}, "Similar past cases (outcome known)"),
        pack.similar_cases.length ? h("table", {}, h("thead", {}, h("tr", {}, ["Case", "Amount", "Decision", "Outcome"].map(x => h("th", {}, x)))),
          h("tbody", {}, pack.similar_cases.map(s => h("tr", {}, h("td", { class: "mono small" }, s.txn_id), h("td", { class: "num" }, rupee(s.amount)), h("td", {}, chip(s.decision)), h("td", { class: "small" }, s.outcome)))))
          : h("p", { class: "muted" }, "No comparable past alerts with known outcomes yet."))),
    (d.rings || []).length ? h("div", { class: "card", style: "margin-top:16px" }, h("h3", {}, "Fraud rings this case belongs to"), d.rings.map(ringCard)) : null,
    timelineCard(d.timeline || []),
    h("div", { class: "card", style: "margin-top:16px" }, h("h3", {}, "Graph explorer (last 7 days)"),
      graphExplorer(d.network, "Around this case"),
      Object.entries(pack.network_findings).filter(([, f]) => f.length).map(([n, f]) => h("div", { class: "finding" }, h("b", {}, n), h("div", {}, f.join(" · ")))),
      pack.money_chains.length ? h("div", { class: "finding" }, h("b", {}, "Money chain"), h("div", {}, [pack.money_chains[0][0].from, ...pack.money_chains[0].map(x => x.to)].join(" → "))) : null),
    h("div", { class: "card", style: "margin-top:16px" },
      h("div", { class: "row" }, h("h3", { style: "margin:0" }, "Investigation copilot"), h("div", { class: "spacer" }), genBtn,
        h("a", { class: "btn ghost sm", href: `/api/cases/${encodeURIComponent(tid)}/report?llm=1`, download: `case_${tid}.md` }, "Download case report")),
      h("div", { style: "margin-top:12px" }, copilot), askBox(tid)),
    h("div", { class: "card", style: "margin-top:16px" }, h("h3", {}, "Analyst decision"),
      d.labels.length ? h("div", { class: "small muted", style: "margin-bottom:8px" }, d.labels.map(l => `${l.time} · ${l.analyst}: ${l.label ? "fraud" : "genuine"} (${l.status})`).join(" | ")) : null,
      note, h("div", { class: "row", style: "margin-top:10px" },
        h("button", { class: "btn crit", onclick: () => decide("fraud") }, "Confirm fraud"),
        h("button", { class: "btn good", onclick: () => decide("genuine") }, "Genuine: release"),
        h("span", { class: "small muted" }, "Contradicting a BLOCK needs a note and a second analyst."))));
}

// ---------------- analyst: check any payment ----------------
function viewCheck() {
  const o = state.options;
  const exs = o.examples || {};
  const [cityIn, cityDl] = cityInput("");
  cityIn.placeholder = "blank = customer's home city";
  const f = { cid: h("input", { placeholder: `e.g. ${exs.customer || ""}`, required: true }), payee: h("input", { placeholder: `e.g. ${exs.merchant || ""}`, required: true }),
    amount: h("input", { type: "number", min: "1", step: "0.01", required: true }), cat: catSelect(),
    ch: chSelect(),
    dev: h("select", {}, h("option", { value: "usual" }, "Customer's usual device"), h("option", { value: "new" }, "New device"), h("option", { value: "custom" }, "Custom device id")),
    cdev: h("input", { placeholder: "custom device id" }), city: cityIn };
  const out = h("div");
  const cidDl = suggest(f.cid, "customer"), payDl = suggest(f.payee, "merchant");
  const field = (l, e, ...more) => h("div", { class: "field" }, h("label", {}, l), e, ...more);
  const form = h("form", { class: "card", onsubmit: async ev => {
    ev.preventDefault();
    try {
      const r = await api("/api/check", { method: "POST", body: { customer_id: f.cid.value, payee: f.payee.value, amount: f.amount.value, category: f.cat.value,
        channel: f.ch.value, device: f.dev.value === "new" ? "this_browser" : f.dev.value, custom_device: f.cdev.value, city: f.city.value } });
      out.replaceChildren(h("div", { class: "card" },
        h("div", { class: "row" }, chip(r.decision), h("strong", {}, `Model risk ${pct(r.risk)}`), h("span", { class: "muted small" }, `${r.latency_ms.toFixed(2)} ms · ${r.txn_id}`)),
        h("div", { class: "small muted", style: "margin-top:10px" }, "Message the customer would see"),
        h("p", { style: "margin-top:2px" }, r.message), policyBox(r.reasons, r.model_decision, r.decision),
        r.reasons.filter(x => !isPolicy(x)).length ? h("div", { style: "margin-top:10px" }, h("h3", {}, "Model reasons"), bars(r.reasons.filter(x => !isPolicy(x)).map(x => ({ ...x, impact: x.impact == null ? 0 : x.impact })), "impact", "reason"))
          : h("p", { class: "muted small" }, "The model found nothing unusual (below its review threshold, no detector fired), so it gives no reasons."),
        r.risk_breakdown.length ? h("div", { style: "margin-top:12px" }, h("h3", {}, "Risk breakdown"), bars(r.risk_breakdown, "contribution", "component", true)) : null,
        r.decision !== "ALLOW" ? h("button", { class: "btn sm", style: "margin-top:12px", onclick: () => { state.selectedCase = r.txn_id; state.tab = "cases"; renderShell(); } }, "Open as case") : null),
        aiCard(r.txn_id, r.decision !== "ALLOW"));
    } catch (e) { toast(e.message, true); }
  } }, h("h2", {}, "Check a payment"), h("p", { class: "small muted" }, "Score a payment for any customer, for example when they phone in. It is recorded and, if flagged, appears in the case queue."),
    field("Customer id", f.cid, cidDl, h("div", { class: "hint" }, "Type an id (SP…) or a city to see customers.")),
    field("Payee (merchant)", f.payee, payDl, h("div", { class: "hint" }, "Type part of a merchant name.")), h("div", { class: "grid g2" }, field("Amount (₹)", f.amount), field("Category", f.cat)),
    h("div", { class: "grid g2" }, field("Channel", f.ch), h("div", { class: "field" }, h("label", {}, "City"), f.city, cityDl)), field("Device", f.dev), f.cdev, h("button", { class: "btn block", style: "margin-top:12px" }, "Check payment"));
  mount(h("h1", {}, "Check a payment"), h("p", { class: "sub" }, "Same engine, same checks as live payments."), h("div", { class: "split" }, form, out));
}

// ---------------- analyst: customer lookup ----------------
function viewLookup() {
  const q = h("input", { placeholder: `Customer id, e.g. ${((state.options || {}).examples || {}).customer || ""}` });
  const out = h("div");
  const go = async ev => {
    ev && ev.preventDefault();
    try {
      const d = await api(`/api/customers/${encodeURIComponent(q.value.trim())}`); const p = d.profile;
      out.replaceChildren(h("div", { class: "grid g4", style: "margin-top:16px" }, kpi("Past payments", p.past_payments), kpi("Usual amount", rupee(p.usual_amount)),
        kpi("Usual device", p.usual_device || "–"), kpi("Usual city", p.usual_city || "–")),
        h("div", { class: "grid g2", style: "margin-top:16px" },
          h("div", { class: "card" }, h("h3", {}, "Payments in this app"), d.payments.length ? h("table", {}, h("tbody", {}, d.payments.map(x => h("tr", {}, h("td", { class: "mono small" }, x.ts), h("td", {}, x.payee), h("td", { class: "num" }, rupee(x.amount)), h("td", {}, chip(x.status, statusText(x.status))))))) : h("p", { class: "muted" }, "None.")),
          h("div", { class: "card" }, h("h3", {}, "Bank records (recent)"), h("table", {}, h("tbody", {}, d.history.map(x => h("tr", {}, h("td", { class: "mono small" }, x.ts.slice(5, 16)), h("td", {}, x.payee), h("td", { class: "num" }, rupee(x.amount)), h("td", { class: "small muted" }, x.device_id))))))));
    } catch (e) { toast(e.message, true); }
  };
  mount(h("h1", {}, "Customer lookup"), h("form", { class: "row", onsubmit: go }, q, suggest(q, "customer"), h("button", { class: "btn" }, "Look up")), out);
}

// ---------------- admin ----------------
async function viewOverview() {
  mount(h("h1", {}, "Overview"), h("div", { class: "card" }, "Loading…"));
  const d = await api("/api/admin/overview");
  const dec = d.decisions, tot = (dec.ALLOW || 0) + (dec.REVIEW || 0) + (dec.BLOCK || 0);
  mount(h("h1", {}, "Overview"), h("p", { class: "sub" }, `Model version ${d.model_version || "–"} · explanations by ${d.llm.provider}`),
    h("div", { class: "grid g4" }, kpi("Payments checked", tot.toLocaleString(), `${dec.REVIEW || 0} review · ${dec.BLOCK || 0} block`),
      kpi("Value protected", rupee(d.protected_value), "held, stopped or confirmed fraud"),
      kpi("Recall (replay labels)", pct(d.replay_recall), "simulated traffic only"), kpi("False-positive rate (replay)", pct(d.replay_fpr, 2), "simulated traffic only")),
    h("div", { class: "grid g3", style: "margin-top:16px" },
      h("div", { class: "card" }, h("h3", {}, "Payment status"), h("div", { class: "row" }, Object.entries(d.payment_status).map(([k, v]) => chip(k, `${statusText(k)}: ${v}`)))),
      h("div", { class: "card" }, h("h3", {}, "Sources"), h("div", { class: "row" }, Object.entries(d.sources).map(([k, v]) => chip("", `${k}: ${v}`)))),
      h("div", { class: "card" }, h("h3", {}, "Analyst labels"), h("div", { class: "row" }, Object.keys(d.feedback).length ? Object.entries(d.feedback).map(([k, v]) => chip("", `${k}: ${v}`)) : h("span", { class: "muted" }, "none yet")))),
    h("div", { class: "card", style: "margin-top:16px" }, h("p", { class: "small muted", style: "margin:0" }, `Average scoring time (last 2,000 payments): ${d.avg_latency_ms ? d.avg_latency_ms.toFixed(2) + " ms" : "–"}. Simulated traffic: ${d.simulator.running ? "running" : "stopped"}.`)));
}

async function viewTraffic() {
  const rate = h("input", { type: "number", min: "10", max: "2000", value: "200" });
  const prog = h("div", { class: "progress" }, h("i", { style: "width:0%" }));
  const info = h("p", { class: "muted" });
  const refresh = async () => { const d = await api("/api/admin/overview"); const s = d.simulator;
    prog.firstChild.style.width = s.total ? (s.processed / s.total * 100).toFixed(1) + "%" : "0%";
    info.textContent = `${s.running ? "Running" : "Stopped"} · ${s.processed.toLocaleString()} / ${s.total.toLocaleString()} payments replayed at ${s.rate}/s`;
    return s.running; };
  const act = async action => { try { await api("/api/admin/simulator", { method: "POST", body: { action, rate: rate.value } }); refresh(); } catch (e) { toast(e.message, true); } };
  const poll = async () => { if (state.tab !== "traffic") return; await refresh().catch(() => {}); setTimeout(poll, 1500); };
  mount(h("h1", {}, "Simulated traffic"), h("p", { class: "sub" }, "Replays the dataset's unseen test period through the live engine as background bank traffic, so the case queue and network fill up. It is labelled 'simulator' everywhere and is never mixed up with real user payments."),
    h("div", { class: "card", style: "max-width:640px" }, h("div", { class: "field" }, h("label", {}, "Payments per second"), rate),
      h("div", { class: "row" }, h("button", { class: "btn", onclick: () => act("start") }, "Start"), h("button", { class: "btn ghost", onclick: () => act("stop") }, "Stop")),
      h("div", { style: "margin-top:16px" }, prog), info));
  poll();
}

async function viewModel() {
  mount(h("h1", {}, "Model evidence"), h("div", { class: "card" }, "Loading…"));
  const d = await api("/api/admin/model");
  const t = d.deployed && d.deployed.test;
  const table = (head, rows) => h("table", {}, h("thead", {}, h("tr", {}, head.map(x => h("th", {}, x)))), h("tbody", {}, rows.map(r => h("tr", {}, r.map((x, i) => h("td", { class: i ? "num" : "" }, x))))));
  const f3 = v => v == null ? "–" : Number(v).toFixed(3);
  const real = d.real && d.real.experiments, ds = d.real && d.real.datasets;
  const row = (name, r) => [name, f3(r.pr_auc), pct(r.review_recall), pct(r.review_precision), pct(r.review_fpr, 2), pct(r.fraud_value_caught_share)];
  const realCard = real ? h("div", { class: "card" }, h("h3", {}, "Real / public data (each dataset's unseen last 25% of time)"),
    table(["Model → tested on", "PR-AUC", "Fraud caught", "Precision", "False alarms", "Fraud value caught"], [
      real.baseline_synthetic_on_sparkov && row("Synthetic-trained (old) → Sparkov", real.baseline_synthetic_on_sparkov),
      real.single_sparkov && row("Sparkov-trained (in the app) → Sparkov", real.single_sparkov),
      real.pooled_sp_ie && row("Sparkov + IEEE combined → Sparkov", real.pooled_sp_ie.sparkov),
      real.transfer_ieee_to_sparkov && row("IEEE-trained → Sparkov", real.transfer_ieee_to_sparkov),
      real.baseline_synthetic_on_ieee && row("Synthetic-trained (old) → IEEE-CIS", real.baseline_synthetic_on_ieee),
      real.single_ieee && row("IEEE, app fields only → IEEE-CIS", real.single_ieee),
      real.pooled_sp_ie && row("Sparkov + IEEE combined → IEEE-CIS", real.pooled_sp_ie.ieee),
      real.transfer_sparkov_to_ieee && row("Sparkov-trained → IEEE-CIS", real.transfer_sparkov_to_ieee),
      d.ieee_native && ["IEEE, all 431 columns (Real data tab) → IEEE-CIS", f3(d.ieee_native.pr_auc), pct(d.ieee_native.review.recall), pct(d.ieee_native.review.precision), pct(d.ieee_native.review.fpr, 2), "–"],
    ].filter(Boolean)),
    h("p", { class: "small muted", style: "margin-top:10px" },
      "PR-AUC: 1.0 is perfect; random guessing scores the fraud rate (Sparkov 0.005, IEEE-CIS 0.035). Sparkov is third-party simulated card data (CC0); IEEE-CIS is real Vesta e-commerce data with anonymised columns. Models did not transfer between datasets, and pooling them lowered Sparkov's PR-AUC, so the app uses the Sparkov-only model and the IEEE-CIS model runs separately in Real data."),
    ds ? h("p", { class: "small muted" }, Object.entries(ds).map(([k, v]) => `${k}: ${v.rows.toLocaleString()} payments, ${(v.fraud_rate * 100).toFixed(2)}% fraud, test ${v.test_rows.toLocaleString()} (${v.test_frauds.toLocaleString()} fraud)`).join(" · ")) : null) : null;
  const driftBox = h("div", { class: "card" }, h("h3", {}, "Concept drift and feedback learning"), h("p", { class: "muted small" }, "Checking…"));
  api("/api/admin/drift").then(dr => driftBox.replaceChildren(driftCard(dr))).catch(e => driftBox.replaceChildren(h("p", { class: "bad" }, e.message)));
  mount(h("h1", {}, "Model evidence"), h("p", { class: "sub" }, `The app runs on: ${d.app_dataset_label || d.app_dataset}. Full detail: reports/PAYTRACE_2_FINAL_EVALUATION.md.`),
    v21Cards(d.v21), driftBox, h("div", { style: "margin-top:16px" }), realCard,
    h("h2", { style: "margin-top:24px" }, "Earlier experiments on our synthetic data"),
    h("p", { class: "small muted" }, "Kept for transparency. These numbers are on synthetic data the old model was built for, and do not carry over to outside data (see above)."),
    t ? h("div", { class: "grid g4" }, kpi("PR-AUC", t.ranking.pr_auc.toFixed(3)), kpi("Recall (review tier)", pct(t.review_tier.recall)), kpi("Precision (review tier)", pct(t.review_tier.precision)), kpi("Block precision", pct(t.block_tier.precision))) : null,
    d.comparison ? h("div", { class: "card", style: "margin-top:16px" }, h("h3", {}, "Model comparison (synthetic data)"),
      table(["Model", "PR-AUC", "Recall", "Precision", "FPR", "Loss"], Object.entries(d.comparison).map(([k, v]) => [k, v.pr_auc.toFixed(3), pct(v.recall), pct(v.precision), pct(v.fpr, 2), rupee(v.loss_inr)]))) : null,
    d.unseen ? h("div", { class: "card", style: "margin-top:16px" }, h("h3", {}, "Fraud pattern removed from training (recall, synthetic data)"),
      table(["Pattern", "Supervised only", "Hybrid system", "After 5 days of labels"], Object.entries(d.unseen).map(([k, v]) => [k, pct(v.supervised, 0), pct(v.hybrid, 0), pct(v.after_5d, 0)]))) : null,
    d.latency ? h("p", { class: "muted small", style: "margin-top:12px" }, `Benchmark: p50 ${d.latency.latency_ms.p50.toFixed(3)} ms, p99 ${d.latency.latency_ms.p99.toFixed(3)} ms, ${Math.round(d.latency.throughput).toLocaleString()} payments/s per core.`) : null);
}

// ---------------- admin: scenario lab ----------------
const FEATURE_LABELS = { amount: "Amount (₹)", hour: "Hour of day", u_txn_count_total: "Customer's past payments", u_secs_since_last: "Seconds since customer's last payment",
  u_cnt_10m: "Customer payments, last 10 min", u_cnt_1h: "Customer payments, last hour", u_cnt_24h: "Customer payments, last 24 h", u_amt_24h: "Customer spend, last 24 h (₹)",
  u_amt_zscore: "Amount vs customer's usual (z-score)", u_amt_ratio_mean: "Amount ÷ customer's average", u_new_device: "New device for customer", u_new_city: "New city for customer",
  u_new_merchant: "First payment to this payee", u_age_hours: "Account age (h)", d_n_users: "Accounts on this device", m_age_hours: "Payee age (h)",
  m_users_24h: "Payee's distinct payers, 24 h", m_new_payer_share_24h: "Payee's first-time payer share, 24 h", u_in_amt_1h: "Money received, last hour (₹)",
  u_passthrough: "Share of received money sent on", m_conf_fraud_rate: "Payee's confirmed-fraud rate", d_conf_fraud_rate: "Device's confirmed-fraud rate" };
const RING_LABELS = { payee_senders_24h: "Payee: distinct senders, 24 h", payee_new_sender_share_24h: "Payee: first-time sender share", payee_age_hours: "Payee age (h)",
  device_accounts_7d: "Device: accounts, 7 days", device_new_accounts_24h: "Device: new accounts, 24 h", received_p2p_1h: "Sender received P2P, last hour (₹)", chain_depth: "Mule-chain depth" };

function stepDetail(st) {
  if (st.error) return h("div", { class: "card" }, h("h3", {}, `Step ${st.step}: declined before scoring`), h("p", {}, st.error));
  const sc = st.scores, tw = Object.entries(sc.tripwires || {});
  const kv = (obj, labels) => h("table", {}, h("tbody", {}, Object.entries(obj).filter(([k]) => labels[k]).map(([k, v]) =>
    h("tr", {}, h("td", { class: "small" }, labels[k]), h("td", { class: "num small" }, typeof v === "number" ? (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString("en-IN") : +v.toFixed(3)) : String(v))))));
  return h("div", { class: "card" },
    h("div", { class: "row" }, h("h3", { style: "margin:0" }, `Step ${st.step} · ${st.note}`), h("div", { class: "spacer" }), h("span", { class: "chip" }, "SIMULATED"), chip(st.decision)),
    h("p", { class: "small muted" }, `${st.input.user_id} → ${st.input.payee} · ${rupee(st.input.amount)} · device ${st.input.device_id} · ${st.input.city} · ${st.ts} · ${st.latency_ms} ms · ${st.txn_id}`),
    h("div", { class: "grid g3", style: "margin-top:10px" },
      h("div", {}, h("h3", {}, "1. Behavioural features"), kv(st.features, FEATURE_LABELS)),
      h("div", {}, h("h3", {}, "2. Model scores"), h("table", {}, h("tbody", {},
          h("tr", {}, h("td", { class: "small" }, "Supervised model (LightGBM, calibrated)"), h("td", { class: "num small" }, pct(sc.supervised_prob, 2))),
          h("tr", {}, h("td", { class: "small" }, "Fused risk"), h("td", { class: "num small" }, pct(sc.fused_risk, 2))),
          h("tr", {}, h("td", { class: "small" }, "Isolation Forest score"), h("td", { class: "num small" }, sc.isolation_forest.toFixed(3))),
          h("tr", {}, h("td", { class: "small" }, "Autoencoder error"), h("td", { class: "num small" }, sc.autoencoder.toFixed(3))),
          ...tw.map(([k, v]) => h("tr", {}, h("td", { class: "small" }, `${k} tripwire (fires above ${v.threshold})`), h("td", { class: "num small" + (v.score > v.threshold ? " bad" : "") }, v.score.toFixed(3)))))),
        h("p", { class: "small muted" }, `Model decision before rules: ${st.model_decision}`)),
      h("div", {}, h("h3", {}, "3. Graph / ring signals"), kv(st.ring, RING_LABELS),
        h("p", { class: "small" }, st.ring.ring_rules.length ? "Ring rules fired: " + st.ring.ring_rules.join(", ") : "No ring rule fired."))),
    h("div", { class: "grid g2", style: "margin-top:12px" },
      h("div", {}, h("h3", {}, "4. Rules and final decision"), st.rules.length ? h("ul", { class: "small" }, st.rules.map(r => h("li", {}, `${r.effect}: ${r.reason}`))) : h("p", { class: "small muted" }, "No bank or ring rule fired."),
        h("p", {}, "Final: ", chip(st.decision), st.case_created ? h("span", { class: "small" }, " → case created for analysts") : h("span", { class: "small muted" }, " → sent, no case"))),
      h("div", {}, h("h3", {}, "5. Explanation"), st.explanation.length ? h("ul", { class: "small" }, st.explanation.map(x => h("li", {}, x))) : h("p", { class: "small muted" }, "Nothing unusual."),
        st.case_created ? h("button", { class: "btn sm", onclick: () => { state.selectedCase = st.txn_id; state.tab = "cases"; renderShell(); } }, "Open the case") : null)));
}

async function viewScenarios() {
  const list = h("div"), steps = h("div"), detail = h("div", { style: "margin-top:16px" });
  let picked = null, lastRun = -1;
  const run = async id => { try { await api("/api/admin/scenarios", { method: "POST", body: { scenario: id } }); picked = null; detail.replaceChildren(); poll(); } catch (e) { toast(e.message, true); } };
  const render = d => {
    list.replaceChildren(h("div", { class: "scen-list" }, d.catalog.map(c => h("div", { class: "scen" + (d.scenario === c.id ? " on" : "") }, h("div", { class: "row" }, h("b", {}, c.title), h("div", { class: "spacer" }),
      h("button", { class: "btn sm", disabled: d.running, onclick: () => run(c.id) }, d.running && d.scenario === c.id ? "Running…" : "Run")), h("p", { class: "small muted" }, c.description)))));
    const cur = d.catalog.find(c => c.id === d.scenario);
    steps.replaceChildren(cur ? h("div", { class: "card", style: "margin-top:16px" },
      h("div", { class: "row" }, h("h3", { style: "margin:0" }, `${cur.title} ${d.running ? "(running…)" : "(finished)"}`), h("div", { class: "spacer" }), h("span", { class: "chip" }, "SIMULATED")),
      d.error ? h("p", { class: "bad" }, d.error) : null,
      h("table", {}, h("thead", {}, h("tr", {}, ["#", "What happens", "Payment", "Amount", "Model", "Ring rules", "Final", ""].map(x => h("th", {}, x)))),
        h("tbody", {}, d.steps.map(st => h("tr", { class: picked === st.step ? "hl" : "" }, h("td", {}, st.step), h("td", { class: "small" }, st.note),
          h("td", { class: "small mono" }, `${st.input.user_id} → ${st.input.payee}`), h("td", { class: "num" }, rupee(st.input.amount)),
          h("td", {}, st.model_decision ? chip(st.model_decision) : "–"), h("td", { class: "small" }, st.ring ? (st.ring.ring_rules.join(", ") || "–") : "–"),
          h("td", {}, chip(st.decision)), h("td", {}, h("button", { class: "btn ghost sm", onclick: () => { picked = st.step; detail.replaceChildren(stepDetail(st)); render(d); } }, "Details"))))))) : null);
    if (d.steps.length && (picked == null || d.run !== lastRun)) { lastRun = d.run; const st = d.steps[d.steps.length - 1]; if (!picked) detail.replaceChildren(stepDetail(st)); }
  };
  const poll = async () => { if (state.tab !== "scenarios") return; try { const d = await api("/api/admin/scenarios"); render(d); if (d.running) setTimeout(poll, 800); } catch (e) { toast(e.message, true); } };
  mount(h("h1", {}, "Scenario lab"), h("p", { class: "sub" }, "Run a SIMULATED fraud scenario. Each payment goes through the live engine exactly like a customer payment (features → models → anomaly detectors → fraud-ring signals → bank rules → decision). Outcomes are not scripted: what you see is what the system decided. Simulated payees and devices are named SIM_…, and every payment is stored as source 'scenario'."),
    list, steps, detail);
  poll();
}

// ---------------- fraud rings ----------------
async function viewRings() {
  mount(h("h1", {}, "Fraud rings"), h("p", { class: "sub" }, "Coordinated activity found in the last days of scored payments: communities (Louvain), where money concentrates (PageRank), brokers between groups (betweenness), circular transfers, mule chains, collector payees and shared devices. Rings are investigation candidates; red marks entities an analyst confirmed as fraud."),
    h("div", { class: "card" }, "Analysing the payment network…"));
  let d;
  try { d = await api("/api/rings?days=7"); } catch (e) { mount(h("h1", {}, "Fraud rings"), h("div", { class: "card" }, e.message)); return; }
  const st = d.stats;
  const explorer = h("div", { class: "card", style: "margin-top:16px" }, h("h3", {}, "Graph explorer"), h("p", { class: "muted small" }, "Pick a ring hub or a top entity to explore its neighbourhood."));
  const explore = async id => {
    try { const g = await api(`/api/graph/node?id=${encodeURIComponent(id)}`);
      explorer.replaceChildren(h("h3", {}, "Graph explorer"), graphExplorer(g, `Around ${id}`), ...(g.rings || []).map(ringCard));
      explorer.scrollIntoView({ behavior: "smooth" }); } catch (e) { toast(e.message, true); }
  };
  const ent = (list, key) => h("table", {}, h("tbody", {}, list.map(x => h("tr", {}, h("td", {}, h("a", { href: "#", onclick: ev => { ev.preventDefault(); explore(x.node); } }, x.node)), h("td", { class: "small muted" }, x.kind), h("td", { class: "num small" }, x[key].toFixed(4))))));
  mount(h("h1", {}, "Fraud rings"), h("p", { class: "sub" }, "Coordinated activity in the last 7 days of scored payments (simulated traffic, scenarios and app payments). Rings are investigation candidates, not verdicts; entities an analyst confirmed as fraud are marked."),
    h("div", { class: "grid g4" }, kpi("Payments analysed", st.payments ? st.payments.toLocaleString() : "0"), kpi("Suspected rings", st.rings), kpi("Circular transfers", st.cycles), kpi("Mule chains", st.chains)),
    h("div", { class: "grid g2", style: "margin-top:16px" },
      h("div", { class: "card" }, h("h3", {}, "Rings, ranked"), d.rings.length ? d.rings.map(r => h("div", {}, ringCard(r), h("button", { class: "btn ghost sm", onclick: () => explore(r.hubs[0]) }, `Explore ${r.hubs[0]}`)))
        : h("p", { class: "muted" }, "No rings in this window. Start Traffic or run a fraud-ring scenario in the Scenario lab.")),
      h("div", {},
        h("div", { class: "card" }, h("h3", {}, "Where money concentrates (PageRank)"), d.top_pagerank.length ? ent(d.top_pagerank, "pagerank") : h("p", { class: "muted" }, "–")),
        h("div", { class: "card", style: "margin-top:16px" }, h("h3", {}, "Brokers between groups (betweenness)"), d.top_betweenness.length ? ent(d.top_betweenness, "betweenness") : h("p", { class: "muted" }, "–")),
        h("div", { class: "card", style: "margin-top:16px" }, h("h3", {}, "Circular transfers and mule chains"),
          d.cycles.length || d.chains.length ? h("ul", { class: "small" }, ...d.cycles.map(c => h("li", {}, `Loop: ${c.accounts.join(" → ")} → back (${c.hours} h)`)),
            ...d.chains.slice(0, 8).map(c => h("li", {}, `Chain: ${[c[0].from, ...c.map(x => x.to)].join(" → ")} (${rupee(c[0].amount)})`))) : h("p", { class: "muted" }, "None found.")))),
    explorer);
}

function driftCard(dr) {
  if (!dr.psi) return h("div", {}, h("h3", {}, "Concept drift and feedback learning"), h("p", { class: "muted" }, dr.status || "No scored events yet. Start Traffic or run scenarios."));
  const f3 = v => v == null ? "–" : v.toFixed(3);
  const fb = dr.feedback;
  return h("div", {},
    h("h3", {}, "Concept drift and feedback learning ", chip(dr.status === "stable" ? "ALLOW" : dr.status === "moderate shift" ? "REVIEW" : "BLOCK", dr.status)),
    h("p", { class: "small muted" }, `Current window ${dr.window.current[0]} → ${dr.window.current[1]} (${dr.window.current[2].toLocaleString()} payments) vs reference ${dr.window.reference[0]} → ${dr.window.reference[1]} (${dr.window.reference[2].toLocaleString()}). PSI < 0.1 stable, 0.1–0.25 moderate, > 0.25 significant.`),
    h("div", { class: "grid g2" },
      h("table", {}, h("thead", {}, h("tr", {}, ["Signal", "PSI", "Level"].map(x => h("th", {}, x)))), h("tbody", {}, Object.entries(dr.psi).map(([k, v]) => h("tr", {}, h("td", {}, k), h("td", { class: "num" }, f3(v)), h("td", { class: "small" }, dr.levels[k] || "–"))))),
      h("table", {}, h("thead", {}, h("tr", {}, ["Rate", "Reference", "Current"].map(x => h("th", {}, x)))), h("tbody", {}, Object.entries(dr.rates).map(([k, v]) => h("tr", {}, h("td", {}, k.replace(/_/g, " ")),
        h("td", { class: "num" }, k === "mean_amount" ? rupee(v.reference) : pct(v.reference, 2)), h("td", { class: "num" }, k === "mean_amount" ? rupee(v.current) : pct(v.current, 2))))))),
    h("p", { class: "small" }, "Page-Hinkley change alarms: " + Object.entries(dr.page_hinkley_alarms).map(([k, v]) => `${k}: ${v.length ? v.map(a => a.at).join(", ") : "none"}`).join(" · ")),
    h("h3", { style: "margin-top:12px" }, "Feedback learning"),
    h("p", { class: "small" }, `Accepted labels: ${fb.accepted_fraud} fraud, ${fb.accepted_genuine} genuine · by status: ${Object.entries(fb.labels_by_status).map(([k, v]) => `${k} ${v}`).join(", ") || "none"} · challenger ${fb.challenger_ready ? "ready to train" : "not ready yet"}`),
    h("ul", { class: "small" }, fb.criteria.map(c => h("li", {}, c))), h("p", { class: "small muted" }, "Run: ", h("code", {}, fb.command)));
}

function v21Cards(v) {
  if (!v) return null;
  const f3 = x => x == null ? "–" : Number(x).toFixed(3);
  const table = (head, rows) => h("table", {}, h("thead", {}, h("tr", {}, head.map(x => h("th", {}, x)))), h("tbody", {}, rows.map(r => h("tr", {}, r.map((x, i) => h("td", { class: i ? "num" : "" }, x))))));
  const cards = [];
  if (v.ieee && v.ieee.experiments) {
    const e = v.ieee.experiments;
    cards.push(h("div", { class: "card" }, h("h3", {}, "IEEE-CIS (real Vesta data) model experiments, unseen last 25%"),
      table(["Experiment", "Learner", "Features", "PR-AUC", "ROC-AUC", "Recall @1% FPR", "Review recall", "Review FPR"],
        Object.entries(e).map(([k, x]) => [k.replace(/_/g, " "), x.learner, x.n_features, f3(x.pr_auc), f3(x.roc_auc), pct(x.recall_at_1pct_fpr), pct(x.review.recall), pct(x.review.fpr, 2)]))));
  }
  if (v.unseen && v.unseen.patterns) {
    const ks = ["lightgbm", "isolation_forest", "autoencoder", "hybrid_2_0", "graph_assisted_2_1"];
    cards.push(h("div", { class: "card", style: "margin-top:16px" }, h("h3", {}, "Previously unseen fraud (pattern removed from training; synthetic UPI data): recall"),
      table(["Held-out pattern", "LightGBM", "Isolation Forest", "Autoencoder", "Hybrid 2.0", "+ fraud rings 2.1"],
        [...Object.entries(v.unseen.patterns).map(([p, r]) => [p, ...ks.map(k => pct(r.detectors[k].recall, 0))]),
         ["mean", ...ks.map(k => pct(v.unseen.mean_recall[k], 0))], ["mean false-positive rate", ...ks.map(k => pct(v.unseen.mean_fpr[k], 2))]])));
  }
  if (v.graph && v.graph.entity_level) {
    const g = v.graph, a = g.transaction_level.rules.any;
    cards.push(h("div", { class: "card", style: "margin-top:16px" }, h("h3", {}, "Fraud-ring detection (synthetic UPI data, test period)"),
      h("p", {}, `Online ring signals: ${pct(a.recall_all_fraud)} of all fraud payments at ${pct(a.fpr, 2)} false positives (QR swap ${pct(a.recall.QR_SWAP, 0)}, device rings ${pct(a.recall.DEVICE_RING, 0)}, rapid movement ${pct(a.recall.RAPID_MOVEMENT, 0)}). Ring reports cover ${pct(g.entity_level.fraud_entity_recall, 0)} of fraud entities; ${pct(g.entity_level.precision, 0)} of ring entities are touched by fraud.`)));
  }
  if (v.fusion) {
    Object.entries(v.fusion).forEach(([ds, f]) => cards.push(h("div", { class: "card", style: "margin-top:16px" }, h("h3", {}, `Risk fusion ablation (${ds}, test period)`),
      table(["Variant", "PR-AUC", "Recall @1% FPR", "Brier", "Review recall", "Review FPR", "Loss"], Object.entries(f.variants).map(([k, x]) => [k.replace(/_/g, " "), f3(x.pr_auc), pct(x.recall_at_1pct_fpr), x.brier.toFixed(4), pct(x.review.recall), pct(x.review.fpr, 2), rupee(x.review.loss_inr)])))));
  }
  return cards.length ? h("div", { style: "margin-bottom:16px" }, ...cards) : null;
}

// ---------------- admin: real-data replay ----------------
async function viewReplay() {
  const rate = h("input", { type: "number", min: "10", max: "5000", value: "300" });
  const body = h("div");
  const act = async action => { try { await api("/api/admin/replay", { method: "POST", body: { action, rate: rate.value } }); draw(); } catch (e) { toast(e.message, true); } };
  const draw = async () => {
    const r = await api("/api/admin/replay"); const st = r.stats;
    if (!r.available) { body.replaceChildren(h("div", { class: "card" }, "The IEEE-CIS model isn't trained on this machine. Run: python -m src.train_ieee_native")); return; }
    const om = r.offline_metrics || {};
    body.replaceChildren(
      h("div", { class: "grid g4" },
        kpi("Real payments replayed", st.processed.toLocaleString(), `of ${Number(r.total || 0).toLocaleString()}`),
        kpi("Fraud caught", st.recall == null ? "–" : pct(st.recall), `${st.caught} of ${st.frauds} real frauds · ${st.missed} missed`),
        kpi("Flags that were real fraud", st.precision == null ? "–" : pct(st.precision), `${st.false_alarms} false alarms`),
        kpi("Fraud value caught", st.value_caught_share == null ? "–" : pct(st.value_caught_share), `${rupee(st.caught_value_inr)} of ${rupee(st.fraud_value_inr)}`)),
      h("div", { class: "progress", style: "margin:14px 0" }, h("i", { style: `width:${r.total ? (st.processed / r.total * 100).toFixed(1) : 0}%` })),
      r.error ? h("p", { class: "err-text" }, r.error) : null,
      h("div", { class: "card" }, h("h3", {}, "Latest flagged payments and real frauds"),
        r.recent.length ? h("table", {}, h("thead", {}, h("tr", {}, ["Time", "Amount", "Decision", "Truth", "Outcome", "Top model reasons (anonymised columns)"].map(x => h("th", {}, x)))),
          h("tbody", {}, r.recent.map(x => h("tr", {}, h("td", { class: "mono small" }, x.time),
            h("td", { class: "num" }, `${rupee(x.amount_inr)}`, h("div", { class: "small muted" }, `$${x.amount_usd}`)),
            h("td", {}, chip(x.decision), h("div", { class: "small muted" }, pct(x.probability, 1))),
            h("td", {}, x.truth), h("td", {}, chip(x.outcome === "caught" ? "completed" : x.outcome === "missed" ? "held" : "review", x.outcome)),
            h("td", { class: "small mono" }, x.reasons.join(", ") || "–")))))
          : h("p", { class: "muted" }, "Press Start to stream real payments through the model.")),
      h("p", { class: "small muted", style: "margin-top:10px" },
        `Offline result on the same unseen period: PR-AUC ${om.pr_auc ? om.pr_auc.toFixed(3) : "–"}, ${om.review ? pct(om.review.recall) : "–"} of fraud caught at ${om.review ? pct(om.review.fpr, 2) : "–"} false alarms. ${r.note || ""}`));
    if (r.running && state.tab === "replay") setTimeout(draw, 1500);
  };
  mount(h("h1", {}, "Real data: IEEE-CIS replay"),
    h("p", { class: "sub" }, "Real card-not-present payments from Vesta (Kaggle IEEE-CIS), from a period the model never saw, scored live and compared with what really happened. Customers don't type these: the data's 431 columns are anonymised bank fields a payment form can't supply."),
    h("div", { class: "card", style: "margin-bottom:16px" }, h("div", { class: "row" }, h("label", { style: "margin:0" }, "Payments per second"), rate,
      h("button", { class: "btn", onclick: () => act("start") }, "Start"), h("button", { class: "btn ghost", onclick: () => act("stop") }, "Pause"),
      h("button", { class: "btn ghost", onclick: () => act("reset") }, "Reset"))),
    body);
  draw();
}

async function viewSystem() {
  mount(h("h1", {}, "System"), h("div", { class: "card" }, "Loading…"));
  const [ov, au] = await Promise.all([api("/api/admin/overview"), api("/api/admin/audit")]);
  const res = h("p", { class: "small" });
  mount(h("h1", {}, "System"),
    h("div", { class: "grid g2" },
      h("div", { class: "card" }, h("h3", {}, "AI explanations (LLM)"),
        h("p", {}, `Provider: ${ov.llm.provider}${ov.llm.model ? " · model " + ov.llm.model : ""}`),
        h("p", { class: "small muted" }, "The LLM only writes explanations from verified evidence; it never decides. Add a free key to the .env file in the project folder and restart: GEMINI_API_KEY=… (aistudio.google.com) or GROQ_API_KEY=… (console.groq.com)."),
        h("button", { class: "btn sm", onclick: async () => { res.textContent = "Testing…"; const r = await api("/api/admin/llm-test", { method: "POST" }); res.textContent = r.ok ? `OK: ${r.provider} replied "${r.reply}"` : `Not working: ${r.error}`; } }, "Test connection"), res),
      h("div", { class: "card" }, h("h3", {}, "Accounts"), h("p", { class: "muted" }, `${au.users.length} accounts.`),
        h("button", { class: "btn ghost sm", onclick: () => { state.tab = "users"; renderShell(); } }, "Manage users"))),
    h("div", { class: "card scroll", style: "margin-top:16px" }, h("h3", {}, "Audit log (latest 200)"),
      h("table", {}, h("thead", {}, h("tr", {}, ["Time (UTC)", "Actor", "Role", "Action", "Target"].map(x => h("th", {}, x)))),
        h("tbody", {}, au.audit.map(a => h("tr", {}, h("td", { class: "mono small" }, a.time), h("td", {}, a.actor), h("td", { class: "small muted" }, a.role), h("td", {}, a.action), h("td", { class: "mono small" }, a.target || "")))))));
}

// ---------------- admin: users ----------------
function secretBox(title, username, password) {
  const copy = h("button", { class: "btn sm ghost", onclick: async () => { try { await navigator.clipboard.writeText(`${username} / ${password}`); toast("Copied"); } catch { toast("Copy failed: select the text instead", true); } } }, "Copy");
  return h("div", { class: "secret" }, h("strong", {}, title),
    h("div", { class: "row", style: "margin-top:8px" }, h("span", { class: "muted small" }, "Username"), h("code", {}, username)),
    h("div", { class: "row" }, h("span", { class: "muted small" }, "Password"), h("code", {}, password), copy),
    h("p", { class: "small muted", style: "margin:8px 0 0" }, "Shown once. Give it to the user; they can change it under My account."));
}

async function viewUsers() {
  const out = h("div");
  const role = h("select", {}, ["analyst", "admin", "customer"].map(r => h("option", { value: r }, r)));
  const uname = h("input", { placeholder: "leave blank for a random username", maxlength: "32" });
  const dname = h("input", { placeholder: "e.g. Priya Sharma", maxlength: "60" });
  const cid = h("input", { placeholder: "existing customer id, e.g. U01022 (blank = new customer)" });
  const cidField = h("div", { class: "field", hidden: true }, h("label", {}, "Customer id"), cid);
  role.onchange = () => cidField.hidden = role.value !== "customer";
  const list = h("div", { class: "card" }, "Loading…");
  const load = async () => {
    const d = await api("/api/admin/users");
    list.replaceChildren(h("h3", {}, `All accounts (${d.users.length})`), h("table", {},
      h("thead", {}, h("tr", {}, ["Username", "Name", "Role", "Customer", "Created", ""].map(x => h("th", {}, x)))),
      h("tbody", {}, d.users.map(u => h("tr", {},
        h("td", { class: "mono small" }, u.username, u.demo ? h("span", { class: "chip", style: "margin-left:6px" }, "demo") : null),
        h("td", {}, u.display_name), h("td", {}, chip(u.role === "admin" ? "BLOCK" : u.role === "analyst" ? "REVIEW" : "", u.role)),
        h("td", { class: "mono small" }, u.customer_id || ""), h("td", { class: "small muted" }, (u.created || "").slice(0, 16)),
        h("td", { class: "num" }, h("div", { class: "row nowrap", style: "justify-content:flex-end" },
          h("button", { class: "btn sm ghost", onclick: async () => {
            try { const r = await api(`/api/admin/users/${encodeURIComponent(u.username)}/reset`, { method: "POST" });
              out.replaceChildren(secretBox("New password", r.username, r.password)); load(); } catch (e) { toast(e.message, true); } } }, "Reset"),
          u.username === state.user.username ? null : h("button", { class: "btn sm ghost danger", onclick: async ev => {
            const b = ev.currentTarget;
            if (b.dataset.armed !== "1") { b.dataset.armed = "1"; b.textContent = "Click again to delete"; setTimeout(() => { b.dataset.armed = ""; b.textContent = "Delete"; }, 3000); return; }
            try { await api(`/api/admin/users/${encodeURIComponent(u.username)}/delete`, { method: "POST" }); toast(`Deleted ${u.username}`); load(); } catch (e) { toast(e.message, true); } } }, "Delete"))))))));
  };
  const form = h("form", { class: "card", onsubmit: async ev => {
    ev.preventDefault();
    try { const r = await api("/api/admin/users/create", { method: "POST", body: { role: role.value, username: uname.value, display_name: dname.value, customer_id: cid.value } });
      out.replaceChildren(secretBox(`Created ${r.role} account`, r.username, r.password)); uname.value = dname.value = cid.value = ""; load();
    } catch (e) { toast(e.message, true); }
  } }, h("h2", {}, "Create an account"),
    h("div", { class: "field" }, h("label", {}, "Role"), role),
    h("div", { class: "field" }, h("label", {}, "Name"), dname),
    h("div", { class: "field" }, h("label", {}, "Username"), uname), cidField,
    h("p", { class: "small muted" }, "A strong random password is generated and shown once."),
    h("button", { class: "btn block" }, "Create account"), h("div", { style: "margin-top:12px" }, out));
  mount(h("h1", {}, "Users"), h("p", { class: "sub" }, "Create staff accounts, reset passwords and remove users. Customers can also sign up themselves."),
    h("div", { class: "split" }, form, list));
  load();
}

// ---------------- everyone: my account ----------------
function viewAccount() {
  const u = state.user;
  const cur = h("input", { type: "password", autocomplete: "current-password", required: true });
  const nw = h("input", { type: "password", autocomplete: "new-password", required: true, minlength: "8" });
  const nw2 = h("input", { type: "password", autocomplete: "new-password", required: true, minlength: "8" });
  const msg = h("p", { class: "small err-text" });
  const form = h("form", { class: "card", onsubmit: async ev => {
    ev.preventDefault(); msg.textContent = "";
    if (nw.value !== nw2.value) { msg.textContent = "The new passwords don't match."; return; }
    try { await api("/api/account/password", { method: "POST", body: { current_password: cur.value, new_password: nw.value } });
      cur.value = nw.value = nw2.value = ""; toast("Password changed"); }
    catch (e) { msg.textContent = e.message; }
  } }, h("h2", {}, "Change password"),
    h("div", { class: "field" }, h("label", {}, "Current password"), cur),
    h("div", { class: "field" }, h("label", {}, "New password"), nw, h("div", { class: "hint" }, "At least 8 characters, with letters and numbers")),
    h("div", { class: "field" }, h("label", {}, "Confirm new password"), nw2), msg, h("button", { class: "btn block" }, "Change password"));
  mount(h("h1", {}, "My account"), h("p", { class: "sub" }, "Your sign-in details."),
    h("div", { class: "split" }, h("div", { class: "card" }, h("h3", {}, "Profile"),
      h("p", {}, u.display_name), h("p", { class: "small muted" }, `Username ${u.username} · role ${u.role}${u.customer_id ? " · account " + u.customer_id : ""}`)), form));
}

boot();
setInterval(() => { if (state.user) tickClock(); }, 30000);
