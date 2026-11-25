(() => {
  const $ = (id) => document.getElementById(id);
  const fmt = new Intl.NumberFormat();
  const adminRequired = document.body.dataset.adminRequired === "true";
  let token = localStorage.getItem("pc_token") || "";

  // ---------- charts ----------
  Chart.defaults.color = "#8b95a3";
  Chart.defaults.borderColor = "#2a313c";
  Chart.defaults.font.family = getComputedStyle(document.documentElement).fontFamily;

  function barChart(canvas, opts = {}) {
    return new Chart(canvas, {
      type: "bar",
      data: { labels: [], datasets: [{ label: "People", data: [], backgroundColor: "#ff5a1f",
        hoverBackgroundColor: "#ff7d4d", borderRadius: 4, maxBarThickness: 48 }] },
      options: {
        responsive: true, maintainAspectRatio: false, animation: { duration: 250 },
        plugins: { legend: { display: false },
          tooltip: { callbacks: { label: (c) => ` ${fmt.format(c.parsed.y)} people` } } },
        scales: {
          x: { grid: { display: false }, ticks: { maxRotation: 0, autoSkip: true, ...(opts.xTick ? { callback: opts.xTick } : {}) } },
          y: { beginAtZero: true, ticks: { precision: 0 } },
        },
      },
    });
  }
  const charts = {
    hour: barChart($("hourChart")),
    day: barChart($("dayChart"), { xTick(v) { return this.getLabelForValue(v).slice(5); } }),
    month: barChart($("monthChart")),
    year: barChart($("yearChart")),
    profile: barChart($("profileChart")),
  };
  function setData(chart, rows) {
    chart.data.labels = rows.map((r) => r.label);
    chart.data.datasets[0].data = rows.map((r) => r.count);
    chart.update();
  }
  const getJSON = (url) => fetch(url).then((r) => r.json());

  // ---------- controls ----------
  const today = () => {
    const d = new Date();
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
  };
  $("hourDate").value = today();
  $("dayMonth").value = today().slice(0, 7);
  $("dayMode").addEventListener("change", () => { $("dayMonth").hidden = $("dayMode").value !== "month"; loadDaily(); });
  $("dayMonth").addEventListener("change", loadDaily);
  $("hourDate").addEventListener("change", loadHourly);
  $("monthYear").addEventListener("change", loadMonthly);

  function loadHourly() { return getJSON(`/api/hourly?date=${$("hourDate").value || today()}`).then((r) => setData(charts.hour, r)); }
  function loadDaily() {
    const url = $("dayMode").value === "month" ? `/api/daily?month=${$("dayMonth").value}` : "/api/daily?days=30";
    return getJSON(url).then((r) => setData(charts.day, r));
  }
  function loadMonthly() {
    const y = $("monthYear").value || new Date().getFullYear();
    return getJSON(`/api/monthly?year=${y}`).then((r) => setData(charts.month, r));
  }
  function loadYearly() {
    return getJSON("/api/yearly").then((rows) => {
      setData(charts.year, rows);
      const sel = $("monthYear"), current = sel.value || String(new Date().getFullYear());
      const years = new Set(rows.map((r) => r.label)); years.add(String(new Date().getFullYear()));
      const sorted = [...years].sort().reverse();
      if (sel.options.length !== sorted.length) {
        sel.innerHTML = sorted.map((y) => `<option${y === current ? " selected" : ""}>${y}</option>`).join("");
      }
    });
  }
  const loadProfile = () => getJSON("/api/profile").then((r) => setData(charts.profile, r));

  function loadSummary() {
    return getJSON("/api/summary").then((s) => {
      $("k_hour").textContent = fmt.format(s.this_hour);
      $("k_today").textContent = fmt.format(s.today);
      $("k_month").textContent = fmt.format(s.this_month);
      $("k_year").textContent = fmt.format(s.this_year);
      $("k_all").textContent = fmt.format(s.all_time);
      $("k_now").textContent = fmt.format(s.now_visible);
      $("k_inout").textContent = s.today_in || s.today_out ? `${fmt.format(s.today_in)} in · ${fmt.format(s.today_out)} out` : "";
      $("k_since").textContent = s.first_event ? `since ${new Date(s.first_event * 1000).toLocaleDateString()}` : "";
      $("k_busiest").textContent = s.busiest_day ? `record: ${fmt.format(s.busiest_day.count)} on ${s.busiest_day.date}` : "";
      $("meta").textContent = `${s.source} · ${s.backend} · ${s.fps} fps · ${s.inference_ms} ms · ${s.status}`;
      $("statusDot").className = "dot " + (s.status === "running" ? "ok" : "err");
      document.querySelectorAll("#viewSeg button").forEach((b) => b.classList.toggle("on", b.dataset.view === s.view));
      $("viewWarn").hidden = !s.view_warning; $("viewWarn").textContent = s.view_warning || "";
    }).catch(() => { $("statusDot").className = "dot err"; $("meta").textContent = "offline"; });
  }

  function refreshAll() {
    loadSummary();
    loadYearly().then(loadMonthly);
    loadHourly(); loadDaily(); loadProfile();
  }
  refreshAll();
  setInterval(loadSummary, 3000);
  setInterval(() => { loadHourly(); loadDaily(); loadYearly().then(loadMonthly); loadProfile(); }, 30000);

  // live feed auto-reconnect
  const video = $("video");
  video.addEventListener("error", () => setTimeout(() => { video.src = "/video_feed?t=" + Date.now(); }, 3000));

  // ---------- camera view switch (angled / top-down) ----------
  document.querySelectorAll("#viewSeg button").forEach((b) => b.addEventListener("click", async () => {
    document.querySelectorAll("#viewSeg button").forEach((x) => x.classList.toggle("on", x === b));
    try { await postSettings({ view: b.dataset.view }); } catch { alert("Could not switch view"); }
    loadSummary();
  }));

  // ---------- settings ----------
  async function postSettings(values) {
    const r = await fetch("/api/settings", { method: "POST",
      headers: { "Content-Type": "application/json", "X-Admin-Token": token }, body: JSON.stringify(values) });
    if (r.status === 401) {
      token = prompt("Admin token") || "";
      localStorage.setItem("pc_token", token);
      if (token) return postSettings(values);
      throw new Error("unauthorised");
    }
    return r.json();
  }

  const dlg = $("settingsDlg"), form = $("settingsForm");
  form.threshold.addEventListener("input", () => { $("thOut").textContent = form.threshold.value; });
  $("tokenRow").hidden = !adminRequired;
  $("settingsBtn").addEventListener("click", async () => {
    const [s, mres] = await Promise.all([getJSON("/api/settings"), getJSON("/api/models")]);
    const models = mres.models.filter((m) => m.view === "angled");
    $("modelSelect").innerHTML = models.map((m) => `<option value="${esc(m.id)}">${esc(m.label)}${m.note ? " — " + esc(m.note) : ""}</option>`).join("");
    $("topdownNote").textContent = s.view === "topdown"
      ? (mres.topdown.available ? "Top-down view is on: the overhead YOLO model is used and the box centre must cross the line." : "Top-down view is on, but the top-down model is missing: " + mres.topdown.reason)
      : "Switch to Top-down above the live image if the camera looks straight down.";
    $("anchorRow").hidden = s.view === "topdown";
    form.model.value = s.model; form.tiles.value = String(s.tiles); form.anchor.value = s.anchor || "center";
    form.snapshots.checked = s.snapshots !== false;
    form.mode.value = s.mode; form.count_dirs.value = s.count_dirs; form.invert.checked = !!s.invert;
    form.threshold.value = s.threshold; $("thOut").textContent = s.threshold; form.min_hits.value = s.min_hits;
    form.token.value = token;
    dlg.showModal();
  });
  $("saveBtn").addEventListener("click", async (e) => {
    e.preventDefault();
    if (adminRequired) { token = form.token.value; localStorage.setItem("pc_token", token); }
    try {
      await postSettings({ mode: form.mode.value, count_dirs: form.count_dirs.value, invert: form.invert.checked,
        model: form.model.value, tiles: form.tiles.value, anchor: form.anchor.value, snapshots: form.snapshots.checked,
        threshold: parseFloat(form.threshold.value), min_hits: parseInt(form.min_hits.value, 10) });
      dlg.close();
    } catch { alert("Could not save settings"); }
  });
  $("resetBtn").addEventListener("click", async () => {
    if (!confirm("Delete ALL recorded counts? This cannot be undone.")) return;
    if (adminRequired) { token = form.token.value; localStorage.setItem("pc_token", token); }
    const r = await fetch("/api/reset", { method: "POST", headers: { "X-Admin-Token": token } });
    if (r.ok) { dlg.close(); refreshAll(); } else alert("Reset failed (check admin token)");
  });


  // ---------- camera ----------
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const camDlg = $("cameraDlg"), cf = $("cameraForm");
  let camType = "usb";
  function showType(t) {
    camType = t;
    document.querySelectorAll("#camType button").forEach((b) => b.classList.toggle("on", b.dataset.type === t));
    document.querySelectorAll("#cameraForm .pane").forEach((p) => { p.hidden = p.dataset.pane !== t; });
    $("camTest").hidden = true;
  }
  document.querySelectorAll("#camType button").forEach((b) => b.addEventListener("click", () => showType(b.dataset.type)));

  function hikPreview() {
    const tls = cf.hik_tls.checked;
    cf.hik_port.placeholder = tls ? "322" : "554";
    const port = cf.hik_port.value || (tls ? 322 : 554);
    const stream = { main: "01", sub: "02", third: "03" }[cf.hik_stream.value];
    const user = cf.hik_user.value || "admin";
    $("hikUrl").innerHTML = `URL: <code>${tls ? "rtsps" : "rtsp"}://${esc(user)}:•••@${cf.hik_host.value ? esc(cf.hik_host.value) : "&lt;ip&gt;"}:${port}/Streaming/Channels/${cf.hik_channel.value || 1}${stream}</code><br>` +
      (tls ? "Enable RTSPS/SRTP on the camera first (Configuration → Network → Advanced). If it won't connect, the camera may require SRTP-encrypted video; use plain RTSP on your LAN instead."
           : "For an NVR, channel 1 = first camera. Sub stream is enough for counting and uses far less CPU.");
  }
  ["hik_host", "hik_port", "hik_user", "hik_channel", "hik_stream", "hik_tls"].forEach((n) => cf[n].addEventListener("input", hikPreview));

  function camPayload() {
    const num = (v) => (v === "" || v == null ? null : Number(v));
    if (camType === "usb") return { type: "usb", device: cf.usb_device.value, width: num(cf.usb_width.value), height: num(cf.usb_height.value), fps: num(cf.usb_fps.value) };
    if (camType === "url") return { type: "url", url: cf.url.value.trim(), transport: cf.transport.value };
    if (camType === "hikvision") return { type: "hikvision", hik_host: cf.hik_host.value.trim(), hik_port: num(cf.hik_port.value),
      hik_user: cf.hik_user.value.trim(), hik_pass: cf.hik_pass.value, hik_channel: num(cf.hik_channel.value) || 1,
      hik_stream: cf.hik_stream.value, hik_tls: cf.hik_tls.checked, transport: "tcp" };
    return { type: "demo" };
  }

  async function adminFetch(url, body) {
    const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json", "X-Admin-Token": token }, body: JSON.stringify(body) });
    if (r.status === 401) {
      token = prompt("Admin token") || ""; localStorage.setItem("pc_token", token);
      if (token) return adminFetch(url, body);
    }
    return r.json();
  }

  $("cameraBtn").addEventListener("click", async () => {
    const [cfg, devices] = await Promise.all([getJSON("/api/camera"), getJSON("/api/camera/devices")]);
    const sel = $("usbDevices");
    const opts = devices.map((d) => `<option value="${esc(d.device)}">${esc(d.device)} — ${esc(d.name)}</option>`);
    if (cfg.type === "usb" && !devices.some((d) => d.device === cfg.device)) opts.unshift(`<option value="${esc(cfg.device)}">${esc(cfg.device)}</option>`);
    if (!opts.length) opts.push('<option value="0">/dev/video0 (not detected)</option>');
    sel.innerHTML = opts.join("");
    if (cfg.type === "usb") sel.value = cfg.device;
    cf.usb_width.value = cfg.width || ""; cf.usb_height.value = cfg.height || ""; cf.usb_fps.value = cfg.fps || "";
    cf.url.value = cfg.type === "url" ? cfg.url || "" : "";
    cf.transport.value = cfg.transport || "tcp";
    cf.hik_host.value = cfg.hik_host || ""; cf.hik_port.value = cfg.hik_port || ""; cf.hik_user.value = cfg.hik_user || "admin";
    cf.hik_pass.value = cfg.hik_pass || ""; cf.hik_channel.value = cfg.hik_channel || 1; cf.hik_stream.value = cfg.hik_stream || "main";
    cf.hik_tls.checked = !!cfg.hik_tls;
    hikPreview(); showType(cfg.type || "usb");
    camDlg.showModal();
  });

  $("camTestBtn").addEventListener("click", async () => {
    const box = $("camTest");
    box.hidden = false; box.className = "cam-test"; box.textContent = "Connecting…";
    $("camTestBtn").disabled = true;
    try {
      const r = await adminFetch("/api/camera/test", camPayload());
      if (r.ok) {
        box.className = "cam-test ok";
        box.innerHTML = r.jpeg ? `Connected · ${r.width}×${r.height}<img src="data:image/jpeg;base64,${r.jpeg}">` : (r.note || "OK");
      } else { box.className = "cam-test err"; box.textContent = r.error || "Failed"; }
    } catch { box.className = "cam-test err"; box.textContent = "Request failed"; }
    $("camTestBtn").disabled = false;
  });

  $("camSaveBtn").addEventListener("click", async (e) => {
    e.preventDefault();
    const r = await adminFetch("/api/camera", camPayload());
    if (r.error) { alert(r.error); return; }
    camDlg.close();
    setTimeout(() => { video.src = "/video_feed?t=" + Date.now(); }, 1500);
  });

  // ---------- draw count line on the live image ----------
  const wrap = $("videoWrap");
  let editing = false, firstPoint = null;
  function imagePoint(ev) {
    // account for object-fit: contain letterboxing
    const rect = video.getBoundingClientRect();
    const iw = video.naturalWidth || 16, ih = video.naturalHeight || 9;
    const scale = Math.min(rect.width / iw, rect.height / ih);
    const dw = iw * scale, dh = ih * scale;
    const ox = (rect.width - dw) / 2, oy = (rect.height - dh) / 2;
    const x = (ev.clientX - rect.left - ox) / dw, y = (ev.clientY - rect.top - oy) / dh;
    return [Math.min(1, Math.max(0, x)), Math.min(1, Math.max(0, y))];
  }
  // tool = null | "line" | "zone"
  let tool = null;
  const zoneBox = $("zoneBox");
  function setTool(t) {
    tool = t; firstPoint = null;
    wrap.classList.toggle("editing", !!t);
    $("lineHint").hidden = !t;
    $("lineHint").textContent = t === "zone"
      ? "Click one corner of the area where people walk, then the opposite corner (click twice in the same spot to reset to full frame)"
      : "Click the start point, then the end point of the line";
    $("editLine").textContent = t === "line" ? "Cancel" : "Draw count line";
    $("editZone").textContent = t === "zone" ? "Cancel" : "Detection zone";
    zoneBox.hidden = true;
  }
  $("editLine").addEventListener("click", () => setTool(tool === "line" ? null : "line"));
  $("editZone").addEventListener("click", () => setTool(tool === "zone" ? null : "zone"));
  function toScreen(p) {
    const rect = video.getBoundingClientRect(), wr = wrap.getBoundingClientRect();
    const iw = video.naturalWidth || 16, ih = video.naturalHeight || 9;
    const scale = Math.min(rect.width / iw, rect.height / ih);
    const ox = (rect.width - iw * scale) / 2 + rect.left - wr.left, oy = (rect.height - ih * scale) / 2 + rect.top - wr.top;
    return [ox + p[0] * iw * scale, oy + p[1] * ih * scale];
  }
  wrap.addEventListener("mousemove", (ev) => {
    if (tool !== "zone" || !firstPoint) return;
    const a = toScreen(firstPoint), b = toScreen(imagePoint(ev));
    Object.assign(zoneBox.style, { left: Math.min(a[0], b[0]) + "px", top: Math.min(a[1], b[1]) + "px",
      width: Math.abs(a[0] - b[0]) + "px", height: Math.abs(a[1] - b[1]) + "px" });
    zoneBox.hidden = false;
  });
  wrap.addEventListener("click", async (ev) => {
    if (!tool) return;
    const p = imagePoint(ev);
    if (!firstPoint) { firstPoint = p; $("lineHint").textContent = tool === "zone" ? "Now click the opposite corner" : "Now click the end point"; return; }
    try {
      if (tool === "line") await postSettings({ mode: "line", line: [...firstPoint, ...p] });
      else {
        const tiny = Math.abs(p[0] - firstPoint[0]) < 0.05 || Math.abs(p[1] - firstPoint[1]) < 0.05;
        await postSettings({ zone: tiny ? [0, 0, 1, 1] : [...firstPoint, ...p] });
      }
    } catch { alert("Could not save"); }
    setTool(null);
  });

  // ---------- recent counts (verification strip) ----------
  async function loadRecent() {
    const rows = await getJSON("/api/recent?limit=40");
    const box = $("recent");
    if (!rows.length) { box.innerHTML = '<p class="muted small">No counts yet.</p>'; return; }
    box.innerHTML = rows.map((r) => {
      const t = new Date(r.ts * 1000);
      const time = t.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
      const img = r.snapshot ? `<img loading="lazy" src="/snapshots/${r.id}.jpg" alt="">` : '<div class="noimg">no snapshot</div>';
      return `<div class="snap" data-id="${r.id}">${img}<button class="del" title="Remove this count">×</button>
        <div class="cap"><span class="dir ${esc(r.direction)}">${esc(r.direction)}</span><span>${time}</span></div></div>`;
    }).join("");
  }
  $("recent").addEventListener("click", async (ev) => {
    const btn = ev.target.closest(".del"); if (!btn) return;
    const id = btn.parentElement.dataset.id;
    const r = await fetch(`/api/events/${id}`, { method: "DELETE", headers: { "X-Admin-Token": token } });
    if (r.status === 401) { token = prompt("Admin token") || ""; localStorage.setItem("pc_token", token); return; }
    btn.parentElement.remove(); loadSummary();
  });
  loadRecent();
  setInterval(loadRecent, 5000);
})();
