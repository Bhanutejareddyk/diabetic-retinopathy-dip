/**
 * main.js
 * Client-side controller for Diabetic Retinopathy DIP Web Application.
 */

let selectedFile = null;
let currentReportId = null;
let chartInstance = null;
let cachedHistory = [];
let activeFilter = 'all';

document.addEventListener("DOMContentLoaded", () => {
  initDropZone();
  initAnalyzeButton();
  loadHistoryData();
  // Auto-load Stage 2 sample as a friendly starter view
  loadSample(2, false);
});

/* =====================================================================
   1. DROP ZONE & FILE UPLOAD
   ===================================================================== */
function initDropZone() {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const btnChange = document.getElementById("btn-change-image");

  dropZone.addEventListener("click", (e) => {
    if (e.target !== btnChange) {
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  btnChange.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.value = "";
    fileInput.click();
  });

  // Drag & drop events
  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.add("border-teal-500", "bg-teal-50/40");
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.remove("border-teal-500", "bg-teal-50/40");
    }, false);
  });

  dropZone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    if (dt && dt.files && dt.files[0]) {
      handleSelectedFile(dt.files[0]);
    }
  });
}

function handleSelectedFile(file) {
  selectedFile = file;
  hideError();

  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById("preview-img").src = e.target.result;
    document.getElementById("preview-filename").textContent = file.name;
    document.getElementById("drop-zone-idle").classList.add("hidden");
    document.getElementById("drop-zone-preview").classList.remove("hidden");
  };
  reader.readAsDataURL(file);
}

/* =====================================================================
   2. ANALYSIS & PROGRESS SIMULATION
   ===================================================================== */
function initAnalyzeButton() {
  const btn = document.getElementById("btn-analyze");
  btn.addEventListener("click", () => {
    if (!selectedFile) {
      showError("Please select a retinal image file first or click one of the sample buttons.");
      return;
    }
    executeUploadAnalysis(selectedFile);
  });
}

function showLoader() {
  const loader = document.getElementById("processing-loader");
  const btn = document.getElementById("btn-analyze");
  loader.classList.remove("hidden");
  btn.disabled = true;
  btn.classList.add("opacity-50", "cursor-not-allowed");
}

function hideLoader() {
  const loader = document.getElementById("processing-loader");
  const btn = document.getElementById("btn-analyze");
  loader.classList.add("hidden");
  btn.disabled = false;
  btn.classList.remove("opacity-50", "cursor-not-allowed");
}

function updateProgress(stepText, percent) {
  const textEl = document.getElementById("loader-step-text");
  const barEl = document.getElementById("loader-bar");
  const pctEl = document.getElementById("loader-percentage");

  textEl.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin text-teal-600 mr-2"></i><span>${stepText}</span>`;
  barEl.style.width = `${percent}%`;
  pctEl.textContent = `${percent}%`;
}

function simulateDIPProgress(callback) {
  showLoader();
  updateProgress("Step 1/5: Extracting Green Channel (Peak Hemoglobin Absorption)...", 20);

  setTimeout(() => {
    updateProgress("Step 2/5: Applying CLAHE Local Contrast Equalization...", 45);
    setTimeout(() => {
      updateProgress("Step 3/5: Morphological Vessel Tree Segmentation & Disc Masking...", 70);
      setTimeout(() => {
        updateProgress("Step 4/5: Detecting Microaneurysms & Exudate Clusters...", 90);
        setTimeout(() => {
          callback();
        }, 300);
      }, 350);
    }, 350);
  }, 350);
}

function executeUploadAnalysis(file) {
  hideError();
  const formData = new FormData();
  formData.append("image", file);

  simulateDIPProgress(() => {
    fetch("/api/analyze", {
      method: "POST",
      body: formData
    })
      .then(res => res.json())
      .then(data => {
        hideLoader();
        if (data.success) {
          renderAnalysisResult(data);
          loadHistoryData();
        } else {
          showError(data.error || "Failed to process retinal fundus image.");
        }
      })
      .catch(err => {
        hideLoader();
        showError("Server connection error: " + err.message);
      });
  });
}

/* =====================================================================
   3. SAMPLE CASES (1-CLICK TEST)
   ===================================================================== */
function loadSample(sampleId, scroll = true) {
  hideError();
  simulateDIPProgress(() => {
    fetch(`/api/sample/${sampleId}`)
      .then(res => res.json())
      .then(data => {
        hideLoader();
        if (data.success) {
          renderAnalysisResult(data);
          loadHistoryData();
          if (scroll) {
            document.getElementById("pipeline-section").scrollIntoView({ behavior: 'smooth' });
          }
        } else {
          showError(data.error || "Failed to load sample image.");
        }
      })
      .catch(err => {
        hideLoader();
        showError("Failed to fetch sample: " + err.message);
      });
  });
}

/* =====================================================================
   4. RENDER ANALYSIS RESULTS & PREDICTION
   ===================================================================== */
function renderAnalysisResult(data) {
  currentReportId = data.report_id;
  const pred = data.prediction || {};
  const feats = data.features || {};
  const imgs = data.images || {};
  const ts = new Date().getTime(); // cache buster

  // Update Report ID
  document.getElementById("report-id-tag").textContent = `ID: #${data.report_id}`;

  // Update Diagnostic Verdict Badge
  const stageBadge = document.getElementById("stage-badge");
  stageBadge.textContent = `${pred.name} (${pred.stage})`;
  stageBadge.style.backgroundColor = `${pred.color}15`;
  stageBadge.style.color = pred.color;
  stageBadge.style.borderColor = pred.color;

  document.getElementById("stage-subtext").textContent = `Severity: ${pred.severity} • Clinical Decision: ${pred.urgency}`;

  // Update Confidence Bar
  const conf = pred.confidence || 0.0;
  document.getElementById("confidence-val").textContent = `${conf}%`;
  document.getElementById("confidence-bar").style.width = `${conf}%`;
  document.getElementById("confidence-bar").style.backgroundColor = pred.color;

  // Update Recommendation Box
  const recBox = document.getElementById("recommendation-box");
  recBox.style.borderLeftColor = pred.color;
  recBox.style.borderLeftWidth = "4px";
  document.getElementById("recommendation-text").innerHTML = `<b>Action:</b> ${pred.recommendation}<br/><span class="text-slate-500 text-[11px] mt-1 block"><b>Findings:</b> ${pred.clinical_notes || ''}</span>`;

  // Update Probabilities Bars
  const probs = pred.probabilities || {};
  for (let i = 0; i < 5; i++) {
    const val = probs[String(i)] || 0.0;
    const valEl = document.getElementById(`prob-val-${i}`);
    const barEl = document.getElementById(`prob-bar-${i}`);
    if (valEl) valEl.textContent = `${val}%`;
    if (barEl) barEl.style.width = `${val}%`;
  }

  // Enable PDF download button
  const btnPdf = document.getElementById("btn-download-pdf");
  btnPdf.disabled = false;
  btnPdf.classList.remove("opacity-50", "cursor-not-allowed", "bg-slate-100", "text-slate-400");
  btnPdf.classList.add("bg-teal-50", "text-teal-800", "hover:bg-teal-100", "border-teal-300");
  btnPdf.onclick = () => {
    window.location.href = `/api/report/${currentReportId}`;
  };

  // Update 6 DIP Intermediate Images
  if (imgs.original_url) document.getElementById("img-original").src = `${imgs.original_url}?t=${ts}`;
  if (imgs.green_url) document.getElementById("img-green").src = `${imgs.green_url}?t=${ts}`;
  if (imgs.clahe_url) document.getElementById("img-clahe").src = `${imgs.clahe_url}?t=${ts}`;
  if (imgs.vessel_url) document.getElementById("img-vessels").src = `${imgs.vessel_url}?t=${ts}`;
  if (imgs.lesion_url) document.getElementById("img-lesions").src = `${imgs.lesion_url}?t=${ts}`;
  if (imgs.overlay_url) document.getElementById("img-overlay").src = `${imgs.overlay_url}?t=${ts}`;

  // Update Biomarker Feature Numbers
  document.getElementById("feat-vessel-density").textContent = `${feats.vessel_density ?? '--'}%`;
  document.getElementById("feat-ma-count").textContent = feats.microaneurysm_count ?? '--';
  document.getElementById("feat-ex-count").textContent = feats.exudate_count ?? '--';
  document.getElementById("feat-ex-area").textContent = `${feats.exudate_area_pct ?? '--'}%`;
  document.getElementById("feat-tortuosity").textContent = feats.vessel_tortuosity ?? '--';
  document.getElementById("feat-mean-intensity").textContent = feats.mean_intensity ?? '--';
}

function showError(msg) {
  const alertEl = document.getElementById("error-alert");
  const msgEl = document.getElementById("error-alert-msg");
  msgEl.textContent = msg;
  alertEl.classList.remove("hidden");
}

function hideError() {
  document.getElementById("error-alert").classList.add("hidden");
}

/* =====================================================================
   5. LIGHTBOX MODAL
   ===================================================================== */
function openLightbox(imgElementId, title, description) {
  const src = document.getElementById(imgElementId).src;
  document.getElementById("lightbox-img").src = src;
  document.getElementById("lightbox-title").textContent = title;
  document.getElementById("lightbox-desc").textContent = description;
  document.getElementById("lightbox-modal").classList.remove("hidden");
}

function closeLightbox() {
  document.getElementById("lightbox-modal").classList.add("hidden");
}

// Close on Escape or click backdrop
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeLightbox();
});
document.getElementById("lightbox-modal").addEventListener("click", (e) => {
  if (e.target.id === "lightbox-modal") closeLightbox();
});

/* =====================================================================
   6. HISTORY & STATS DASHBOARD
   ===================================================================== */
function loadHistoryData() {
  fetch("/api/history")
    .then(res => res.json())
    .then(data => {
      if (data.success) {
        cachedHistory = data.history || [];
        updateStatsDashboard(data.stats);
        renderHistoryTable(cachedHistory);
      }
    })
    .catch(err => console.error("Error loading history:", err));
}

function updateStatsDashboard(stats) {
  if (!stats) return;
  document.getElementById("stat-total-scans").textContent = stats.total_scans;
  document.getElementById("stat-normal-scans").textContent = stats.normal_scans;
  document.getElementById("stat-normal-pct").textContent = `${stats.normal_pct}% normal`;
  document.getElementById("stat-dr-scans").textContent = stats.dr_detected;
  document.getElementById("stat-dr-pct").textContent = `${stats.dr_pct}% triage rate`;

  renderStatsChart(stats);
}

function renderStatsChart(stats) {
  const ctx = document.getElementById('drStatsChart');
  if (!ctx || typeof Chart === 'undefined') return;

  const normal = stats.normal_scans || 0;
  const dr = stats.dr_detected || 0;

  if (chartInstance) {
    chartInstance.destroy();
  }

  chartInstance = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Normal Retinas', 'DR Pathology'],
      datasets: [{
        data: [normal, dr],
        backgroundColor: ['#10B981', '#EF4444'],
        borderWidth: 2,
        borderColor: '#ffffff'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'right',
          labels: { boxWidth: 10, font: { size: 10 } }
        }
      },
      cutout: '65%'
    }
  });
}

function renderHistoryTable(items) {
  const tbody = document.getElementById("history-table-body");
  tbody.innerHTML = "";

  if (!items || items.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" class="p-6 text-center text-slate-400">No records found. Analyze an image or test a sample.</td></tr>`;
    return;
  }

  items.forEach(item => {
    const pred = item.prediction || {};
    const feats = item.features || {};
    const tr = document.createElement("tr");
    tr.className = "hover:bg-slate-50 transition";

    const thumbUrl = item.images?.overlay || item.images?.original || "/static/samples/sample_2_moderate.jpg";
    const colorHex = pred.color || "#10B981";

    tr.innerHTML = `
      <td class="p-3">
        <img src="${thumbUrl}" alt="Thumbnail" class="w-10 h-10 object-cover rounded-lg border border-slate-200 cursor-pointer" onclick="viewHistoryItem('${item.report_id}')" />
      </td>
      <td class="p-3 font-mono text-[11px] font-bold text-slate-700">#${item.report_id}</td>
      <td class="p-3">
        <span class="inline-block px-2 py-0.5 rounded text-[11px] font-semibold" style="background-color: ${colorHex}15; color: ${colorHex}; border: 1px solid ${colorHex}40;">
          ${pred.name || 'Unknown'}
        </span>
      </td>
      <td class="p-3 font-semibold text-slate-700">${pred.confidence || 0}%</td>
      <td class="p-3 text-slate-600">
        <span class="text-red-600 font-bold">${feats.microaneurysm_count ?? 0}</span> MA / 
        <span class="text-yellow-600 font-bold">${feats.exudate_count ?? 0}</span> EX
      </td>
      <td class="p-3 text-[11px] text-slate-500">${item.timestamp || 'N/A'}</td>
      <td class="p-3 text-right space-x-2">
        <button onclick="viewHistoryItem('${item.report_id}')" title="Reload Analysis" class="p-1.5 text-teal-600 hover:bg-teal-50 rounded transition">
          <i class="fa-solid fa-eye"></i>
        </button>
        <button onclick="downloadReportById('${item.report_id}')" title="Download PDF" class="p-1.5 text-slate-500 hover:text-red-600 hover:bg-red-50 rounded transition">
          <i class="fa-solid fa-file-pdf"></i>
        </button>
        <button onclick="deleteHistoryItem('${item.report_id}')" title="Delete record" class="p-1.5 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded transition">
          <i class="fa-solid fa-trash-can"></i>
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function viewHistoryItem(reportId) {
  const item = cachedHistory.find(h => h.report_id === reportId);
  if (item) {
    renderAnalysisResult({
      report_id: item.report_id,
      prediction: item.prediction,
      features: item.features,
      images: {
        original_url: item.images.original,
        green_url: item.images.green,
        clahe_url: item.images.clahe,
        vessel_url: item.images.vessels,
        lesion_url: item.images.lesions,
        overlay_url: item.images.overlay
      }
    });
    document.getElementById("pipeline-section").scrollIntoView({ behavior: 'smooth' });
  }
}

function downloadReportById(reportId) {
  window.location.href = `/api/report/${reportId}`;
}

function deleteHistoryItem(reportId) {
  if (!confirm("Are you sure you want to delete this scan record?")) return;
  fetch(`/api/history/${reportId}`, { method: "DELETE" })
    .then(res => res.json())
    .then(() => loadHistoryData())
    .catch(err => console.error(err));
}

function clearAllHistory() {
  if (!confirm("Clear all scan history?")) return;
  fetch("/api/history/clear", { method: "POST" })
    .then(res => res.json())
    .then(() => loadHistoryData())
    .catch(err => console.error(err));
}

function filterHistory(filterType) {
  activeFilter = filterType;
  document.querySelectorAll(".hist-filter-btn").forEach(btn => {
    if (btn.dataset.filter === filterType) {
      btn.className = "hist-filter-btn px-3 py-1 rounded-lg text-xs font-semibold bg-teal-600 text-white";
    } else {
      btn.className = "hist-filter-btn px-3 py-1 rounded-lg text-xs font-semibold bg-slate-100 hover:bg-slate-200 text-slate-700";
    }
  });

  applyFilters();
}

function searchHistory(query) {
  applyFilters(query);
}

function applyFilters(searchQuery = "") {
  let filtered = [...cachedHistory];

  if (activeFilter === "0") {
    filtered = filtered.filter(h => h.prediction?.class_id === 0);
  } else if (activeFilter === "dr") {
    filtered = filtered.filter(h => h.prediction?.class_id > 0);
  }

  const query = (searchQuery || document.getElementById("hist-search").value || "").toLowerCase().trim();
  if (query) {
    filtered = filtered.filter(h =>
      (h.report_id && h.report_id.toLowerCase().includes(query)) ||
      (h.image_name && h.image_name.toLowerCase().includes(query)) ||
      (h.prediction?.name && h.prediction.name.toLowerCase().includes(query))
    );
  }

  renderHistoryTable(filtered);
}
