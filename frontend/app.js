// TrafficGuard AI Dashboard Controller

let currentJobId = null;
let pollInterval = null;
let activeEventForReview = null;
let isProcessing = false;

document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    initHealthCheck();
    loadDashboardStats();
    loadViolationsTable();
    loadSettings();
    initUploadHandlers();
    initSettingsListeners();

    // Standard Refresh button: Reloads stats and event list without deleting or resetting session
    document.getElementById("btn-refresh").addEventListener("click", () => {
        loadDashboardStats();
        loadViolationsTable();
    });

    // Reset Analysis / New Analysis button
    const btnReset = document.getElementById("btn-reset-analysis");
    if (btnReset) {
        btnReset.addEventListener("click", resetAnalysis);
    }

    // Clear Historical Violations button
    const btnClearHistory = document.getElementById("btn-clear-history");
    if (btnClearHistory) {
        btnClearHistory.addEventListener("click", handleClearHistory);
    }

    // Quick demo video triggers
    document.getElementById("btn-quick-sample").addEventListener("click", handleRunSampleDemo);
    document.getElementById("btn-process-sample").addEventListener("click", handleRunSampleDemo);
});

// Tab Navigation
function initNavigation() {
    const navItems = document.querySelectorAll(".nav-item");
    navItems.forEach(item => {
        item.addEventListener("click", () => {
            const targetTab = item.getAttribute("data-tab");
            switchTab(targetTab);
        });
    });
}

function switchTab(tabId) {
    document.querySelectorAll(".nav-item").forEach(el => el.classList.remove("active"));
    document.querySelectorAll(".tab-pane").forEach(el => el.classList.remove("active"));

    const navBtn = document.querySelector(`.nav-item[data-tab="${tabId}"]`);
    const pane = document.getElementById(`tab-${tabId}`);

    if (navBtn) navBtn.classList.add("active");
    if (pane) pane.classList.add("active");

    const titles = {
        "dashboard": ["Executive Dashboard", "Real-time traffic compliance and violation metrics"],
        "video-analysis": ["Video Analysis Engine", "Process street camera feeds and uploaded traffic footage"],
        "violations": ["Violations Review Center", "Audit AI-flagged candidate violations and verify plate numbers"],
        "settings": ["System Settings & Calibration", "Configure vision thresholds, stop-line coordinates, and privacy"]
    };

    if (titles[tabId]) {
        document.getElementById("page-title").textContent = titles[tabId][0];
        document.getElementById("page-subtitle").textContent = titles[tabId][1];
    }

    if (tabId === "violations") {
        loadViolationsTable();
    }
}

// Health Check
async function initHealthCheck() {
    try {
        const res = await fetch("/api/health");
        if (res.ok) {
            const data = await res.json();
            const devText = data.cuda_available ? `GPU: ${data.device}` : "CPU Inference";
            document.getElementById("system-device-text").textContent = devText;
            document.getElementById("health-device").textContent = data.device;
            document.getElementById("health-cuda").textContent = data.cuda_available ? "Enabled (CUDA)" : "CPU Fallback";
            document.getElementById("model-status-text").textContent = `Model: ${data.model_name}`;
        }
    } catch (err) {
        console.warn("Health check error:", err);
    }
}

// Load Dashboard Statistics
async function loadDashboardStats() {
    try {
        const res = await fetch("/api/statistics");
        if (!res.ok) return;
        const data = await res.json();

        document.getElementById("stat-total").textContent = data.total_violations;
        document.getElementById("stat-red-light").textContent = data.red_light_count;
        document.getElementById("stat-triple-riding").textContent = data.triple_riding_count;
        document.getElementById("stat-no-helmet").textContent = data.no_helmet_count;

        document.getElementById("stat-pending").textContent = data.pending_review_count;
        document.getElementById("stat-confirmed").textContent = data.confirmed_count;
        document.getElementById("stat-dismissed").textContent = data.dismissed_count;

        // Pending badge on sidebar
        document.getElementById("pending-badge").textContent = data.pending_review_count;

        // Populate recent events table
        const tbody = document.getElementById("recent-events-tbody");
        if (data.recent_events && data.recent_events.length > 0) {
            tbody.innerHTML = data.recent_events.map(ev => `
                <tr>
                    <td><code>${ev.id.substring(0, 8)}</code></td>
                    <td><strong>${formatEventType(ev.event_type)}</strong></td>
                    <td>${ev.plate_text && ev.plate_text !== 'UNREADABLE' ? `<code>${ev.plate_text}</code>` : '<span class="text-muted">Unreadable</span>'}</td>
                    <td>${(ev.confidence * 100).toFixed(1)}%</td>
                    <td><span class="badge-${ev.severity.toLowerCase()}">${ev.severity}</span></td>
                    <td><span class="status-${ev.status.toLowerCase()}">${formatStatus(ev.status)}</span></td>
                    <td>${formatDate(ev.timestamp)}</td>
                </tr>
            `).join("");
        } else {
            tbody.innerHTML = `<tr><td colspan="7" class="text-center">No violation events recorded yet.</td></tr>`;
        }
    } catch (err) {
        console.error("Failed to load dashboard statistics:", err);
    }
}

// Load Violations Table
async function loadViolationsTable() {
    const type = document.getElementById("filter-type").value;
    const status = document.getElementById("filter-status").value;
    const search = document.getElementById("filter-search").value;

    let url = `/api/events?page=1&page_size=50`;
    if (type) url += `&event_type=${encodeURIComponent(type)}`;
    if (status) url += `&status=${encodeURIComponent(status)}`;
    if (search) url += `&search=${encodeURIComponent(search)}`;

    try {
        const res = await fetch(url);
        if (!res.ok) return;
        const data = await res.json();
        const tbody = document.getElementById("violations-table-tbody");

        if (data.items && data.items.length > 0) {
            tbody.innerHTML = data.items.map(ev => `
                <tr>
                    <td>
                        <img src="/api/events/${ev.id}/evidence" 
                             alt="Evidence" 
                             style="width: 70px; height: 42px; object-fit: cover; border-radius: 4px; cursor: pointer;"
                             onclick='openReviewModal(${JSON.stringify(ev).replace(/'/g, "&apos;")})'
                             onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 60%22><rect width=%22100%22 height=%2260%22 fill=%22%23334155%22/><text x=%2250%22 y=%2235%22 fill=%22%2394a3b8%22 font-size=%2210%22 text-anchor=%22middle%22>No Image</text></svg>'">
                    </td>
                    <td><strong>${formatEventType(ev.event_type)}</strong></td>
                    <td>${ev.plate_text && ev.plate_text !== 'UNREADABLE' ? `<code>${ev.plate_text}</code>` : '<span class="text-muted">Unreadable</span>'}</td>
                    <td>${(ev.confidence * 100).toFixed(1)}%</td>
                    <td><span class="badge-${ev.severity.toLowerCase()}">${ev.severity}</span></td>
                    <td><span class="status-${ev.status.toLowerCase()}">${formatStatus(ev.status)}</span></td>
                    <td>${formatDate(ev.timestamp)}</td>
                    <td>
                        <button class="btn btn-sm btn-secondary" onclick='openReviewModal(${JSON.stringify(ev).replace(/'/g, "&apos;")})'>
                            <i class="fa-solid fa-eye"></i> Review
                        </button>
                    </td>
                </tr>
            `).join("");
        } else {
            tbody.innerHTML = `<tr><td colspan="8" class="text-center">No violations matching selected criteria.</td></tr>`;
        }
    } catch (err) {
        console.error("Failed to load violations table:", err);
    }
}

// Upload & Video Processing Handlers
function initUploadHandlers() {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("video-file-input");

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "var(--primary)";
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.style.borderColor = "var(--border-color)";
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "var(--border-color)";
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            uploadAndProcessVideo(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files && fileInput.files.length > 0) {
            uploadAndProcessVideo(fileInput.files[0]);
        }
    });

    document.getElementById("btn-apply-filters").addEventListener("click", loadViolationsTable);
}

async function uploadAndProcessVideo(file) {
    if (isProcessing) {
        alert("A video analysis job is currently active. Please wait or click 'Reset Analysis' before uploading another file.");
        return;
    }

    hideVideoError();
    const formData = new FormData();
    formData.append("file", file);

    switchTab("video-analysis");
    showProgress("Uploading video file...", 5);
    isProcessing = true;

    try {
        const uploadRes = await fetch("/api/video/upload", {
            method: "POST",
            body: formData
        });

        if (!uploadRes.ok) {
            const err = await uploadRes.json();
            showVideoError(`Upload failed: ${err.detail || "Server error"}`);
            hideProgress();
            isProcessing = false;
            return;
        }

        const data = await uploadRes.json();
        currentJobId = data.job_id;

        // Trigger processing
        const processRes = await fetch(`/api/video/process/${currentJobId}`, {
            method: "POST"
        });

        if (!processRes.ok) {
            showVideoError("Could not start background video processing.");
            hideProgress();
            isProcessing = false;
            return;
        }

        startPollingJob(currentJobId);
    } catch (err) {
        console.error("Video upload error:", err);
        showVideoError("Video upload failed. Check server connection.");
        hideProgress();
        isProcessing = false;
    }
}

async function handleRunSampleDemo() {
    if (isProcessing) {
        alert("A video analysis job is currently running. Please wait for completion or click 'Reset Analysis'.");
        return;
    }

    hideVideoError();
    switchTab("video-analysis");
    showProgress("Initializing sample demonstration footage...", 10);
    isProcessing = true;

    try {
        const res = await fetch("/api/video/upload-sample", { method: "POST" });
        if (!res.ok) {
            showVideoError("Could not initialize sample demonstration video.");
            hideProgress();
            isProcessing = false;
            return;
        }

        const job = await res.json();
        currentJobId = job.job_id;
        startPollingJob(currentJobId);
    } catch (err) {
        console.error("Demo run error:", err);
        showVideoError("Failed to start sample demonstration: " + err);
        hideProgress();
        isProcessing = false;
    }
}

function startPollingJob(jobId) {
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
        try {
            const res = await fetch(`/api/video/status/${jobId}`);
            if (!res.ok) return;
            const job = await res.json();

            // Ignore stale polling if session has been reset
            if (jobId !== currentJobId) {
                clearInterval(pollInterval);
                return;
            }

            updateProgress(job);

            if (job.status === "COMPLETED") {
                clearInterval(pollInterval);
                onJobCompleted(job);
            } else if (job.status === "FAILED") {
                clearInterval(pollInterval);
                showVideoError(`Video processing failed: ${job.error_message || "Unknown processing error"}`);
                hideProgress();
                isProcessing = false;
            } else if (job.status === "CANCELLED") {
                clearInterval(pollInterval);
                hideProgress();
                isProcessing = false;
            }
        } catch (err) {
            console.error("Error polling job status:", err);
        }
    }, 1000);
}

function showProgress(label, percent) {
    const box = document.getElementById("processing-progress-box");
    box.classList.remove("hidden");
    document.getElementById("progress-status-label").textContent = label;
    document.getElementById("progress-percent").textContent = `${Math.round(percent)}%`;
    document.getElementById("progress-fill").style.width = `${percent}%`;
}

function updateProgress(job) {
    showProgress(`Processing: ${job.status}`, job.progress);
    document.getElementById("progress-frames").textContent = `Frame: ${job.current_frame} / ${job.total_frames}`;
    document.getElementById("progress-fps").textContent = `FPS: ${job.fps ? job.fps.toFixed(1) : '0.0'}`;
    document.getElementById("progress-violations").textContent = `Violations: ${job.violations_count}`;
}

function hideProgress() {
    document.getElementById("processing-progress-box").classList.add("hidden");
}

function showVideoError(msg) {
    const errorBox = document.getElementById("video-error-box");
    if (errorBox) {
        errorBox.textContent = msg;
        errorBox.classList.remove("hidden");
    }
}

function hideVideoError() {
    const errorBox = document.getElementById("video-error-box");
    if (errorBox) {
        errorBox.textContent = "";
        errorBox.classList.add("hidden");
    }
}

function onJobCompleted(job) {
    showProgress("Processing Complete!", 100);
    isProcessing = false;

    const player = document.getElementById("annotated-video-player");
    const placeholder = document.getElementById("video-placeholder");
    const downloadBtn = document.getElementById("btn-download-video");

    const streamUrl = `/api/video/stream/${job.job_id}?t=${Date.now()}`;

    // Configure Download link
    if (downloadBtn) {
        downloadBtn.href = streamUrl;
        downloadBtn.classList.remove("hidden");
    }

    // Attach Video Player
    player.src = streamUrl;
    player.style.display = "block";
    placeholder.style.display = "none";
    hideVideoError();

    player.onerror = () => {
        showVideoError("The browser could not directly render this video stream. You can download and inspect the annotated MP4 file using the button above.");
    };

    player.load();
    player.play().catch(e => {
        console.log("Autoplay was prevented by browser:", e);
    });

    loadDashboardStats();
    loadViolationsTable();
}

// Reset / New Analysis: Clears current video analysis session without wiping historical database
function resetAnalysis() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }

    if (currentJobId && isProcessing) {
        fetch(`/api/video/cancel/${currentJobId}`, { method: "POST" }).catch(() => {});
    }

    currentJobId = null;
    isProcessing = false;

    // Reset file input
    const fileInput = document.getElementById("video-file-input");
    if (fileInput) fileInput.value = "";

    // Reset player
    const player = document.getElementById("annotated-video-player");
    const placeholder = document.getElementById("video-placeholder");
    const downloadBtn = document.getElementById("btn-download-video");

    if (player) {
        player.pause();
        player.removeAttribute("src");
        player.load();
        player.style.display = "none";
    }

    if (placeholder) placeholder.style.display = "flex";
    if (downloadBtn) downloadBtn.classList.add("hidden");

    hideProgress();
    hideVideoError();
    console.log("Video analysis session reset successfully.");
}

// Clear History: Deletes historical violation events from database upon explicit user confirmation
async function handleClearHistory() {
    const confirmed = confirm("Are you sure you want to permanently clear all historical violation events from the database? This action cannot be undone.");
    if (!confirmed) return;

    try {
        const res = await fetch("/api/events/clear?confirm=true", { method: "DELETE" });
        if (res.ok) {
            alert("All historical violation events have been cleared.");
            loadDashboardStats();
            loadViolationsTable();
        } else {
            const err = await res.json();
            alert("Clear failed: " + (err.detail || "Server error"));
        }
    } catch (e) {
        alert("Failed to clear history: " + e);
    }
}

// Modal Review Actions
function openReviewModal(event) {
    activeEventForReview = event;
    const modal = document.getElementById("review-modal");
    
    document.getElementById("modal-title").textContent = `Review Violation: ${formatEventType(event.event_type)}`;
    document.getElementById("modal-event-id").textContent = event.id;
    document.getElementById("modal-event-type").textContent = formatEventType(event.event_type);
    document.getElementById("modal-track-id").textContent = event.tracking_id ? `#${event.tracking_id}` : "N/A";
    document.getElementById("modal-confidence").textContent = `${(event.confidence * 100).toFixed(1)}%`;
    document.getElementById("modal-timestamp").textContent = formatDate(event.timestamp);

    document.getElementById("modal-evidence-img").src = `/api/events/${event.id}/evidence`;
    document.getElementById("modal-plate-input").value = event.plate_text || "";
    document.getElementById("modal-notes-input").value = event.reviewer_notes || "";

    modal.classList.remove("hidden");
}

function closeModal() {
    document.getElementById("review-modal").classList.add("hidden");
    activeEventForReview = null;
}

document.getElementById("btn-modal-confirm").addEventListener("click", async () => {
    if (!activeEventForReview) return;
    await updateEventStatus(activeEventForReview.id, "CONFIRMED_BY_REVIEWER");
    closeModal();
    loadDashboardStats();
    loadViolationsTable();
});

document.getElementById("btn-modal-dismiss").addEventListener("click", async () => {
    if (!activeEventForReview) return;
    await updateEventStatus(activeEventForReview.id, "DISMISSED");
    closeModal();
    loadDashboardStats();
    loadViolationsTable();
});

document.getElementById("btn-update-plate").addEventListener("click", async () => {
    if (!activeEventForReview) return;
    const newPlate = document.getElementById("modal-plate-input").value.trim().toUpperCase();
    const notes = document.getElementById("modal-notes-input").value.trim();

    try {
        const res = await fetch(`/api/events/${activeEventForReview.id}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ plate_text: newPlate, reviewer_notes: notes })
        });
        if (res.ok) {
            alert("Plate number and notes updated successfully!");
            activeEventForReview.plate_text = newPlate;
            loadViolationsTable();
        }
    } catch (err) {
        alert("Failed to update event: " + err);
    }
});

async function updateEventStatus(eventId, newStatus) {
    const notes = document.getElementById("modal-notes-input").value.trim();
    try {
        await fetch(`/api/events/${eventId}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ status: newStatus, reviewer_notes: notes })
        });
    } catch (err) {
        console.error("Error updating event status:", err);
    }
}

// Settings Loading and Saving
async function loadSettings() {
    try {
        const res = await fetch("/api/settings");
        if (!res.ok) return;
        const s = await res.json();

        document.getElementById("input-conf").value = s.confidence_threshold;
        document.getElementById("val-conf").textContent = s.confidence_threshold;

        document.getElementById("input-skip").value = s.frame_skip;
        document.getElementById("val-skip").textContent = s.frame_skip;

        document.getElementById("check-signal").checked = s.enable_signal_jump;
        document.getElementById("check-triple").checked = s.enable_triple_riding;
        document.getElementById("check-helmet").checked = s.enable_helmet_check;
        document.getElementById("check-plate").checked = s.enable_plate_ocr;
        document.getElementById("check-privacy").checked = s.privacy_blur;

        if (s.calibration && s.calibration.stop_line) {
            const sl = s.calibration.stop_line;
            document.getElementById("stop-x1").value = sl[0];
            document.getElementById("stop-y1").value = sl[1];
            document.getElementById("stop-x2").value = sl[2];
            document.getElementById("stop-y2").value = sl[3];
        }

        if (s.calibration && s.calibration.signal_roi) {
            const sr = s.calibration.signal_roi;
            document.getElementById("sig-x1").value = sr[0];
            document.getElementById("sig-y1").value = sr[1];
            document.getElementById("sig-x2").value = sr[2];
            document.getElementById("sig-y2").value = sr[3];
        }

        if (s.calibration && s.calibration.signal_mode) {
            document.getElementById("select-sig-mode").value = s.calibration.signal_mode;
        }
    } catch (err) {
        console.error("Failed to load settings:", err);
    }
}

function initSettingsListeners() {
    const confInput = document.getElementById("input-conf");
    confInput.addEventListener("input", () => {
        document.getElementById("val-conf").textContent = confInput.value;
    });

    const skipInput = document.getElementById("input-skip");
    skipInput.addEventListener("input", () => {
        document.getElementById("val-skip").textContent = skipInput.value;
    });

    document.getElementById("btn-save-settings").addEventListener("click", async () => {
        const newSettings = {
            confidence_threshold: parseFloat(document.getElementById("input-conf").value),
            iou_threshold: 0.45,
            frame_skip: parseInt(document.getElementById("input-skip").value),
            max_processing_fps: 30,
            enable_signal_jump: document.getElementById("check-signal").checked,
            enable_triple_riding: document.getElementById("check-triple").checked,
            enable_helmet_check: document.getElementById("check-helmet").checked,
            enable_plate_ocr: document.getElementById("check-plate").checked,
            privacy_blur: document.getElementById("check-privacy").checked,
            calibration: {
                stop_line: [
                    parseFloat(document.getElementById("stop-x1").value),
                    parseFloat(document.getElementById("stop-y1").value),
                    parseFloat(document.getElementById("stop-x2").value),
                    parseFloat(document.getElementById("stop-y2").value),
                ],
                signal_roi: [
                    parseFloat(document.getElementById("sig-x1").value),
                    parseFloat(document.getElementById("sig-y1").value),
                    parseFloat(document.getElementById("sig-x2").value),
                    parseFloat(document.getElementById("sig-y2").value),
                ],
                signal_mode: document.getElementById("select-sig-mode").value
            }
        };

        try {
            const res = await fetch("/api/settings", {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(newSettings)
            });
            if (res.ok) {
                alert("Settings saved successfully!");
            }
        } catch (err) {
            alert("Error saving settings: " + err);
        }
    });
}

// Helpers
function formatEventType(type) {
    const map = {
        "RED_LIGHT_CROSSING": "Red Light Crossing",
        "TRIPLE_RIDING": "Triple Riding",
        "NO_HELMET": "No Helmet"
    };
    return map[type] || type;
}

function formatStatus(status) {
    const map = {
        "PENDING_REVIEW": "Pending Review",
        "CONFIRMED_BY_REVIEWER": "Confirmed",
        "DISMISSED": "Dismissed"
    };
    return map[status] || status;
}

function formatDate(isoStr) {
    if (!isoStr) return "-";
    const d = new Date(isoStr);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}
