// TrafficGuard AI Dashboard Controller

let currentJobId = null;
let pollInterval = null;
let activeEventForReview = null;
let isProcessing = false;
let calibrationMedia = null;
let calibrationObjectUrl = null;
let calibrationMode = null;
let calibrationLineStart = null;
let calibrationRoiStart = null;
let uploadedVideoFile = null;
let isUploading = false;
let isStagedUpload = false;
let signalRoiCounter = 0;
let selectedSignalRoiId = null;
let useUploadedVideoForCalibration = null;
let loadedSettings = null;
let settingsSaveQueue = Promise.resolve();
let settingsSaveTimer = null;
let draggingStopEndpoint = null;
let draggingSignalRoiId = null;
let draggingSignalCorner = null;
let lastCalibrationJobId = null;
let settingsLoadedPromise = null;

document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    initHealthCheck();
    loadDashboardStats();
    loadViolationsTable();
    settingsLoadedPromise = loadSettings();
    initUploadHandlers();
    initImageDetectionHandlers();
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
        "image-analysis": ["Snapshot Detection Engine", "Instant YOLO inference, helmet verification & license plate OCR"],
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
            stageVideoUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files && fileInput.files.length > 0) {
            stageVideoUpload(fileInput.files[0]);
        }
    });

    document.getElementById("btn-start-analysis").addEventListener("click", startStagedAnalysis);
    document.getElementById("btn-calibrate-uploaded-video").addEventListener("click", () => {
        switchTab("settings");
        if (uploadedVideoFile && useUploadedVideoForCalibration) {
            useUploadedVideoForCalibration(uploadedVideoFile);
        }
    });
    document.getElementById("btn-use-uploaded-video").addEventListener("click", () => {
        if (uploadedVideoFile && useUploadedVideoForCalibration) {
            useUploadedVideoForCalibration(uploadedVideoFile);
        }
    });
    document.getElementById("btn-return-to-video").addEventListener("click", () => switchTab("video-analysis"));

    document.getElementById("btn-apply-filters").addEventListener("click", loadViolationsTable);
}

async function stageVideoUpload(file) {
    if (isProcessing || isUploading || isStagedUpload) {
        alert("A video analysis job is currently active. Please wait or click 'Reset Analysis' before uploading another file.");
        return;
    }

    hideVideoError();
    const formData = new FormData();
    formData.append("file", file);

    switchTab("video-analysis");
    showProgress("Uploading video file...", 5);
    isUploading = true;

    try {
        const uploadRes = await fetch("/api/video/upload", {
            method: "POST",
            body: formData
        });

        if (!uploadRes.ok) {
            const err = await uploadRes.json();
            showVideoError(`Upload failed: ${err.detail || "Server error"}`);
            hideProgress();
            isUploading = false;
            return;
        }

        const data = await uploadRes.json();
        currentJobId = data.job_id;
        uploadedVideoFile = file;
        isStagedUpload = true;
        document.getElementById("staged-video-name").textContent = file.name;
        document.getElementById("staged-video-status").textContent =
            "Upload complete. Video is ready; calibration is opened automatically.";
        document.getElementById("staged-video-panel").classList.remove("hidden");
        document.getElementById("btn-start-analysis").disabled = true;
        document.getElementById("btn-use-uploaded-video").classList.remove("hidden");
        document.getElementById("progress-status-label").textContent = "Upload complete. Analysis is waiting to start.";
        document.getElementById("progress-percent").textContent = "Ready";
        document.getElementById("progress-fill").style.width = "0%";
        document.getElementById("progress-frames").textContent = "Waiting for Start Analysis";
        document.getElementById("progress-fps").textContent = "";
        document.getElementById("progress-violations").textContent = "";

        lastCalibrationJobId = currentJobId;
        switchTab("settings");
        document.getElementById("calibration-video-link").classList.remove("hidden");
        document.getElementById("calibration-video-name").textContent = file.name;
        document.getElementById("calibration-prediction-status").textContent =
            "Video loaded. Finding a stop-line suggestion and traffic signal boxes...";
        if (!useUploadedVideoForCalibration) {
            throw new Error("Calibration preview is not ready.");
        }
        useUploadedVideoForCalibration(file);
        await loadCalibrationSuggestions(currentJobId);
        if (isStagedUpload && currentJobId === lastCalibrationJobId) {
            document.getElementById("btn-start-analysis").disabled = false;
        }
    } catch (err) {
        console.error("Video upload error:", err);
        showVideoError("Video upload failed. Check server connection.");
        hideProgress();
    } finally {
        isUploading = false;
    }
}

async function loadCalibrationSuggestions(jobId) {
    try {
        const response = await fetch(`/api/video/calibrate/${jobId}`);
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Could not inspect the uploaded video.");
        }
        const suggestions = await response.json();
        if (jobId !== currentJobId) return;

        if (suggestions.signal_rois.length) {
            renderSignalRois(
                suggestions.signal_rois,
                suggestions.stop_lines || suggestions.signal_rois.map(() =>
                    suggestions.stop_line || [0.05, 0.65, 0.95, 0.65]
                )
            );
        } else {
            renderSignalRois(
                [[0.75, 0.05, 0.95, 0.30]],
                [suggestions.stop_line || [0.05, 0.65, 0.95, 0.65]]
            );
        }
        const details = [];
        details.push(suggestions.stop_line
            ? "A computer-vision stop-line suggestion is shown; verify it against the road markings."
            : "No confident stop-line suggestion was found; adjust or draw the line.");
        details.push(suggestions.signal_rois.length
            ? `${suggestions.signal_rois.length} traffic-light box${suggestions.signal_rois.length === 1 ? "" : "es"} suggested by ${suggestions.signal_model}.`
            : `No traffic-light boxes were detected by ${suggestions.signal_model}; add or draw a box.`);
        document.getElementById("calibration-prediction-status").textContent = details.join(" ");
        if (Number.isFinite(suggestions.frame_time_seconds) && calibrationMedia instanceof HTMLVideoElement) {
            calibrationMedia.currentTime = suggestions.frame_time_seconds;
            document.getElementById("calibration-frame-seek").value = suggestions.frame_time_seconds;
        }
        redrawCalibrationFrame();
    } catch (err) {
        console.error("Calibration suggestion failed:", err);
        if (jobId === currentJobId) {
            document.getElementById("calibration-prediction-status").textContent =
                `Automatic suggestions unavailable: ${err.message} You can draw the line and boxes manually.`;
        }
    }
}

async function startStagedAnalysis() {
    if (!currentJobId || !isStagedUpload || isProcessing) return;
    const startButton = document.getElementById("btn-start-analysis");
    startButton.disabled = true;
    hideVideoError();

    try {
        if (settingsLoadedPromise) await settingsLoadedPromise;
        const settingsSaved = await saveSettings(false);
        if (!settingsSaved) {
            throw new Error("Calibration could not be saved. Fix the save error before starting analysis.");
        }
        const processRes = await fetch(`/api/video/process/${currentJobId}`, { method: "POST" });
        if (!processRes.ok) {
            const err = await processRes.json();
            throw new Error(err.detail || "Could not start background video processing.");
        }

        isStagedUpload = false;
        isProcessing = true;
        document.getElementById("staged-video-status").textContent = "Analysis started.";
        showProgress("Starting video analysis...", 0);
        startPollingJob(currentJobId);
    } catch (err) {
        startButton.disabled = false;
        showVideoError(`Could not start analysis: ${err.message}`);
    }
}

async function handleRunSampleDemo() {
    if (isProcessing || isUploading || isStagedUpload) {
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

    if (currentJobId && (isProcessing || isStagedUpload)) {
        fetch(`/api/video/cancel/${currentJobId}`, { method: "POST" }).catch(() => {});
    }

    currentJobId = null;
    isProcessing = false;
    isUploading = false;
    isStagedUpload = false;
    uploadedVideoFile = null;
    lastCalibrationJobId = null;
    document.getElementById("staged-video-panel").classList.add("hidden");
    document.getElementById("staged-video-name").textContent = "";
    document.getElementById("btn-start-analysis").disabled = true;
    document.getElementById("btn-use-uploaded-video").classList.add("hidden");
    document.getElementById("calibration-video-link").classList.add("hidden");

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
        loadedSettings = s;

        document.getElementById("input-conf").value = s.confidence_threshold;
        document.getElementById("val-conf").textContent = s.confidence_threshold;

        document.getElementById("input-skip").value = s.frame_skip;
        document.getElementById("val-skip").textContent = s.frame_skip;

        document.getElementById("check-signal").checked = s.enable_signal_jump;
        document.getElementById("check-triple").checked = s.enable_triple_riding;
        document.getElementById("check-helmet").checked = s.enable_helmet_check;
        document.getElementById("check-plate").checked = s.enable_plate_ocr;
        document.getElementById("check-privacy").checked = s.privacy_blur;

        const signalRois = s.calibration?.signal_rois?.length
            ? s.calibration.signal_rois
            : [s.calibration.signal_roi];
        const stopLines = s.calibration?.stop_lines?.length
            ? s.calibration.stop_lines
            : signalRois.map(() => s.calibration.stop_line);
        renderSignalRois(signalRois, stopLines);

        if (s.calibration && s.calibration.signal_mode) {
            document.getElementById("select-sig-mode").value = s.calibration.signal_mode;
        }
    } catch (err) {
        console.error("Failed to load settings:", err);
    }
}

function initSettingsListeners() {
    initCalibrationCanvas();
    if (!document.getElementById("signal-roi-list").children.length) {
        renderSignalRois([[0.75, 0.05, 0.95, 0.30]], [[0.05, 0.65, 0.95, 0.65]]);
    }
    document.getElementById("btn-add-signal-roi").addEventListener("click", () => {
        const stopLines = readStopLines();
        const previousStopLine = stopLines[stopLines.length - 1] || [0.05, 0.65, 0.95, 0.65];
        addSignalRoiRow([0.75, 0.05, 0.95, 0.30], previousStopLine);
        scheduleCalibrationSave();
    });
    document.getElementById("signal-roi-list").addEventListener("input", event => {
        const input = event.target.closest("[data-stop-coordinate]");
        if (!input) return;
        const row = input.closest(".signal-roi-row");
        const inputs = Array.from(row.querySelectorAll("[data-stop-coordinate]"));
        const index = Number(input.dataset.stopCoordinate);
        const relatedIndex = index === 0 ? 2 : index === 2 ? 0 : index === 1 ? 3 : 1;
        const relatedInput = inputs[relatedIndex];
        if (!relatedInput) return;
        const current = Number(input.value);
        const related = Number(relatedInput.value);
        if ((index === 0 || index === 1) && current >= related) {
            relatedInput.value = Math.min(1, current + 0.001);
        } else if ((index === 2 || index === 3) && current <= related) {
            relatedInput.value = Math.max(0, current - 0.001);
        }
        updateRangeOutput(input);
        updateRangeOutput(relatedInput);
        redrawCalibrationFrame();
        scheduleCalibrationSave();
    });

    const confInput = document.getElementById("input-conf");
    confInput.addEventListener("input", () => {
        document.getElementById("val-conf").textContent = confInput.value;
    });

    const skipInput = document.getElementById("input-skip");
    skipInput.addEventListener("input", () => {
        document.getElementById("val-skip").textContent = skipInput.value;
    });

    document.getElementById("btn-save-settings").addEventListener("click", () => saveSettings(true));
}

function readStopLines() {
    return Array.from(document.querySelectorAll(".signal-roi-row"), row =>
        Array.from(row.querySelectorAll("[data-stop-coordinate]"), input => Number(input.value))
    );
}

function updateRangeOutput(input) {
    const output = input.parentElement.querySelector("output");
    if (output) output.value = Number(input.value).toFixed(3);
}

function createCalibrationSlider(label, value, index) {
    const wrapper = document.createElement("label");
    wrapper.className = "calibration-slider";
    wrapper.innerHTML = `<span>${label}</span><input type="range" min="0" max="1" step="0.001" data-roi-coordinate="${index}"><output></output>`;
    const input = wrapper.querySelector("input");
    input.value = value;
    updateRangeOutput(input);
    input.addEventListener("input", () => {
        updateRangeOutput(input);
        redrawCalibrationFrame();
        scheduleCalibrationSave();
    });
    return wrapper;
}

function renderSignalRois(rois, stopLines = []) {
    const list = document.getElementById("signal-roi-list");
    list.replaceChildren();
    signalRoiCounter = 0;
    (rois.length ? rois : [[0.75, 0.05, 0.95, 0.30]]).forEach((roi, index) =>
        addSignalRoiRow(roi, stopLines[index] || stopLines[0] || [0.05, 0.65, 0.95, 0.65])
    );
    selectedSignalRoiId = list.firstElementChild?.dataset.roiId || null;
}

function addSignalRoiRow(values, stopLine = [0.05, 0.65, 0.95, 0.65]) {
    const list = document.getElementById("signal-roi-list");
    const roiId = `signal-roi-${signalRoiCounter++}`;
    const row = document.createElement("div");
    row.className = "signal-roi-row";
    row.dataset.roiId = roiId;
    row.innerHTML = `
        <div class="signal-roi-heading">
            <strong>Approach ${list.children.length + 1}</strong>
            <div class="header-btns">
                <button type="button" class="btn btn-secondary btn-sm draw-signal-roi">Draw box</button>
                <button type="button" class="btn btn-secondary btn-sm draw-stop-line">Draw stop line</button>
                <button type="button" class="btn btn-secondary btn-sm remove-signal-roi" aria-label="Remove signal box">Remove</button>
            </div>
        </div>
        <div class="calibration-slider-grid signal-roi-sliders"></div>
        <strong class="stop-line-label">Stop line</strong>
        <div class="calibration-slider-grid stop-line-sliders"></div>
    `;
    const sliderGroup = row.querySelector(".signal-roi-sliders");
    ["Left X", "Top Y", "Right X", "Bottom Y"].forEach((label, index) =>
        sliderGroup.appendChild(createCalibrationSlider(label, values[index], index))
    );
    const stopSliderGroup = row.querySelector(".stop-line-sliders");
    ["Start X", "Start Y", "End X", "End Y"].forEach((label, index) => {
        const wrapper = createCalibrationSlider(label, stopLine[index], index);
        wrapper.querySelector("input").dataset.stopCoordinate = index;
        wrapper.querySelector("input").removeAttribute("data-roi-coordinate");
        stopSliderGroup.appendChild(wrapper);
    });
    row.querySelector(".draw-signal-roi").addEventListener("click", () => {
        selectedSignalRoiId = roiId;
        calibrationMode = "signal";
        calibrationRoiStart = null;
        document.getElementById("calibration-help").textContent =
            `Drag a new signal box for approach ${Array.from(list.children).indexOf(row) + 1} onto the preview.`;
    });
    row.querySelector(".draw-stop-line").addEventListener("click", () => {
        selectedSignalRoiId = roiId;
        calibrationMode = "line";
        calibrationLineStart = null;
        document.getElementById("calibration-help").textContent =
            `Click both ends of the stop line for approach ${Array.from(list.children).indexOf(row) + 1}.`;
    });
    row.querySelector(".remove-signal-roi").addEventListener("click", () => {
        if (list.children.length === 1) {
            alert("At least one signal box is required.");
            return;
        }
        row.remove();
        Array.from(list.children).forEach((item, index) => {
            item.querySelector(".signal-roi-heading strong").textContent = `Approach ${index + 1}`;
        });
        if (selectedSignalRoiId === roiId) {
            selectedSignalRoiId = list.firstElementChild.dataset.roiId;
        }
        redrawCalibrationFrame();
        scheduleCalibrationSave();
    });
    list.appendChild(row);
    if (!selectedSignalRoiId) selectedSignalRoiId = roiId;
    redrawCalibrationFrame();
}

function readSignalRois() {
    return Array.from(document.querySelectorAll(".signal-roi-row"), row =>
        Array.from(row.querySelectorAll("[data-roi-coordinate]"), input => Number(input.value))
    );
}

function scheduleCalibrationSave() {
    const status = document.getElementById("calibration-prediction-status");
    if (!status || !document.getElementById("tab-settings").classList.contains("active")) return;
    status.textContent = "Calibration changed — saving automatically…";
    clearTimeout(settingsSaveTimer);
    settingsSaveTimer = setTimeout(() => saveSettings(false), 400);
}

function buildSettingsPayload() {
    if (!loadedSettings) throw new Error("Settings have not finished loading.");
    const signalRois = readSignalRois();
    const stopLines = readStopLines();
    const validCoordinates = values => values.length === 4 &&
        values.every(value => Number.isFinite(value) && value >= 0 && value <= 1);
    if (stopLines.length !== signalRois.length || stopLines.some(line => !validCoordinates(line)) || signalRois.some(roi =>
        !validCoordinates(roi) || roi[2] <= roi[0] || roi[3] <= roi[1]
    )) {
        throw new Error("Coordinates must be between 0 and 1 and each signal box must have positive width and height.");
    }
    return {
        ...loadedSettings,
        confidence_threshold: Number(document.getElementById("input-conf").value),
        frame_skip: Number(document.getElementById("input-skip").value),
        enable_signal_jump: document.getElementById("check-signal").checked,
        enable_triple_riding: document.getElementById("check-triple").checked,
        enable_helmet_check: document.getElementById("check-helmet").checked,
        enable_plate_ocr: document.getElementById("check-plate").checked,
        privacy_blur: document.getElementById("check-privacy").checked,
        calibration: {
            ...loadedSettings.calibration,
            stop_line: stopLines[0],
            stop_lines: stopLines,
            signal_roi: signalRois[0],
            signal_rois: signalRois,
            signal_mode: document.getElementById("select-sig-mode").value
        }
    };
}

async function saveSettings(showAlert = false) {
    clearTimeout(settingsSaveTimer);
    const status = document.getElementById("calibration-prediction-status");
    try {
        const settings = buildSettingsPayload();
        const saveRequest = settingsSaveQueue.catch(() => {}).then(() =>
            fetch("/api/settings", {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(settings)
            })
        );
        settingsSaveQueue = saveRequest;
        const response = await saveRequest;
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Server error while saving settings.");
        }
        loadedSettings = await response.json();
        if (status) status.textContent = "Calibration saved automatically. Changes will be used when analysis starts.";
        if (showAlert) alert("Settings saved successfully!");
        return true;
    } catch (error) {
        console.error("Failed to save settings:", error);
        if (status) status.textContent = `Could not save calibration: ${error.message}`;
        if (showAlert) alert(`Error saving settings: ${error.message}`);
        return false;
    }
}

function redrawCalibrationFrame() {
    if (typeof window.redrawCalibrationFrame === "function") {
        window.redrawCalibrationFrame();
    }
}

function initCalibrationCanvas() {
    const fileInput = document.getElementById("calibration-media-input");
    const video = document.createElement("video");
    const canvas = document.getElementById("calibration-canvas");
    const preview = document.getElementById("calibration-preview");
    const seek = document.getElementById("calibration-frame-seek");
    const help = document.getElementById("calibration-help");
    const context = canvas.getContext("2d");
    let calibrationRoiCurrent = null;

    function showCalibrationFrame() {
        if (!calibrationMedia) return;

        const mediaWidth = video.videoWidth || calibrationMedia.naturalWidth;
        const mediaHeight = video.videoHeight || calibrationMedia.naturalHeight;
        if (!mediaWidth || !mediaHeight || (calibrationMedia === video && video.readyState < 2)) return;

        if (canvas.width !== mediaWidth || canvas.height !== mediaHeight) {
            canvas.width = mediaWidth;
            canvas.height = mediaHeight;
        }
        context.drawImage(calibrationMedia, 0, 0, mediaWidth, mediaHeight);

        readStopLines().forEach((stopLine, index) => {
            if (stopLine.length !== 4 || !stopLine.every(Number.isFinite)) return;
            context.beginPath();
            context.moveTo(stopLine[0] * mediaWidth, stopLine[1] * mediaHeight);
            context.lineTo(stopLine[2] * mediaWidth, stopLine[3] * mediaHeight);
            context.strokeStyle = ["#ff3434", "#00d4ff", "#ff9f43", "#a66cff"][index % 4];
            context.lineWidth = Math.max(4, mediaWidth / 250);
            context.stroke();
            [[stopLine[0], stopLine[1]], [stopLine[2], stopLine[3]]].forEach(([x, y]) => {
                context.beginPath();
                context.arc(x * mediaWidth, y * mediaHeight, Math.max(8, mediaWidth / 100), 0, Math.PI * 2);
                context.fillStyle = ["#ff3434", "#00d4ff", "#ff9f43", "#a66cff"][index % 4];
                context.fill();
                context.strokeStyle = "#ffffff";
                context.lineWidth = 2;
                context.stroke();
            });
            context.fillStyle = context.strokeStyle;
            context.font = `${Math.max(14, mediaWidth / 60)}px sans-serif`;
            context.fillText(`Stop ${index + 1}`, stopLine[0] * mediaWidth, stopLine[1] * mediaHeight - 6);
        });

        readSignalRois().forEach((signalRoi, index) => {
            if (signalRoi.length !== 4 || !signalRoi.every(Number.isFinite)) return;
            const roiId = document.querySelectorAll(".signal-roi-row")[index]?.dataset.roiId;
            const selected = roiId === selectedSignalRoiId;
            context.strokeStyle = selected ? "#ffffff" : "#ffd43b";
            context.lineWidth = Math.max(3, mediaWidth / 500);
            context.strokeRect(
                signalRoi[0] * mediaWidth,
                signalRoi[1] * mediaHeight,
                (signalRoi[2] - signalRoi[0]) * mediaWidth,
                (signalRoi[3] - signalRoi[1]) * mediaHeight
            );
            context.fillStyle = selected ? "#ffffff" : "#ffd43b";
            context.font = `${Math.max(14, mediaWidth / 60)}px sans-serif`;
            context.fillText(`Signal ${index + 1}`, signalRoi[0] * mediaWidth, signalRoi[1] * mediaHeight - 6);
            [[signalRoi[0], signalRoi[1]], [signalRoi[2], signalRoi[3]]].forEach(([x, y]) => {
                context.beginPath();
                context.arc(x * mediaWidth, y * mediaHeight, Math.max(6, mediaWidth / 140), 0, Math.PI * 2);
                context.fill();
            });
        });

        if (calibrationRoiStart && calibrationRoiCurrent && calibrationMode === "signal") {
            context.strokeStyle = "#ffffff";
            context.lineWidth = Math.max(3, mediaWidth / 500);
            context.strokeRect(
                Math.min(calibrationRoiStart.x, calibrationRoiCurrent.x) * mediaWidth,
                Math.min(calibrationRoiStart.y, calibrationRoiCurrent.y) * mediaHeight,
                Math.abs(calibrationRoiCurrent.x - calibrationRoiStart.x) * mediaWidth,
                Math.abs(calibrationRoiCurrent.y - calibrationRoiStart.y) * mediaHeight
            );
        }

        if (calibrationLineStart) {
            context.beginPath();
            context.arc(
                calibrationLineStart.x * mediaWidth,
                calibrationLineStart.y * mediaHeight,
                Math.max(4, mediaWidth / 200),
                0,
                Math.PI * 2
            );
            context.fillStyle = "#ff3434";
            context.fill();
        }
    }

    function getNormalizedPoint(event) {
        const rect = canvas.getBoundingClientRect();
        const scale = Math.min(rect.width / canvas.width, rect.height / canvas.height);
        const drawnWidth = canvas.width * scale;
        const drawnHeight = canvas.height * scale;
        const offsetX = (rect.width - drawnWidth) / 2;
        const offsetY = (rect.height - drawnHeight) / 2;
        return {
            x: Math.min(1, Math.max(0, (event.clientX - rect.left - offsetX) / drawnWidth)),
            y: Math.min(1, Math.max(0, (event.clientY - rect.top - offsetY) / drawnHeight))
        };
    }

    function writeCoordinates(ids, values) {
        ids.forEach((id, index) => {
            const input = typeof id === "string" ? document.getElementById(id) : id;
            input.value = values[index];
            updateRangeOutput(input);
        });
        showCalibrationFrame();
    }

    function setCalibrationFile(file) {
        if (!file) return;

        if (calibrationObjectUrl) URL.revokeObjectURL(calibrationObjectUrl);
        calibrationObjectUrl = URL.createObjectURL(file);
        calibrationMode = null;
        calibrationLineStart = null;
        calibrationRoiStart = null;
        calibrationRoiCurrent = null;
        preview.classList.remove("hidden");
        if (lastCalibrationJobId === currentJobId) {
            document.getElementById("calibration-prediction-status").textContent =
                "Video loaded. Finding suggested stop line and traffic signal boxes...";
        }

        if (file.type.startsWith("video/")) {
            calibrationMedia = video;
            seek.classList.remove("hidden");
            video.onloadedmetadata = () => {
                seek.max = video.duration || 0;
                seek.value = 0;
                video.currentTime = 0;
                showCalibrationFrame();
            };
            video.onseeked = showCalibrationFrame;
            video.src = calibrationObjectUrl;
            video.load();
            help.textContent = "Drag a stop-line endpoint or signal-box handle to edit it. Adjust all positions and box sizes with the sliders.";
        } else if (file.type.startsWith("image/")) {
            seek.classList.add("hidden");
            const image = new Image();
            image.onload = showCalibrationFrame;
            calibrationMedia = image;
            image.src = calibrationObjectUrl;
            help.textContent = "Drag the stop-line endpoints and signal-box handles to edit their positions and sizes.";
        } else {
            preview.classList.add("hidden");
            help.textContent = "Choose a supported image or video file.";
        }
    }

    useUploadedVideoForCalibration = setCalibrationFile;
    fileInput.addEventListener("change", () => setCalibrationFile(fileInput.files[0]));

    seek.addEventListener("input", () => {
        if (calibrationMedia === video) {
            video.currentTime = Number(seek.value);
        }
    });
    video.addEventListener("timeupdate", () => {
        if (calibrationMedia === video) seek.value = video.currentTime;
    });

    window.redrawCalibrationFrame = showCalibrationFrame;

    canvas.addEventListener("pointerdown", event => {
        if (!calibrationMedia) return;
        const point = getNormalizedPoint(event);
        if (calibrationMode === "line") {
            if (!calibrationLineStart) {
                calibrationLineStart = point;
                help.textContent = "Now click the other end of this approach's stop line.";
            } else {
                const row = Array.from(document.querySelectorAll(".signal-roi-row"))
                    .find(item => item.dataset.roiId === selectedSignalRoiId);
                if (!row) {
                    calibrationLineStart = null;
                    calibrationMode = null;
                    return;
                }
                writeCoordinates(
                    Array.from(row?.querySelectorAll("[data-stop-coordinate]") || []),
                    [calibrationLineStart.x, calibrationLineStart.y, point.x, point.y]
                );
                calibrationLineStart = null;
                calibrationMode = null;
                help.textContent = "Stop line updated. Changes save automatically.";
                scheduleCalibrationSave();
            }
        } else if (calibrationMode === "signal") {
            calibrationRoiStart = point;
            calibrationRoiCurrent = point;
            canvas.setPointerCapture(event.pointerId);
        } else {
            let closestStopEndpoint = null;
            let closestStopDistance = canvas.width / 35;
            readStopLines().forEach((line, index) => {
                [[line[0], line[1]], [line[2], line[3]]].forEach(([x, y], endpointIndex) => {
                    const distance = Math.hypot((point.x - x) * canvas.width, (point.y - y) * canvas.height);
                    if (distance < closestStopDistance) {
                        closestStopDistance = distance;
                        closestStopEndpoint = {
                            roiId: document.querySelectorAll(".signal-roi-row")[index].dataset.roiId,
                            endpointIndex
                        };
                    }
                });
            });
            if (closestStopEndpoint) {
                selectedSignalRoiId = closestStopEndpoint.roiId;
                draggingStopEndpoint = closestStopEndpoint;
                canvas.setPointerCapture(event.pointerId);
                return;
            }

            const rois = readSignalRois();
            let nearestCorner = null;
            let nearestDistance = canvas.width / 35;
            rois.forEach((roi, index) => {
                [[roi[0], roi[1]], [roi[2], roi[3]]].forEach(([x, y], cornerIndex) => {
                    const distance = Math.hypot((point.x - x) * canvas.width, (point.y - y) * canvas.height);
                    if (distance < nearestDistance) {
                        nearestDistance = distance;
                        nearestCorner = {
                            roiId: document.querySelectorAll(".signal-roi-row")[index].dataset.roiId,
                            cornerIndex
                        };
                    }
                });
            });
            if (nearestCorner) {
                selectedSignalRoiId = nearestCorner.roiId;
                draggingSignalCorner = nearestCorner;
                canvas.setPointerCapture(event.pointerId);
                return;
            }
            const roiIndex = rois.findIndex(roi =>
                point.x >= roi[0] && point.x <= roi[2] && point.y >= roi[1] && point.y <= roi[3]
            );
            if (roiIndex !== -1) {
                const row = document.querySelectorAll(".signal-roi-row")[roiIndex];
                selectedSignalRoiId = row.dataset.roiId;
                draggingSignalRoiId = row.dataset.roiId;
                calibrationRoiStart = point;
                canvas.setPointerCapture(event.pointerId);
            }
        }
        showCalibrationFrame();
    });

    canvas.addEventListener("pointermove", event => {
        if (!calibrationMedia) return;
        const point = getNormalizedPoint(event);
        if (draggingStopEndpoint !== null) {
            const row = Array.from(document.querySelectorAll(".signal-roi-row"))
                .find(item => item.dataset.roiId === draggingStopEndpoint.roiId);
            if (!row) return;
            const line = Array.from(row.querySelectorAll("[data-stop-coordinate]"), input => Number(input.value));
            line[draggingStopEndpoint.endpointIndex * 2] = point.x;
            line[draggingStopEndpoint.endpointIndex * 2 + 1] = point.y;
            writeCoordinates(Array.from(row.querySelectorAll("[data-stop-coordinate]")), line);
            return;
        }
        if (draggingSignalCorner) {
            const row = Array.from(document.querySelectorAll(".signal-roi-row"))
                .find(item => item.dataset.roiId === draggingSignalCorner.roiId);
            if (!row) return;
            const roiIndex = Array.from(row.parentElement.children).indexOf(row);
            const roi = readSignalRois()[roiIndex];
            if (draggingSignalCorner.cornerIndex === 0) {
                roi[0] = Math.min(point.x, roi[2] - 0.001);
                roi[1] = Math.min(point.y, roi[3] - 0.001);
            } else {
                roi[2] = Math.max(point.x, roi[0] + 0.001);
                roi[3] = Math.max(point.y, roi[1] + 0.001);
            }
            writeCoordinates(Array.from(row.querySelectorAll("[data-roi-coordinate]")), roi);
            return;
        }
        if (draggingSignalRoiId && calibrationRoiStart) {
            const row = Array.from(document.querySelectorAll(".signal-roi-row"))
                .find(item => item.dataset.roiId === draggingSignalRoiId);
            if (!row) return;
            const roiIndex = Array.from(row.parentElement.children).indexOf(row);
            const roi = readSignalRois()[roiIndex];
            const dx = point.x - calibrationRoiStart.x;
            const dy = point.y - calibrationRoiStart.y;
            const width = roi[2] - roi[0];
            const height = roi[3] - roi[1];
            const left = Math.min(1 - width, Math.max(0, roi[0] + dx));
            const top = Math.min(1 - height, Math.max(0, roi[1] + dy));
            writeCoordinates(
                Array.from(row.querySelectorAll("[data-roi-coordinate]")),
                [left, top, left + width, top + height]
            );
            calibrationRoiStart = point;
            return;
        }
        if (calibrationMode === "signal" && calibrationRoiStart) {
            calibrationRoiCurrent = point;
            showCalibrationFrame();
        }
    });

    canvas.addEventListener("pointerup", event => {
        if (draggingStopEndpoint !== null) {
            draggingStopEndpoint = null;
            scheduleCalibrationSave();
            return;
        }
        if (draggingSignalCorner) {
            draggingSignalCorner = null;
            scheduleCalibrationSave();
            return;
        }
        if (draggingSignalRoiId) {
            draggingSignalRoiId = null;
            calibrationRoiStart = null;
            calibrationRoiCurrent = null;
            scheduleCalibrationSave();
            return;
        }
        if (!calibrationRoiStart || calibrationMode !== "signal") return;
        const point = getNormalizedPoint(event);
        const selectedRow = Array.from(document.querySelectorAll(".signal-roi-row"))
            .find(row => row.dataset.roiId === selectedSignalRoiId);
        if (!selectedRow) return;
        writeCoordinates(
            Array.from(selectedRow.querySelectorAll("[data-roi-coordinate]")),
            [
                Math.min(calibrationRoiStart.x, point.x),
                Math.min(calibrationRoiStart.y, point.y),
                Math.max(calibrationRoiStart.x, point.x),
                Math.max(calibrationRoiStart.y, point.y)
            ]
        );
        calibrationRoiStart = null;
        calibrationMode = null;
        help.textContent = "Signal box updated. Changes save automatically.";
        scheduleCalibrationSave();
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

// ==========================================
// Snapshot / Image Detection Controller
// ==========================================

function initImageDetectionHandlers() {
    const fileInput = document.getElementById("image-file-input");
    const dropZone = document.getElementById("image-drop-zone");
    const btnReset = document.getElementById("btn-reset-image");

    if (btnReset) {
        btnReset.addEventListener("click", resetImageAnalysis);
    }

    if (fileInput) {
        fileInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files[0]) {
                handleImageUpload(e.target.files[0]);
            }
        });
    }

    if (dropZone) {
        ['dragenter', 'dragover'].forEach(name => {
            dropZone.addEventListener(name, (e) => {
                e.preventDefault();
                dropZone.classList.add("dragover");
            });
        });
        ['dragleave', 'drop'].forEach(name => {
            dropZone.addEventListener(name, (e) => {
                e.preventDefault();
                dropZone.classList.remove("dragover");
            });
        });
        dropZone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropZone.classList.remove("dragover");
            if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
                handleImageUpload(e.dataTransfer.files[0]);
            }
        });
    }
}

async function handleImageUpload(file) {
    const errorBox = document.getElementById("image-error-box");
    const loadingBox = document.getElementById("image-loading-box");
    const placeholder = document.getElementById("image-placeholder");
    const previewImg = document.getElementById("annotated-image-preview");
    const metricsStrip = document.getElementById("image-metrics-strip");
    const findingsPanel = document.getElementById("image-findings-panel");
    const btnDownload = document.getElementById("btn-download-image");
    const checkDb = document.getElementById("check-save-img-violations");

    errorBox.classList.add("hidden");
    placeholder.classList.add("hidden");
    previewImg.style.display = "none";
    metricsStrip.classList.add("hidden");
    findingsPanel.classList.add("hidden");
    btnDownload.classList.add("hidden");
    loadingBox.classList.remove("hidden");

    try {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("save_violations_to_db", checkDb ? checkDb.checked : true);

        const res = await fetch("/api/image/detect", {
            method: "POST",
            body: formData
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({ detail: "Image inference failed" }));
            throw new Error(errData.detail || `Server error: ${res.status}`);
        }

        const data = await res.json();
        renderImageResults(data);
    } catch (err) {
        errorBox.textContent = `Detection Error: ${err.message}`;
        errorBox.classList.remove("hidden");
        placeholder.classList.remove("hidden");
    } finally {
        loadingBox.classList.add("hidden");
    }
}

function renderImageResults(data) {
    const previewImg = document.getElementById("annotated-image-preview");
    const placeholder = document.getElementById("image-placeholder");
    const metricsStrip = document.getElementById("image-metrics-strip");
    const findingsPanel = document.getElementById("image-findings-panel");
    const findingsContent = document.getElementById("image-findings-content");
    const btnDownload = document.getElementById("btn-download-image");

    // Display image
    previewImg.src = data.annotated_image_url;
    previewImg.style.display = "block";
    placeholder.classList.add("hidden");

    // Configure download button
    btnDownload.href = data.annotated_image_url;
    btnDownload.setAttribute("download", `annotated_${data.filename}`);
    btnDownload.classList.remove("hidden");

    // Update count metrics
    document.getElementById("img-stat-vehicles").textContent = data.counts.vehicles || 0;
    document.getElementById("img-stat-helmets").textContent = `${data.counts.helmets_detected || 0} / ${data.counts.no_helmets_detected || 0}`;
    document.getElementById("img-stat-plates").textContent = data.counts.plates_detected || 0;
    document.getElementById("img-stat-violations").textContent = data.counts.violations || 0;
    metricsStrip.classList.remove("hidden");

    // Build findings details HTML
    let html = `<div style="display: flex; flex-direction: column; gap: 8px;">`;
    
    // Performance badge
    html += `<div style="font-size: 0.85rem; color: var(--text-muted);">
        <span>Resolution: <strong>${data.width}x${data.height}</strong></span> | 
        <span>Inference Time: <strong>${data.inference_time_ms} ms</strong></span>
    </div>`;

    // Violations list
    if (data.violations && data.violations.length > 0) {
        html += `<div style="margin-top: 6px;"><strong>Traffic Violations Detected (${data.violations.length}):</strong></div><ul style="margin: 4px 0 10px 16px; padding: 0;">`;
        data.violations.forEach(v => {
            html += `<li style="color: #ef4444; margin-bottom: 4px;">
                <strong>${formatEventType(v.event_type)}</strong> (Conf: ${(v.confidence * 100).toFixed(0)}%) — ${v.description}
            </li>`;
        });
        html += `</ul>`;
    } else {
        html += `<div style="color: #10b981; margin: 4px 0;"><strong>No Violations Detected:</strong> Vehicles comply with traffic rules.</div>`;
    }

    // License Plates list
    if (data.plate_detections && data.plate_detections.length > 0) {
        html += `<div style="margin-top: 6px;"><strong>License Plates & OCR Results:</strong></div>`;
        html += `<table class="data-table" style="font-size: 0.85rem; margin-top: 4px;">
            <thead>
                <tr>
                    <th>Detected Text</th>
                    <th>Raw OCR</th>
                    <th>Confidence</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>`;
        data.plate_detections.forEach(p => {
            const verifiedBadge = p.is_verified 
                ? `<span class="badge badge-success" style="background: rgba(16,185,129,0.2); color:#10b981; padding:2px 8px; border-radius:4px;">Verified</span>`
                : `<span class="badge" style="background: rgba(156,163,175,0.2); color:#9ca3af; padding:2px 8px; border-radius:4px;">Unverified / Candidate</span>`;
            html += `<tr>
                <td><strong>${p.plate_text}</strong></td>
                <td><code>${p.raw_text || '-'}</code></td>
                <td>${(p.confidence * 100).toFixed(0)}%</td>
                <td>${verifiedBadge}</td>
            </tr>`;
        });
        html += `</tbody></table>`;
    }

    html += `</div>`;
    findingsContent.innerHTML = html;
    findingsPanel.classList.remove("hidden");

    // Refresh violations review and dashboard counters if violations logged
    if (data.counts.violations > 0) {
        loadDashboardStats();
    }
}

function resetImageAnalysis() {
    const fileInput = document.getElementById("image-file-input");
    const previewImg = document.getElementById("annotated-image-preview");
    const placeholder = document.getElementById("image-placeholder");
    const metricsStrip = document.getElementById("image-metrics-strip");
    const findingsPanel = document.getElementById("image-findings-panel");
    const btnDownload = document.getElementById("btn-download-image");
    const errorBox = document.getElementById("image-error-box");
    const loadingBox = document.getElementById("image-loading-box");

    if (fileInput) fileInput.value = "";
    if (previewImg) {
        previewImg.src = "";
        previewImg.style.display = "none";
    }
    if (placeholder) placeholder.classList.remove("hidden");
    if (metricsStrip) metricsStrip.classList.add("hidden");
    if (findingsPanel) findingsPanel.classList.add("hidden");
    if (btnDownload) btnDownload.classList.add("hidden");
    if (errorBox) errorBox.classList.add("hidden");
    if (loadingBox) loadingBox.classList.add("hidden");
}
