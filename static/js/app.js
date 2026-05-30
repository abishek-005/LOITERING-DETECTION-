/* ═══════════════════════════════════════════════════════════════════════════
   SENTINEL v2.0 — AI SURVEILLANCE SYSTEM
   Frontend Application Controller
   ═══════════════════════════════════════════════════════════════════════════ */

'use strict';

// ─── STATE ─────────────────────────────────────────────────────────────────
let ws = null;                      // WebSocket connection
let reconnectTimer = null;          // Auto-reconnect timer ID
let reconnectAttempts = 0;          // Track reconnect attempts
let isPlaying = false;              // Video playback state
let incidentCount = 0;              // Total incidents tracked
let settingsDebounceTimer = null;   // Debounce timer for settings updates
const MAX_INCIDENTS = 50;           // Max rows in incident log
const RECONNECT_DELAY = 3000;       // Reconnect delay in ms
const SETTINGS_DEBOUNCE = 300;      // Settings update debounce in ms

// ─── DOM REFERENCES ────────────────────────────────────────────────────────
const DOM = {
    // Top bar
    timestamp:          document.getElementById('timestamp'),
    liveIndicator:      document.getElementById('live-indicator'),
    recIndicator:       document.getElementById('rec-indicator'),

    // Video
    videoContainer:     document.getElementById('video-container'),
    videoFeed:          document.getElementById('video-feed'),
    noSignal:           document.getElementById('no-signal'),
    scanLineOverlay:    document.getElementById('scan-line-overlay'),
    hudFrame:           document.getElementById('hud-frame'),
    hudFps:             document.getElementById('hud-fps'),
    hudTimestamp:        document.getElementById('hud-timestamp'),

    // Subject panel
    subjectPanel:       document.getElementById('subject-panel'),
    subjectCards:       document.getElementById('subject-cards'),
    noSubjects:         document.getElementById('no-subjects'),
    panelSubjectCount:  document.getElementById('panel-subject-count'),

    // Upload overlay
    uploadOverlay:      document.getElementById('upload-overlay'),
    dropZone:           document.getElementById('drop-zone'),
    videoUpload:        document.getElementById('video-upload'),
    uploadProgress:     document.getElementById('upload-progress'),
    uploadProgressFill: document.querySelector('.upload-progress-fill'),
    uploadProgressText: document.querySelector('.upload-progress-text'),
    uploadError:        document.getElementById('upload-error'),

    // Controls
    uploadBtn:          document.getElementById('upload-btn'),
    playBtn:            document.getElementById('play-btn'),
    pauseBtn:           document.getElementById('pause-btn'),
    thresholdSlider:    document.getElementById('threshold-slider'),
    thresholdValue:     document.getElementById('threshold-value'),
    confidenceSlider:   document.getElementById('confidence-slider'),
    confidenceValue:    document.getElementById('confidence-value'),

    // Incident log
    incidentTableBody:  document.getElementById('incident-table-body'),
    noIncidents:        document.getElementById('no-incidents'),
    incidentCount:      document.getElementById('incident-count'),

    // Bottom bar
    subjectCount:       document.getElementById('subject-count'),
    alertCount:         document.getElementById('alert-count'),
    fpsValue:           document.getElementById('fps-value'),
    gpuStatus:          document.getElementById('gpu-status'),
    connectionStatus:   document.getElementById('connection-status'),
};


/* ═══════════════════════════════════════════════════════════════════════════
   1. WEBSOCKET CONNECTION
   ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Establish WebSocket connection with auto-reconnect on close.
 * The server sends JSON messages with 'type' field: 'update' or 'incident'.
 */
function connectWebSocket() {
    // Determine the WS protocol (ws:// or wss://) based on page protocol
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    console.log(`[SENTINEL] Connecting to ${wsUrl}...`);
    updateConnectionStatus('reconnecting');

    try {
        ws = new WebSocket(wsUrl);
    } catch (err) {
        console.error('[SENTINEL] WebSocket creation failed:', err);
        scheduleReconnect();
        return;
    }

    ws.onopen = () => {
        console.log('[SENTINEL] WebSocket connected');
        reconnectAttempts = 0;
        updateConnectionStatus('connected');
    };

    ws.onmessage = (event) => {
        handleMessage(event);
    };

    ws.onclose = (event) => {
        console.warn(`[SENTINEL] WebSocket closed (code: ${event.code})`);
        updateConnectionStatus('disconnected');
        scheduleReconnect();
    };

    ws.onerror = (error) => {
        console.error('[SENTINEL] WebSocket error:', error);
        updateConnectionStatus('disconnected');
    };
}

/**
 * Schedule a reconnect attempt after a delay.
 */
function scheduleReconnect() {
    if (reconnectTimer) clearTimeout(reconnectTimer);
    reconnectAttempts++;
    // Exponential backoff capped at 30s
    const delay = Math.min(RECONNECT_DELAY * Math.pow(1.5, reconnectAttempts - 1), 30000);
    console.log(`[SENTINEL] Reconnecting in ${Math.round(delay / 1000)}s (attempt ${reconnectAttempts})...`);
    updateConnectionStatus('reconnecting');
    reconnectTimer = setTimeout(() => {
        connectWebSocket();
    }, delay);
}


/* ═══════════════════════════════════════════════════════════════════════════
   2. MESSAGE HANDLER
   ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Parse incoming WebSocket JSON messages and dispatch to appropriate handlers.
 */
function handleMessage(event) {
    let data;
    try {
        data = JSON.parse(event.data);
    } catch (err) {
        console.error('[SENTINEL] Failed to parse message:', err);
        return;
    }

    switch (data.type) {
        case 'update':
            handleUpdate(data);
            break;
        case 'incident':
            addIncident(data);
            break;
        default:
            console.warn('[SENTINEL] Unknown message type:', data.type);
    }
}

/**
 * Handle 'update' messages — update subjects, stats, and video state.
 */
function handleUpdate(data) {
    // Update subject cards
    if (data.subjects !== undefined) {
        updateSubjectCards(data.subjects);
    }

    // Update statistics
    updateStats(data);

    // Update HUD overlay info
    if (data.frame_number !== undefined) {
        DOM.hudFrame.textContent = `FRM: ${String(data.frame_number).padStart(6, '0')}`;
    }
    if (data.fps !== undefined) {
        DOM.hudFps.textContent = `FPS: ${data.fps.toFixed(1)}`;
    }
    if (data.timestamp) {
        DOM.hudTimestamp.textContent = data.timestamp;
    }

    // Handle video feed visibility
    if (data.is_playing !== undefined) {
        if (data.is_playing && !isPlaying) {
            showVideoFeed();
        } else if (!data.is_playing && isPlaying) {
            hideVideoFeed();
        }
        isPlaying = data.is_playing;
    }
}


/* ═══════════════════════════════════════════════════════════════════════════
   3. SUBJECT CARDS
   ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Dynamically create, update, and remove subject identity cards.
 * @param {Array} subjects — array of subject objects from the server
 */
function updateSubjectCards(subjects) {
    if (!subjects || subjects.length === 0) {
        // No subjects — show placeholder
        DOM.subjectCards.innerHTML = '';
        DOM.noSubjects.classList.remove('hidden');
        DOM.panelSubjectCount.textContent = '0';
        return;
    }

    DOM.noSubjects.classList.add('hidden');
    DOM.panelSubjectCount.textContent = subjects.length;

    // Track which subject IDs are currently in the update
    const currentIds = new Set(subjects.map(s => String(s.id)));

    // Remove cards for subjects no longer present
    const existingCards = DOM.subjectCards.querySelectorAll('.subject-card');
    existingCards.forEach(card => {
        if (!currentIds.has(card.dataset.id)) {
            card.style.opacity = '0';
            card.style.transform = 'translateY(-10px)';
            setTimeout(() => card.remove(), 200);
        }
    });

    // Create or update cards
    subjects.forEach(subject => {
        const subjectId = String(subject.id);
        let card = DOM.subjectCards.querySelector(`.subject-card[data-id="${subjectId}"]`);

        // Determine card status for styling
        const threatPercent = Math.round((subject.threat || 0) * 100);
        const statusClass = getStatusClass(subject.status, subject.threat);
        const threatColorClass = getThreatColorClass(subject.threat);

        if (card) {
            // ── UPDATE existing card ──
            updateExistingCard(card, subject, statusClass, threatPercent, threatColorClass);
        } else {
            // ── CREATE new card ──
            card = createSubjectCard(subject, subjectId, statusClass, threatPercent, threatColorClass);
            DOM.subjectCards.appendChild(card);
        }
    });
}

/**
 * Create a new subject identity card DOM element.
 */
function createSubjectCard(subject, subjectId, statusClass, threatPercent, threatColorClass) {
    const card = document.createElement('div');
    card.className = 'subject-card';
    card.dataset.id = subjectId;
    card.dataset.status = statusClass;

    const statusLabel = getStatusLabel(subject.status);
    const durationFormatted = subject.duration_formatted || formatDuration(subject.duration || 0);

    card.innerHTML = `
        <div class="card-header">
            <span class="card-id">SUBJECT_${subjectId.padStart(3, '0')}</span>
            <span class="card-status ${statusClass}">${statusLabel}</span>
        </div>
        <div class="card-body">
            <div class="card-field">
                <span class="field-label">TIME</span>
                <span class="field-value duration-value">${durationFormatted}</span>
            </div>
            <div class="card-field">
                <span class="field-label">POSE</span>
                <span class="field-value pose-value">${subject.pose || 'UNKNOWN'}</span>
            </div>
            <div class="card-field">
                <span class="field-label">THREAT</span>
                <div class="threat-bar">
                    <div class="threat-fill ${threatColorClass}" style="width: ${threatPercent}%"></div>
                </div>
                <span class="threat-value ${threatColorClass}">${threatPercent}%</span>
            </div>
        </div>
    `;

    return card;
}

/**
 * Update an existing subject card with new data (smooth transitions).
 */
function updateExistingCard(card, subject, statusClass, threatPercent, threatColorClass) {
    // Update data attribute for CSS styling
    card.dataset.status = statusClass;

    // Update status badge
    const statusEl = card.querySelector('.card-status');
    if (statusEl) {
        statusEl.className = `card-status ${statusClass}`;
        statusEl.textContent = getStatusLabel(subject.status);
    }

    // Update duration
    const durationEl = card.querySelector('.duration-value');
    if (durationEl) {
        durationEl.textContent = subject.duration_formatted || formatDuration(subject.duration || 0);
    }

    // Update pose
    const poseEl = card.querySelector('.pose-value');
    if (poseEl) {
        poseEl.textContent = subject.pose || 'UNKNOWN';
    }

    // Update threat bar (CSS transition handles animation)
    const threatFill = card.querySelector('.threat-fill');
    if (threatFill) {
        threatFill.style.width = `${threatPercent}%`;
        threatFill.className = `threat-fill ${threatColorClass}`;
    }

    const threatVal = card.querySelector('.threat-value');
    if (threatVal) {
        threatVal.textContent = `${threatPercent}%`;
        threatVal.className = `threat-value ${threatColorClass}`;
    }
}

/**
 * Get CSS status class based on subject status and threat level.
 */
function getStatusClass(status, threat) {
    if (threat >= 0.7) return 'high-threat';
    if (status && status.toUpperCase() === 'LOITERING') return 'loitering';
    return 'tracking';
}

/**
 * Get display label for subject status.
 */
function getStatusLabel(status) {
    if (!status) return 'TRACKING';
    const upper = status.toUpperCase();
    if (upper === 'LOITERING') return '⚠ LOITERING';
    return upper;
}


/* ═══════════════════════════════════════════════════════════════════════════
   4. STATS UPDATE
   ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Update bottom bar statistics from the server update message.
 */
function updateStats(data) {
    if (data.total_subjects !== undefined) {
        DOM.subjectCount.textContent = data.total_subjects;
    }

    if (data.active_alerts !== undefined) {
        DOM.alertCount.textContent = data.active_alerts;
        // Highlight alerts when > 0
        DOM.alertCount.classList.toggle('alert-value', data.active_alerts > 0);
    }

    if (data.fps !== undefined) {
        DOM.fpsValue.textContent = data.fps.toFixed(1);
    }
}


/* ═══════════════════════════════════════════════════════════════════════════
   5. INCIDENT LOG
   ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Prepend a new incident row to the incident log table.
 * Auto-scrolls and limits to MAX_INCIDENTS rows.
 */
function addIncident(incident) {
    // Hide the "no incidents" message
    DOM.noIncidents.classList.add('hidden');

    incidentCount++;
    DOM.incidentCount.textContent = `${incidentCount} EVENT${incidentCount !== 1 ? 'S' : ''}`;

    // Build the table row
    const row = document.createElement('tr');
    const threatPercent = Math.round((incident.threat || 0) * 100);
    const threatClass = getThreatCellClass(incident.threat);
    const isLoitering = incident.event && incident.event.toUpperCase().includes('LOITERING');

    if (isLoitering) row.classList.add('loitering-row');
    row.classList.add('new-row');

    row.innerHTML = `
        <td>${incident.timestamp || '--:--:--'}</td>
        <td>SUBJECT_${String(incident.subject_id || '???').padStart(3, '0')}</td>
        <td>${incident.event || 'UNKNOWN'}</td>
        <td>${incident.duration || '--:--'}</td>
        <td>${incident.pose || '--'}</td>
        <td class="${threatClass}">${threatPercent}%</td>
    `;

    // Prepend to table body (newest first)
    DOM.incidentTableBody.insertBefore(row, DOM.incidentTableBody.firstChild);

    // Remove the flash animation class after it completes
    setTimeout(() => row.classList.remove('new-row'), 1500);

    // Enforce max rows limit
    while (DOM.incidentTableBody.children.length > MAX_INCIDENTS) {
        DOM.incidentTableBody.removeChild(DOM.incidentTableBody.lastChild);
    }
}

/**
 * Get CSS class for threat level cell coloring.
 */
function getThreatCellClass(threat) {
    if (threat >= 0.7) return 'threat-high';
    if (threat >= 0.3) return 'threat-medium';
    return 'threat-low';
}


/* ═══════════════════════════════════════════════════════════════════════════
   6. REAL-TIME CLOCK
   ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Update the timestamp display with the current local time.
 * Runs every second via setInterval.
 */
function updateClock() {
    const now = new Date();
    const year  = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const day   = String(now.getDate()).padStart(2, '0');
    const hours = String(now.getHours()).padStart(2, '0');
    const mins  = String(now.getMinutes()).padStart(2, '0');
    const secs  = String(now.getSeconds()).padStart(2, '0');

    DOM.timestamp.textContent = `${year}-${month}-${day} ${hours}:${mins}:${secs}`;
}


/* ═══════════════════════════════════════════════════════════════════════════
   7. VIDEO UPLOAD
   ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Handle video file upload via FormData POST to /api/upload.
 * Shows loading state, hides overlay on success, shows error on failure.
 */
function uploadVideo(file) {
    if (!file) return;

    // Validate file type
    const validTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/x-msvideo'];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(mp4|avi|mov|mkv)$/i)) {
        showUploadError('INVALID FILE FORMAT — USE MP4, AVI, MOV, OR MKV');
        return;
    }

    // Show progress, hide error
    DOM.uploadError.classList.add('hidden');
    DOM.uploadProgress.classList.remove('hidden');
    DOM.uploadProgressFill.style.width = '0%';
    DOM.uploadProgressText.textContent = 'UPLOADING...';

    const formData = new FormData();
    formData.append('video', file);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/upload', true);

    // Track upload progress
    xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
            const percent = Math.round((e.loaded / e.total) * 100);
            DOM.uploadProgressFill.style.width = `${percent}%`;
            DOM.uploadProgressText.textContent = `UPLOADING... ${percent}%`;
        }
    };

    xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
            // Success — hide overlay, show video feed
            DOM.uploadProgressFill.style.width = '100%';
            DOM.uploadProgressText.textContent = 'PROCESSING...';

            setTimeout(() => {
                hideUploadOverlay();
                showVideoFeed();
                // Reconnect WebSocket to start receiving updates
                if (!ws || ws.readyState !== WebSocket.OPEN) {
                    connectWebSocket();
                }
            }, 500);
        } else {
            let errorMsg = 'UPLOAD FAILED';
            try {
                const resp = JSON.parse(xhr.responseText);
                errorMsg = resp.error || errorMsg;
            } catch (e) { /* ignore parse error */ }
            showUploadError(errorMsg.toUpperCase());
        }
    };

    xhr.onerror = () => {
        showUploadError('NETWORK ERROR — CHECK CONNECTION');
    };

    xhr.send(formData);
}

/**
 * Show upload error message.
 */
function showUploadError(message) {
    DOM.uploadProgress.classList.add('hidden');
    DOM.uploadError.classList.remove('hidden');
    DOM.uploadError.querySelector('.error-text').textContent = message;
}

/**
 * Hide the upload overlay with fade transition.
 */
function hideUploadOverlay() {
    DOM.uploadOverlay.classList.add('hidden');
}

/**
 * Show the upload overlay.
 */
function showUploadOverlay() {
    DOM.uploadOverlay.classList.remove('hidden');
    DOM.uploadProgress.classList.add('hidden');
    DOM.uploadError.classList.add('hidden');
    // Reset file input
    DOM.videoUpload.value = '';
}


/* ═══════════════════════════════════════════════════════════════════════════
   8. SETTINGS UPDATE
   ═══════════════════════════════════════════════════════════════════════════ */

/**
 * POST threshold and confidence values to /api/settings (debounced).
 */
function updateSettings() {
    // Clear existing debounce timer
    if (settingsDebounceTimer) clearTimeout(settingsDebounceTimer);

    settingsDebounceTimer = setTimeout(() => {
        const threshold = parseInt(DOM.thresholdSlider.value, 10);
        const confidence = parseInt(DOM.confidenceSlider.value, 10) / 100;

        fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                loitering_threshold: threshold,
                confidence_threshold: confidence
            })
        })
        .then(response => {
            if (!response.ok) {
                console.error('[SENTINEL] Settings update failed:', response.status);
            }
        })
        .catch(err => {
            console.error('[SENTINEL] Settings update error:', err);
        });
    }, SETTINGS_DEBOUNCE);
}


/* ═══════════════════════════════════════════════════════════════════════════
   9. DRAG & DROP
   ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Initialize drag-and-drop handlers for the upload area.
 */
function initDragDrop() {
    const dropZone = DOM.dropZone;

    // Prevent default drag behaviors on document
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        document.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
        });
    });

    // Visual feedback for drag over
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('drag-over');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('drag-over');
        });
    });

    // Handle dropped files
    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            uploadVideo(files[0]);
        }
    });

    // Click on drop zone triggers file input
    dropZone.addEventListener('click', (e) => {
        // Don't trigger if clicking the label/button directly (it has its own handler)
        if (e.target.closest('.upload-label')) return;
        DOM.videoUpload.click();
    });
}


/* ═══════════════════════════════════════════════════════════════════════════
   10. UTILITY FUNCTIONS
   ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Format seconds to MM:SS string.
 * @param {number} seconds — duration in seconds
 * @returns {string} formatted as MM:SS
 */
function formatDuration(seconds) {
    if (!seconds || seconds < 0) return '00:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

/**
 * Get CSS color class based on threat level.
 * 0–0.3: green, 0.3–0.7: orange, 0.7+: red
 */
function getThreatColorClass(level) {
    if (level >= 0.7) return 'red';
    if (level >= 0.3) return 'orange';
    return 'green';
}

/**
 * Get CSS hex color based on threat level.
 */
function getThreatColor(level) {
    if (level >= 0.7) return '#FF0000';
    if (level >= 0.3) return '#FFA500';
    return '#00FF41';
}


/* ═══════════════════════════════════════════════════════════════════════════
   11. CONNECTION STATUS
   ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Update the connection status indicator in the bottom bar.
 * @param {'connected'|'disconnected'|'reconnecting'} status
 */
function updateConnectionStatus(status) {
    const el = DOM.connectionStatus;
    if (!el) return;

    // Remove all status classes
    el.classList.remove('connected', 'disconnected', 'reconnecting');
    el.classList.add(status);

    const textEl = el.querySelector('.conn-text');
    if (textEl) {
        const labels = {
            connected:     'CONNECTED',
            disconnected:  'DISCONNECTED',
            reconnecting:  'RECONNECTING...'
        };
        textEl.textContent = labels[status] || status.toUpperCase();
    }
}


/* ═══════════════════════════════════════════════════════════════════════════
   12. VIDEO FEED VISIBILITY
   ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Show the MJPEG video feed and hide the "NO SIGNAL" placeholder.
 */
function showVideoFeed() {
    DOM.noSignal.classList.add('hidden');
    DOM.videoFeed.classList.remove('hidden');
    // Force reload the MJPEG stream
    DOM.videoFeed.src = '/video_feed?' + Date.now();
    isPlaying = true;
}

/**
 * Hide the video feed and show the "NO SIGNAL" placeholder.
 */
function hideVideoFeed() {
    DOM.videoFeed.classList.add('hidden');
    DOM.noSignal.classList.remove('hidden');
    DOM.videoFeed.src = '';
    isPlaying = false;
}


/* ═══════════════════════════════════════════════════════════════════════════
   13. PLAY / PAUSE CONTROLS
   ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Send play command to server.
 */
function playVideo() {
    fetch('/api/play', { method: 'POST' })
        .then(res => {
            if (res.ok) {
                showVideoFeed();
            }
        })
        .catch(err => console.error('[SENTINEL] Play error:', err));
}

/**
 * Send pause command to server.
 */
function pauseVideo() {
    fetch('/api/pause', { method: 'POST' })
        .then(res => {
            if (res.ok) {
                // Don't hide feed on pause — just stop updates
                console.log('[SENTINEL] Video paused');
            }
        })
        .catch(err => console.error('[SENTINEL] Pause error:', err));
}


/* ═══════════════════════════════════════════════════════════════════════════
   14. INITIALIZATION
   ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Initialize the SENTINEL dashboard on page load.
 */
function init() {
    console.log('[SENTINEL] Initializing SENTINEL v2.0 Dashboard...');

    // ── Start real-time clock ──
    updateClock();
    setInterval(updateClock, 1000);

    // ── Connect WebSocket ──
    connectWebSocket();

    // ── Setup slider event listeners with debounce ──
    DOM.thresholdSlider.addEventListener('input', () => {
        DOM.thresholdValue.textContent = `${DOM.thresholdSlider.value}s`;
        updateSettings();
    });

    DOM.confidenceSlider.addEventListener('input', () => {
        const val = (parseInt(DOM.confidenceSlider.value, 10) / 100).toFixed(2);
        DOM.confidenceValue.textContent = val;
        updateSettings();
    });

    // ── Setup upload button ──
    DOM.uploadBtn.addEventListener('click', () => {
        showUploadOverlay();
    });

    // ── File input change handler ──
    DOM.videoUpload.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            uploadVideo(e.target.files[0]);
        }
    });

    // ── Play / Pause buttons ──
    DOM.playBtn.addEventListener('click', playVideo);
    DOM.pauseBtn.addEventListener('click', pauseVideo);

    // ── Initialize drag & drop ──
    initDragDrop();

    // ── Keyboard shortcuts ──
    document.addEventListener('keydown', (e) => {
        // Space — toggle subject panel visibility
        if (e.code === 'Space' && e.target === document.body) {
            e.preventDefault();
            DOM.subjectPanel.classList.toggle('hidden');
        }

        // Escape — close upload overlay if visible
        if (e.code === 'Escape' && !DOM.uploadOverlay.classList.contains('hidden')) {
            hideUploadOverlay();
        }
    });

    // ── Handle video feed error (e.g., stream not available yet) ──
    DOM.videoFeed.addEventListener('error', () => {
        if (isPlaying) {
            // Stream broke — show no signal
            hideVideoFeed();
        }
    });

    console.log('[SENTINEL] Dashboard initialized successfully');
}


// ─── LAUNCH ────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', init);
