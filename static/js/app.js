/**
 * SignFlow AI - Frontend Application Controller
 * Handles SPA Navigation, Camera Streaming, Web Speech API (TTS),
 * Real-time Predictions, and the Text-to-Sign Learning Engine.
 */

document.addEventListener("DOMContentLoaded", () => {
    // -------------------------------------------------------------
    // 1. Single Page Application (SPA) Tab Switching
    // -------------------------------------------------------------
    const navLinks = document.querySelectorAll(".nav-link-custom");
    const pages = document.querySelectorAll(".app-page");

    function showPage(pageId) {
        pages.forEach(p => p.classList.remove("active-page"));
        navLinks.forEach(l => l.classList.remove("active"));

        const targetPage = document.getElementById(pageId);
        if (targetPage) {
            targetPage.classList.add("active-page");
        }

        const activeLink = document.querySelector(`[data-target="${pageId}"]`);
        if (activeLink) {
            activeLink.classList.add("active");
        }

        window.scrollTo({ top: 0, behavior: "smooth" });
    }

    navLinks.forEach(link => {
        link.addEventListener("click", (e) => {
            e.preventDefault();
            const target = link.getAttribute("data-target");
            showPage(target);
        });
    });

    // Home Page CTA Buttons
    const heroStartBtn = document.getElementById("heroStartBtn");
    if (heroStartBtn) {
        heroStartBtn.addEventListener("click", () => {
            showPage("page-live");
            startWebcam();
        });
    }

    const heroLearnBtn = document.getElementById("heroLearnBtn");
    if (heroLearnBtn) {
        heroLearnBtn.addEventListener("click", () => {
            showPage("page-learn");
        });
    }

    // -------------------------------------------------------------
    // 2. Real-Time Webcam Detection
    // -------------------------------------------------------------
    const video = document.getElementById("webcamVideo");
    const canvas = document.getElementById("outputCanvas");
    const ctx = canvas.getContext("2d");
    const placeholder = document.getElementById("cameraPlaceholder");

    const startCamBtn = document.getElementById("startCamBtn");
    const stopCamBtn = document.getElementById("stopCamBtn");
    const cameraStatusBadge = document.getElementById("cameraStatusBadge");
    const handNotice = document.getElementById("handNotice");

    const predSignBadge = document.getElementById("predSignBadge");
    const predConfBadge = document.getElementById("predConfBadge");
    const latencyTag = document.getElementById("latencyTag");
    const top3Container = document.getElementById("top3Container");
    const sentenceOutput = document.getElementById("sentenceOutput");

    let mediaStream = null;
    let isDetecting = false;
    let isRequestPending = false;

    async function startWebcam() {
        if (mediaStream) return;

        try {
            cameraStatusBadge.className = "status-pill status-waiting";
            cameraStatusBadge.innerHTML = '<span class="pulse-indicator"></span> Starting Camera...';

            mediaStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
                audio: false
            });

            video.srcObject = mediaStream;
            await video.play();

            placeholder.style.display = "none";
            canvas.style.display = "block";

            cameraStatusBadge.className = "status-pill status-active";
            cameraStatusBadge.innerHTML = '<span class="pulse-indicator"></span> Camera Live';
            startCamBtn.disabled = true;
            stopCamBtn.disabled = false;

            isDetecting = true;
            requestAnimationFrame(detectionLoop);
        } catch (err) {
            console.error("Camera access error:", err);
            cameraStatusBadge.className = "status-pill status-idle";
            cameraStatusBadge.innerHTML = '<i class="bi bi-exclamation-triangle"></i> Permission Denied';
            alert("Could not access your webcam. Please ensure camera permissions are enabled in your browser settings.");
        }
    }

    function stopWebcam() {
        isDetecting = false;
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            mediaStream = null;
        }
        video.srcObject = null;

        canvas.style.display = "none";
        placeholder.style.display = "flex";

        cameraStatusBadge.className = "status-pill status-idle";
        cameraStatusBadge.innerHTML = '<span class="pulse-indicator"></span> Camera Idle';
        startCamBtn.disabled = false;
        stopCamBtn.disabled = true;

        predSignBadge.textContent = "--";
        predConfBadge.textContent = "0%";
        handNotice.textContent = "Camera stopped. Click 'Start Camera' to begin.";
    }

    startCamBtn.addEventListener("click", startWebcam);
    stopCamBtn.addEventListener("click", stopWebcam);

    async function detectionLoop() {
        if (!isDetecting) return;

        if (!isRequestPending && video.readyState === video.HAVE_ENOUGH_DATA) {
            isRequestPending = true;

            // Grab frame from video
            const offscreen = document.createElement("canvas");
            offscreen.width = 640;
            offscreen.height = 480;
            const offCtx = offscreen.getContext("2d");
            offCtx.drawImage(video, 0, 0, 640, 480);

            const base64Data = offscreen.toDataURL("image/jpeg", 0.65);

            try {
                const response = await fetch("/api/predict_frame", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ image: base64Data, flip: true })
                });

                if (response.ok) {
                    const data = await response.json();
                    renderDetectionResults(data);
                }
            } catch (error) {
                console.warn("Prediction request error:", error);
            } finally {
                isRequestPending = false;
            }
        }

        // Run smoothly at ~25 FPS
        setTimeout(() => {
            requestAnimationFrame(detectionLoop);
        }, 40);
    }

    function renderDetectionResults(data) {
        // Draw annotated frame
        if (data.annotated_image) {
            const img = new Image();
            img.onload = () => {
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            };
            img.src = data.annotated_image;
        }

        // Hand Presence & Notice
        if (data.hand_detected) {
            handNotice.className = "small text-success fw-semibold";
            handNotice.innerHTML = '<i class="bi bi-check-circle-fill me-1"></i> Hand detected & tracked';
        } else {
            handNotice.className = "small text-muted";
            handNotice.innerHTML = '<i class="bi bi-info-circle me-1"></i> No hand detected. Position your hand in frame.';
        }

        // Prediction Badge
        predSignBadge.textContent = (data.prediction && data.prediction !== "None") ? data.prediction : "--";
        const confPercent = Math.round((data.confidence || 0) * 100);
        predConfBadge.textContent = `${confPercent}%`;
        latencyTag.textContent = `Latency: ${data.latency_ms || 0} ms`;

        // Top 3 Confidence Bars
        if (data.top3 && data.top3.length > 0) {
            top3Container.innerHTML = data.top3.map(item => `
                <div class="mb-2">
                    <div class="d-flex justify-content-between small fw-semibold mb-1">
                        <span>${item.label}</span>
                        <span class="text-muted">${Math.round(item.prob * 100)}%</span>
                    </div>
                    <div class="progress" style="height: 6px; background: rgba(0,0,0,0.06); border-radius: 4px;">
                        <div class="progress-bar" style="width: ${Math.round(item.prob * 100)}%; background: linear-gradient(90deg, #8b5cf6, #38bdf8); border-radius: 4px;"></div>
                    </div>
                </div>
            `).join("");
        }

        // Translated Sentence
        if (data.sentence && data.sentence.trim().length > 0) {
            sentenceOutput.textContent = data.sentence;
            sentenceOutput.classList.remove("text-muted");
        } else {
            sentenceOutput.textContent = "(Hold up sign gestures to formulate text...)";
            sentenceOutput.classList.add("text-muted");
        }
    }

    // Sentence Action Controls
    async function executeSentenceAction(action) {
        try {
            const res = await fetch("/api/sentence/action", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ action })
            });
            const data = await res.json();
            if (data.sentence && data.sentence.trim().length > 0) {
                sentenceOutput.textContent = data.sentence;
                sentenceOutput.classList.remove("text-muted");
            } else {
                sentenceOutput.textContent = "(Hold up sign gestures to formulate text...)";
                sentenceOutput.classList.add("text-muted");
            }
        } catch (e) {
            console.error("Sentence action error:", e);
        }
    }

    document.getElementById("clearSentenceBtn").addEventListener("click", () => executeSentenceAction("clear"));
    document.getElementById("spaceSentenceBtn").addEventListener("click", () => executeSentenceAction("space"));
    document.getElementById("backspaceSentenceBtn").addEventListener("click", () => executeSentenceAction("backspace"));

    // -------------------------------------------------------------
    // 3. Web Speech API (Text-to-Speech Engine)
    // -------------------------------------------------------------
    const synth = window.speechSynthesis;
    let voices = [];
    const voiceSelect = document.getElementById("voiceSelect");
    const rateRange = document.getElementById("rateRange");
    const pitchRange = document.getElementById("pitchRange");
    const rateVal = document.getElementById("rateVal");
    const pitchVal = document.getElementById("pitchVal");
    const ttsText = document.getElementById("ttsText");
    const visualizer = document.getElementById("audioVisualizer");

    function loadVoices() {
        if (!synth) return;
        voices = synth.getVoices();
        if (voiceSelect) {
            voiceSelect.innerHTML = voices.map((v, i) => `
                <option value="${i}">${v.name} (${v.lang})</option>
            `).join("");
        }
    }

    if (synth) {
        loadVoices();
        if (speechSynthesis.onvoiceschanged !== undefined) {
            speechSynthesis.onvoiceschanged = loadVoices;
        }
    }

    if (rateRange && rateVal) {
        rateRange.addEventListener("input", () => rateVal.textContent = `${rateRange.value}x`);
    }
    if (pitchRange && pitchVal) {
        pitchRange.addEventListener("input", () => pitchVal.textContent = `${pitchRange.value}`);
    }

    function speakText(textToSpeak) {
        if (!synth) {
            alert("Web Speech API is not supported in this browser.");
            return;
        }
        if (!textToSpeak || textToSpeak.includes("Hold up sign gestures")) {
            return;
        }

        synth.cancel(); // Stop any pending speech

        const utterance = new SpeechSynthesisUtterance(textToSpeak);
        const selectedVoiceIdx = voiceSelect ? voiceSelect.value : 0;
        if (voices[selectedVoiceIdx]) {
            utterance.voice = voices[selectedVoiceIdx];
        }

        utterance.rate = rateRange ? parseFloat(rateRange.value) : 1.0;
        utterance.pitch = pitchRange ? parseFloat(pitchRange.value) : 1.0;

        utterance.onstart = () => {
            if (visualizer) visualizer.classList.add("speaking");
        };

        utterance.onend = () => {
            if (visualizer) visualizer.classList.remove("speaking");
        };

        utterance.onerror = () => {
            if (visualizer) visualizer.classList.remove("speaking");
        };

        synth.speak(utterance);
    }

    // Live Detection Page Speech Controls
    document.getElementById("speakSentenceBtn").addEventListener("click", () => {
        speakText(sentenceOutput.textContent);
    });

    document.getElementById("pauseSentenceBtn").addEventListener("click", () => {
        if (synth.speaking && !synth.paused) {
            synth.pause();
            if (visualizer) visualizer.classList.remove("speaking");
        } else if (synth.paused) {
            synth.resume();
            if (visualizer) visualizer.classList.add("speaking");
        }
    });

    document.getElementById("stopSentenceBtn").addEventListener("click", () => {
        synth.cancel();
        if (visualizer) visualizer.classList.remove("speaking");
    });

    // Text-to-Speech Lab Controls
    const ttsSpeakBtn = document.getElementById("ttsSpeakBtn");
    if (ttsSpeakBtn) {
        ttsSpeakBtn.addEventListener("click", () => {
            speakText(ttsText.value);
        });
    }

    const ttsPauseBtn = document.getElementById("ttsPauseBtn");
    if (ttsPauseBtn) {
        ttsPauseBtn.addEventListener("click", () => {
            if (synth.speaking && !synth.paused) synth.pause();
            else if (synth.paused) synth.resume();
        });
    }

    const ttsStopBtn = document.getElementById("ttsStopBtn");
    if (ttsStopBtn) {
        ttsStopBtn.addEventListener("click", () => synth.cancel());
    }

    // Preset phrase buttons in TTS Lab
    document.querySelectorAll(".quick-phrase-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            const phrase = btn.getAttribute("data-phrase");
            if (ttsText) {
                ttsText.value = phrase;
                speakText(phrase);
            }
        });
    });

    // -------------------------------------------------------------
    // 4. Learn Sign Language & Text-to-Sign Engine
    // -------------------------------------------------------------
    const signGallery = document.getElementById("signGallery");
    const searchSignInput = document.getElementById("searchSignInput");
    const categoryFilters = document.querySelectorAll(".sign-filter-pill");

    function renderSignCards(filterCategory = "all", searchQuery = "") {
        if (!signGallery || typeof SIGN_DICTIONARY === "undefined") return;

        let filtered = SIGN_DICTIONARY;

        if (filterCategory !== "all") {
            filtered = filtered.filter(s => s.category === filterCategory);
        }

        if (searchQuery.trim().length > 0) {
            const q = searchQuery.toLowerCase();
            filtered = filtered.filter(s =>
                s.char.toLowerCase().includes(q) ||
                s.name.toLowerCase().includes(q) ||
                s.description.toLowerCase().includes(q)
            );
        }

        if (filtered.length === 0) {
            signGallery.innerHTML = `
                <div class="col-12 text-center py-5 text-muted">
                    <i class="bi bi-search fs-2 mb-2"></i>
                    <p>No sign language gestures found matching your search.</p>
                </div>
            `;
            return;
        }

        signGallery.innerHTML = filtered.map(sign => `
            <div class="col-6 col-md-4 col-lg-3 col-xl-2 mb-3">
                <div class="sign-card h-100" onclick="openSignDetails('${sign.char}')">
                    <div class="sign-illustration-container mb-2">
                        <img src="/static/images/signs/${sign.char}.png" alt="Sign ${sign.char}" class="img-fluid rounded-3 shadow-sm border" style="max-height: 110px; width: auto; object-fit: contain;" onerror="this.onerror=null; this.src='/static/images/signs/A.png';">
                    </div>
                    <div class="sign-card-char mb-1">${sign.char.replace('_', ' ')}</div>
                    <div class="small fw-semibold text-muted text-truncate">${sign.name}</div>
                    <span class="badge bg-light text-secondary border mt-2" style="font-size: 0.7rem;">${sign.category.toUpperCase()}</span>
                </div>
            </div>
        `).join("");
    }

    if (signGallery) {
        renderSignCards();
    }

    // Category filter pills
    categoryFilters.forEach(pill => {
        pill.addEventListener("click", () => {
            categoryFilters.forEach(p => p.classList.remove("active"));
            pill.classList.add("active");
            const cat = pill.getAttribute("data-category");
            const query = searchSignInput ? searchSignInput.value : "";
            renderSignCards(cat, query);
        });
    });

    if (searchSignInput) {
        searchSignInput.addEventListener("input", () => {
            const activeFilter = document.querySelector(".sign-filter-pill.active");
            const cat = activeFilter ? activeFilter.getAttribute("data-category") : "all";
            renderSignCards(cat, searchSignInput.value);
        });
    }

    // Sign Details Modal helper
    window.openSignDetails = function (char) {
        const sign = SIGN_DICTIONARY.find(s => s.char === char);
        if (!sign) return;

        const modalTitle = document.getElementById("signModalTitle");
        const modalSvg = document.getElementById("signModalSvg");
        const modalDesc = document.getElementById("signModalDesc");
        const modalTips = document.getElementById("signModalTips");

        if (modalTitle) modalTitle.textContent = `${sign.name} (${sign.char.replace('_', ' ')})`;
        if (modalSvg) {
            modalSvg.innerHTML = `
                <img src="/static/images/signs/${sign.char}.png" alt="Sign ${sign.char}" class="img-fluid rounded-4 shadow border" style="max-height: 280px; width: auto; object-fit: contain;">
            `;
        }
        if (modalDesc) modalDesc.textContent = sign.description;
        if (modalTips) modalTips.textContent = sign.tips;

        const modalElem = document.getElementById("signDetailsModal");
        if (modalElem) {
            const modal = new bootstrap.Modal(modalElem);
            modal.show();
        }
    };

    // --- Text-to-Sign Interactive Converter ---
    const textToSignInput = document.getElementById("textToSignInput");
    const signStrip = document.getElementById("signStrip");
    const playSignSeqBtn = document.getElementById("playSignSeqBtn");

    function updateTextToSign(text) {
        if (!signStrip) return;
        const clean = text.toUpperCase().trim();

        if (clean.length === 0) {
            signStrip.innerHTML = `<span class="text-muted small">Type above to see live sign language translation...</span>`;
            return;
        }

        const itemsHtml = [];
        for (let i = 0; i < clean.length; i++) {
            const char = clean[i];
            if (char === " ") {
                itemsHtml.push(`
                    <div class="sign-strip-item" style="border-style: dashed; background: transparent;">
                        <div class="text-muted small fw-bold" style="height: 60px; display: flex; align-items: center;">[SPACE]</div>
                        <div class="text-muted" style="font-size: 0.75rem;">Space</div>
                    </div>
                `);
            } else {
                const match = SIGN_DICTIONARY.find(s => s.char === char);
                if (match) {
                    itemsHtml.push(`
                        <div class="sign-strip-item" id="strip-item-${i}">
                            <div style="width: 60px; height: 60px; display: flex; align-items: center; justify-content: center;">
                                <img src="/static/images/signs/${match.char}.png" alt="Sign ${match.char}" class="img-fluid rounded border shadow-sm" style="max-height: 58px; max-width: 58px; object-fit: contain;">
                            </div>
                            <div class="fw-bold mt-1" style="color: var(--lavender-600);">${match.char}</div>
                        </div>
                    `);
                } else {
                    itemsHtml.push(`
                        <div class="sign-strip-item">
                            <div class="text-secondary" style="height: 50px; display: flex; align-items: center; font-size: 1.4rem;">?</div>
                            <div class="small text-muted">${char}</div>
                        </div>
                    `);
                }
            }
        }

        signStrip.innerHTML = itemsHtml.join("");
    }

    if (textToSignInput) {
        textToSignInput.addEventListener("input", (e) => {
            updateTextToSign(e.target.value);
        });
        // Initial example
        updateTextToSign(textToSignInput.value || "HELLO 123");
    }

    if (playSignSeqBtn) {
        playSignSeqBtn.addEventListener("click", () => {
            const items = document.querySelectorAll(".sign-strip-item");
            if (items.length === 0) return;

            items.forEach((item, idx) => {
                setTimeout(() => {
                    items.forEach(el => el.style.borderColor = "var(--lavender-200)");
                    item.style.borderColor = "var(--sky-500)";
                    item.style.transform = "scale(1.1)";
                    item.scrollIntoView({ behavior: "smooth", inline: "center", block: "nearest" });
                    setTimeout(() => item.style.transform = "scale(1)", 400);
                }, idx * 600);
            });
        });
    }
});

