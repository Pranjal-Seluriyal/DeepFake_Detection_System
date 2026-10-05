/**
 * DeepTrust AI - Main Application Controller
 * Handles image uploading, live camera capture, AI analysis, result rendering, and history.
 */

document.addEventListener('DOMContentLoaded', () => {
    // ----------------------------------------------------
    // STATE
    // ----------------------------------------------------
    let currentMediaType = 'image'; // 'image' or 'audio'
    let currentInputMode = 'upload'; // 'upload' or 'camera'
    let selectedImageFile = null;
    let cameraCapturedBlob = null;
    let cameraCapturedDataUrl = null;
    let selectedAudioFile = null;
    let isAnalyzing = false;

    // ----------------------------------------------------
    // DOM ELEMENTS
    // ----------------------------------------------------
    // Mode toggles
    const tabImage = document.getElementById('tabImage');
    const tabAudio = document.getElementById('tabAudio');
    const imageSection = document.getElementById('imageSection');
    const audioSection = document.getElementById('audioSection');

    // Input mode switch (Upload vs Camera)
    const btnModeUpload = document.getElementById('btnModeUpload');
    const btnModeCamera = document.getElementById('btnModeCamera');
    const uploadContainer = document.getElementById('uploadContainer');
    const cameraContainer = document.getElementById('cameraContainer');

    // Image Upload
    const fileInput = document.getElementById('fileInput');
    const dropzone = document.getElementById('dropzone');
    const imagePreviewContainer = document.getElementById('imagePreviewContainer');
    const previewImg = document.getElementById('previewImg');
    const previewFileName = document.getElementById('previewFileName');
    const previewFileSize = document.getElementById('previewFileSize');
    const btnRemoveImage = document.getElementById('btnRemoveImage');
    const btnAnalyzeImage = document.getElementById('btnAnalyzeImage');

    // Camera
    const cameraVideo = document.getElementById('cameraVideo');
    const btnSwitchCamera = document.getElementById('btnSwitchCamera');
    const btnCapturePhoto = document.getElementById('btnCapturePhoto');
    const cameraReviewContainer = document.getElementById('cameraReviewContainer');
    const cameraReviewImg = document.getElementById('cameraReviewImg');
    const btnRetakeCamera = document.getElementById('btnRetakeCamera');
    const btnAnalyzeCamera = document.getElementById('btnAnalyzeCamera');
    const cameraActiveView = document.getElementById('cameraActiveView');
    const cameraErrorMsg = document.getElementById('cameraErrorMsg');

    // Audio
    const audioFileInput = document.getElementById('audioFileInput');
    const audioDropzone = document.getElementById('audioDropzone');
    const audioPreviewContainer = document.getElementById('audioPreviewContainer');
    const audioPlayer = document.getElementById('audioPlayer');
    const audioFileName = document.getElementById('audioFileName');
    const btnRemoveAudio = document.getElementById('btnRemoveAudio');
    const btnAnalyzeAudio = document.getElementById('btnAnalyzeAudio');

    // Analysis Status & Result Cards
    const analyzingState = document.getElementById('analyzingState');
    const analyzingStatusText = document.getElementById('analyzingStatusText');
    const analyzingSubtext = document.getElementById('analyzingSubtext');

    const resultCard = document.getElementById('resultCard');
    const resultBadge = document.getElementById('resultBadge');
    const resultConfidence = document.getElementById('resultConfidence');
    const resultRawScore = document.getElementById('resultRawScore');
    const resultThumb = document.getElementById('resultThumb');
    const resultFilename = document.getElementById('resultFilename');
    const resultStorageBadge = document.getElementById('resultStorageBadge');
    const resultMetricsContainer = document.getElementById('resultMetricsContainer');
    const btnAnalyzeAnother = document.getElementById('btnAnalyzeAnother');

    // History section
    const historyGrid = document.getElementById('historyGrid');
    const historyEmpty = document.getElementById('historyEmpty');
    const btnRefreshHistory = document.getElementById('btnRefreshHistory');

    // System Status
    const systemStatusBadge = document.getElementById('systemStatusBadge');

    // ----------------------------------------------------
    // INITIALIZE CAMERA CONTROLLER
    // ----------------------------------------------------
    let cameraController = null;
    if (window.CameraController) {
        cameraController = new CameraController('cameraVideo', null, (blob, dataUrl) => {
            cameraCapturedBlob = blob;
            cameraCapturedDataUrl = dataUrl;
        });
    }

    // ----------------------------------------------------
    // SYSTEM HEALTH CHECK
    // ----------------------------------------------------
    async function checkSystemHealth() {
        try {
            const res = await fetch('/api/health');
            const data = await res.json();
            if (systemStatusBadge) {
                if (data.status === 'healthy') {
                    systemStatusBadge.innerHTML = `
                        <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                        <span>Neural Engine Active &bull; ${data.storage === 'cloudinary' ? 'Cloud Storage' : 'Ready'}</span>
                    `;
                    systemStatusBadge.className = "inline-flex items-center gap-2 border border-emerald-500/20 bg-emerald-500/10 px-3.5 py-1.5 rounded-full text-xs font-semibold text-emerald-300";
                } else {
                    systemStatusBadge.innerHTML = `
                        <span class="w-2 h-2 rounded-full bg-amber-400"></span>
                        <span>Neural Engine Standby</span>
                    `;
                }
            }
        } catch (e) {
            console.warn("Health check unreachable:", e);
        }
    }
    checkSystemHealth();

    // ----------------------------------------------------
    // MEDIA TYPE SWITCH (Image vs Audio)
    // ----------------------------------------------------
    if (tabImage && tabAudio) {
        tabImage.addEventListener('click', () => {
            currentMediaType = 'image';
            tabImage.className = "px-6 py-2.5 rounded-xl font-bold text-sm bg-blue-600 text-white shadow-lg shadow-blue-600/30 transition";
            tabAudio.className = "px-6 py-2.5 rounded-xl font-semibold text-sm text-slate-400 hover:text-white transition";
            imageSection.classList.remove('hidden');
            audioSection.classList.add('hidden');
            resetResults();
        });

        tabAudio.addEventListener('click', () => {
            currentMediaType = 'audio';
            tabAudio.className = "px-6 py-2.5 rounded-xl font-bold text-sm bg-purple-600 text-white shadow-lg shadow-purple-600/30 transition";
            tabImage.className = "px-6 py-2.5 rounded-xl font-semibold text-sm text-slate-400 hover:text-white transition";
            audioSection.classList.remove('hidden');
            imageSection.classList.add('hidden');
            if (cameraController) cameraController.stopCamera();
            resetResults();
        });
    }

    // ----------------------------------------------------
    // INPUT MODE SWITCH (Upload vs Camera)
    // ----------------------------------------------------
    if (btnModeUpload && btnModeCamera) {
        btnModeUpload.addEventListener('click', () => {
            currentInputMode = 'upload';
            btnModeUpload.className = "flex-1 py-3 rounded-xl font-bold text-xs uppercase tracking-wider bg-white/10 text-white border border-white/20 transition";
            btnModeCamera.className = "flex-1 py-3 rounded-xl font-bold text-xs uppercase tracking-wider text-slate-400 hover:text-white transition";
            uploadContainer.classList.remove('hidden');
            cameraContainer.classList.add('hidden');
            if (cameraController) cameraController.stopCamera();
        });

        btnModeCamera.addEventListener('click', async () => {
            currentInputMode = 'camera';
            btnModeCamera.className = "flex-1 py-3 rounded-xl font-bold text-xs uppercase tracking-wider bg-white/10 text-white border border-white/20 transition";
            btnModeUpload.className = "flex-1 py-3 rounded-xl font-bold text-xs uppercase tracking-wider text-slate-400 hover:text-white transition";
            cameraContainer.classList.remove('hidden');
            uploadContainer.classList.add('hidden');
            startCameraPreview();
        });
    }

    // ----------------------------------------------------
    // IMAGE UPLOAD LOGIC
    // ----------------------------------------------------
    if (dropzone && fileInput) {
        dropzone.addEventListener('click', () => fileInput.click());

        fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                handleSelectedImage(e.target.files[0]);
            }
        });

        // Drag & Drop
        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropzone.classList.add('drag-active');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropzone.classList.remove('drag-active');
            });
        });

        dropzone.addEventListener('drop', (e) => {
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                handleSelectedImage(e.dataTransfer.files[0]);
            }
        });
    }

    function handleSelectedImage(file) {
        const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/jpg'];
        if (!validTypes.includes(file.type)) {
            showToast("Invalid format. Please upload JPG, PNG, or WebP.", "error");
            return;
        }

        if (file.size > 32 * 1024 * 1024) {
            showToast("File size exceeds 32MB limit.", "error");
            return;
        }

        selectedImageFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            previewFileName.textContent = file.name;
            previewFileSize.textContent = (file.size / (1024 * 1024)).toFixed(2) + " MB";

            dropzone.classList.add('hidden');
            imagePreviewContainer.classList.remove('hidden');
            resetResults();
        };
        reader.readAsDataURL(file);
    }

    if (btnRemoveImage) {
        btnRemoveImage.addEventListener('click', () => {
            selectedImageFile = null;
            fileInput.value = '';
            imagePreviewContainer.classList.add('hidden');
            dropzone.classList.remove('hidden');
            resetResults();
        });
    }

    // ----------------------------------------------------
    // CAMERA LOGIC
    // ----------------------------------------------------
    async function startCameraPreview() {
        if (!cameraController) return;
        cameraErrorMsg.classList.add('hidden');
        cameraActiveView.classList.remove('hidden');
        cameraReviewContainer.classList.add('hidden');

        try {
            await cameraController.startCamera();
        } catch (err) {
            cameraActiveView.classList.add('hidden');
            cameraErrorMsg.classList.remove('hidden');
            cameraErrorMsg.textContent = err.message || "Failed to access camera.";
            showToast(err.message, "error");
        }
    }

    if (btnSwitchCamera) {
        btnSwitchCamera.addEventListener('click', async () => {
            if (!cameraController) return;
            try {
                await cameraController.switchFacingMode();
            } catch (err) {
                showToast("Could not switch camera: " + err.message, "error");
            }
        });
    }

    if (btnCapturePhoto) {
        btnCapturePhoto.addEventListener('click', async () => {
            if (!cameraController) return;
            try {
                const { blob, dataUrl } = await cameraController.captureFrame();
                cameraCapturedBlob = blob;
                cameraCapturedDataUrl = dataUrl;

                cameraReviewImg.src = dataUrl;
                cameraActiveView.classList.add('hidden');
                cameraReviewContainer.classList.remove('hidden');
                cameraController.stopCamera();
                resetResults();
            } catch (err) {
                showToast("Capture failed: " + err.message, "error");
            }
        });
    }

    if (btnRetakeCamera) {
        btnRetakeCamera.addEventListener('click', () => {
            cameraCapturedBlob = null;
            cameraCapturedDataUrl = null;
            startCameraPreview();
            resetResults();
        });
    }

    // ----------------------------------------------------
    // AUDIO UPLOAD LOGIC
    // ----------------------------------------------------
    if (audioDropzone && audioFileInput) {
        audioDropzone.addEventListener('click', () => audioFileInput.click());

        audioFileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                handleSelectedAudio(e.target.files[0]);
            }
        });

        audioDropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            audioDropzone.classList.add('drag-active');
        });

        audioDropzone.addEventListener('dragleave', () => {
            audioDropzone.classList.remove('drag-active');
        });

        audioDropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            audioDropzone.classList.remove('drag-active');
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                handleSelectedAudio(e.dataTransfer.files[0]);
            }
        });
    }

    function handleSelectedAudio(file) {
        selectedAudioFile = file;
        audioFileName.textContent = file.name;
        audioPlayer.src = URL.createObjectURL(file);

        audioDropzone.classList.add('hidden');
        audioPreviewContainer.classList.remove('hidden');
        resetResults();
    }

    if (btnRemoveAudio) {
        btnRemoveAudio.addEventListener('click', () => {
            selectedAudioFile = null;
            audioFileInput.value = '';
            audioPlayer.src = '';
            audioPreviewContainer.classList.add('hidden');
            audioDropzone.classList.remove('hidden');
            resetResults();
        });
    }

    // ----------------------------------------------------
    // ANALYSIS SUBMISSION
    // ----------------------------------------------------
    if (btnAnalyzeImage) {
        btnAnalyzeImage.addEventListener('click', () => {
            if (!selectedImageFile) {
                showToast("Please choose an image to analyze.", "warning");
                return;
            }
            performAnalysis('image', selectedImageFile);
        });
    }

    if (btnAnalyzeCamera) {
        btnAnalyzeCamera.addEventListener('click', () => {
            if (!cameraCapturedBlob) {
                showToast("Please capture a photo first.", "warning");
                return;
            }
            const file = new File([cameraCapturedBlob], "camera_capture.jpg", { type: "image/jpeg" });
            performAnalysis('image', file);
        });
    }

    if (btnAnalyzeAudio) {
        btnAnalyzeAudio.addEventListener('click', () => {
            if (!selectedAudioFile) {
                showToast("Please select an audio file to analyze.", "warning");
                return;
            }
            performAnalysis('audio', selectedAudioFile);
        });
    }

    if (btnAnalyzeAnother) {
        btnAnalyzeAnother.addEventListener('click', () => {
            resetResults();
            if (currentMediaType === 'image' && currentInputMode === 'camera') {
                startCameraPreview();
            }
        });
    }

    // ----------------------------------------------------
    // PERFORM ANALYSIS (Multi-Stage Animated Pipeline)
    // ----------------------------------------------------
    async function performAnalysis(type, fileOrBlob) {
        if (isAnalyzing) return;
        isAnalyzing = true;

        // Show loading state
        analyzingState.classList.remove('hidden');
        resultCard.classList.add('hidden');

        // Single generic loading state (no technical stages or rotating intervals)
        if (analyzingStatusText) analyzingStatusText.textContent = "Analyzing your media...";
        if (analyzingSubtext) analyzingSubtext.textContent = "Please wait while we examine it.";

        const formData = new FormData();
        formData.append('file', fileOrBlob);

        const endpoint = type === 'image' ? '/api/predict/image' : '/api/predict/audio';

        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                throw new Error(data.error || "Analysis failed. Please try again.");
            }

            // Display Results
            renderResults(type, data);
            loadPredictionHistory(); // Refresh history table
            showToast("Analysis complete!", "success");

        } catch (err) {
            console.error("Analysis Error:", err);
            showToast(err.message || "An error occurred during scan.", "error");
        } finally {
            analyzingState.classList.add('hidden');
            isAnalyzing = false;
        }
    }

    // ----------------------------------------------------
    // RENDER RESULTS
    // ----------------------------------------------------
    function renderResults(type, data) {
        resultCard.classList.remove('hidden');

        const isFake = (data.status === 'Deepfake');
        const confVal = data.confidence_value || parseFloat(data.confidence) || 0;

        // Status Badge
        if (isFake) {
            resultBadge.textContent = "SYNTHETIC / DEEPFAKE";
            resultBadge.className = "px-4 py-1.5 rounded-full font-black text-xs uppercase tracking-widest bg-red-500/20 text-red-400 border border-red-500/30";
            resultCard.style.borderColor = "rgba(239, 68, 68, 0.4)";
        } else {
            resultBadge.textContent = "AUTHENTIC / REAL";
            resultBadge.className = "px-4 py-1.5 rounded-full font-black text-xs uppercase tracking-widest bg-emerald-500/20 text-emerald-400 border border-emerald-500/30";
            resultCard.style.borderColor = "rgba(16, 185, 129, 0.4)";
        }

        // Confidence
        resultConfidence.textContent = `${confVal.toFixed(1)}%`;
        resultConfidence.className = isFake ? "text-4xl md:text-5xl font-black text-red-400" : "text-4xl md:text-5xl font-black text-emerald-400";

        if (resultRawScore) {
            resultRawScore.textContent = data.raw_score !== undefined ? data.raw_score : "--";
        }

        // Thumbnail
        if (resultThumb) {
            if (data.image_url) {
                resultThumb.src = data.image_url;
                resultThumb.classList.remove('hidden');
            } else {
                resultThumb.classList.add('hidden');
            }
        }

        // Filename & Storage Badge
        if (resultFilename) {
            resultFilename.textContent = data.prediction_id ? `Scan #${data.prediction_id}` : "Forensic Scan";
        }

        if (resultStorageBadge) {
            resultStorageBadge.textContent = data.storage_provider === 'cloudinary' ? 'Cloud Verified' : 'Persisted';
        }

        // Forensic Metrics Breakdown
        if (resultMetricsContainer && data.metrics) {
            resultMetricsContainer.innerHTML = `
                <div class="bg-white/5 rounded-2xl p-3 border border-white/5 text-center">
                    <p class="text-[10px] uppercase font-bold text-slate-400 mb-1">Frequency Artifacts</p>
                    <p class="text-sm font-black text-cyan-400">${data.metrics.frequency_dispersion || 0}%</p>
                </div>
                <div class="bg-white/5 rounded-2xl p-3 border border-white/5 text-center">
                    <p class="text-[10px] uppercase font-bold text-slate-400 mb-1">Edge Boundary Flow</p>
                    <p class="text-sm font-black text-indigo-400">${data.metrics.edge_density || 0}%</p>
                </div>
                <div class="bg-white/5 rounded-2xl p-3 border border-white/5 text-center">
                    <p class="text-[10px] uppercase font-bold text-slate-400 mb-1">Sensor Acquisition</p>
                    <p class="text-sm font-black text-purple-400">${data.metrics.acquisition_noise || 0}%</p>
                </div>
            `;
        }

        // Smooth scroll to result
        resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function resetResults() {
        resultCard.classList.add('hidden');
        analyzingState.classList.add('hidden');
    }

    // ----------------------------------------------------
    // PREDICTION HISTORY LOADER
    // ----------------------------------------------------
    async function loadPredictionHistory() {
        if (!historyGrid) return;

        try {
            const res = await fetch('/api/predictions?limit=12');
            const data = await res.json();

            if (!data.success || !data.predictions || data.predictions.length === 0) {
                historyGrid.innerHTML = '';
                if (historyEmpty) historyEmpty.classList.remove('hidden');
                return;
            }

            if (historyEmpty) historyEmpty.classList.add('hidden');

            historyGrid.innerHTML = data.predictions.map(item => {
                const isFake = (item.status === 'Deepfake');
                const badgeColor = isFake 
                    ? 'bg-red-500/20 text-red-400 border-red-500/30' 
                    : 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
                
                const thumbHtml = item.prediction_type === 'image'
                    ? `<img src="${item.media_url}" alt="Scan thumb" class="w-full h-36 object-cover rounded-xl mb-3 border border-white/5" onerror="this.src='/static/images/real_fake_diamond.png'">`
                    : `<div class="w-full h-36 rounded-xl bg-purple-950/40 border border-purple-500/20 flex items-center justify-center text-3xl mb-3">🎵</div>`;

                return `
                    <div class="glass-card rounded-2xl p-4 flex flex-col justify-between hover:border-white/20 transition">
                        <div>
                            ${thumbHtml}
                            <div class="flex items-center justify-between gap-2 mb-2">
                                <span class="px-2.5 py-0.5 rounded-full text-[10px] font-black uppercase tracking-wider border ${badgeColor}">
                                    ${item.status}
                                </span>
                                <span class="text-xs font-black text-slate-300">
                                    ${item.confidence}
                                </span>
                            </div>
                            <p class="text-xs text-slate-400 font-medium truncate mb-1" title="${item.original_filename}">
                                ${item.original_filename}
                            </p>
                        </div>
                        <p class="text-[10px] text-slate-500 mt-2 border-t border-white/5 pt-2">
                            ${item.created_at_display || 'Recent'}
                        </p>
                    </div>
                `;
            }).join('');

        } catch (err) {
            console.error("Failed to load history:", err);
        }
    }

    if (btnRefreshHistory) {
        btnRefreshHistory.addEventListener('click', loadPredictionHistory);
    }
    loadPredictionHistory();

    // ----------------------------------------------------
    // TOAST NOTIFICATIONS
    // ----------------------------------------------------
    function showToast(message, type = 'info') {
        let container = document.getElementById('toastContainer');
        if (!container) {
            container = document.createElement('div');
            container.id = 'toastContainer';
            container.className = 'fixed bottom-6 right-6 z-50 flex flex-col gap-2 max-w-sm';
            document.body.appendChild(container);
        }

        const colors = {
            error: 'bg-red-950/90 border-red-500/50 text-red-200',
            success: 'bg-emerald-950/90 border-emerald-500/50 text-emerald-200',
            warning: 'bg-amber-950/90 border-amber-500/50 text-amber-200',
            info: 'bg-slate-900/90 border-blue-500/50 text-blue-200'
        };

        const toast = document.createElement('div');
        toast.className = `p-4 rounded-2xl border backdrop-blur-xl shadow-2xl text-xs font-semibold flex items-center justify-between gap-3 transition-all duration-300 transform translate-y-2 opacity-0 ${colors[type] || colors.info}`;
        toast.innerHTML = `
            <span>${message}</span>
            <button class="opacity-70 hover:opacity-100 font-bold ml-2">&times;</button>
        `;

        toast.querySelector('button').onclick = () => {
            toast.remove();
        };

        container.appendChild(toast);

        // Animate in
        requestAnimationFrame(() => {
            toast.classList.remove('translate-y-2', 'opacity-0');
        });

        setTimeout(() => {
            toast.classList.add('opacity-0', 'translate-y-2');
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }
});
