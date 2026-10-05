/**
 * DeepTrust AI - Live Browser Camera Controller
 * Manages media streams, camera facingMode switching, snapshots, and error handling.
 */
class CameraController {
    constructor(videoElementId, previewElementId, onCaptureCallback) {
        this.videoEl = document.getElementById(videoElementId);
        this.previewEl = document.getElementById(previewElementId);
        this.onCapture = onCaptureCallback;

        this.currentStream = null;
        this.facingMode = 'environment'; // Default to rear camera on mobile
        this.capturedBlob = null;
        this.capturedDataUrl = null;
        this.isActive = false;
    }

    /**
     * Checks if browser supports camera API.
     */
    static isSupported() {
        return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    }

    /**
     * Starts camera video stream.
     */
    async startCamera() {
        if (!CameraController.isSupported()) {
            throw new Error("Camera API is not supported in this browser. Please use HTTPS or an updated browser.");
        }

        this.stopCamera(); // Stop any previous stream

        const constraints = {
            audio: false,
            video: {
                facingMode: { ideal: this.facingMode },
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }
        };

        try {
            this.currentStream = await navigator.mediaDevices.getUserMedia(constraints);
            if (this.videoEl) {
                this.videoEl.srcObject = this.currentStream;
                await this.videoEl.play();
                this.isActive = true;
            }
            return true;
        } catch (err) {
            console.error("Camera access error:", err);
            if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
                throw new Error("Camera permission was denied. Please allow camera access in your browser settings.");
            } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
                throw new Error("No camera device was detected on your system.");
            } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
                throw new Error("Camera is currently in use by another application.");
            } else {
                throw new Error("Failed to initialize camera: " + (err.message || err.name));
            }
        }
    }

    /**
     * Switches between front and rear cameras (useful for mobile).
     */
    async switchFacingMode() {
        this.facingMode = (this.facingMode === 'environment') ? 'user' : 'environment';
        return await this.startCamera();
    }

    /**
     * Captures current frame to high-resolution canvas.
     */
    captureFrame() {
        if (!this.isActive || !this.videoEl) {
            throw new Error("Camera is not actively streaming.");
        }

        const video = this.videoEl;
        const width = video.videoWidth || 640;
        const height = video.videoHeight || 480;

        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;

        const ctx = canvas.getContext('2d');
        // Mirror if using front camera for natural preview
        if (this.facingMode === 'user') {
            ctx.translate(width, 0);
            ctx.scale(-1, 1);
        }
        ctx.drawImage(video, 0, 0, width, height);

        this.capturedDataUrl = canvas.toDataURL('image/jpeg', 0.95);

        return new Promise((resolve) => {
            canvas.toBlob((blob) => {
                this.capturedBlob = blob;
                if (this.onCapture) {
                    this.onCapture(blob, this.capturedDataUrl);
                }
                resolve({ blob, dataUrl: this.capturedDataUrl });
            }, 'image/jpeg', 0.95);
        });
    }

    /**
     * Stops all active camera tracks.
     */
    stopCamera() {
        if (this.currentStream) {
            this.currentStream.getTracks().forEach(track => {
                track.stop();
            });
            this.currentStream = null;
        }
        if (this.videoEl) {
            this.videoEl.srcObject = null;
        }
        this.isActive = false;
    }
}

window.CameraController = CameraController;
