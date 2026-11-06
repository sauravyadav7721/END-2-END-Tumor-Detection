/**
 * Tumor Detection AI - Frontend Controller
 * Handles image upload, API calls, and dynamic UI updates.
 */

document.addEventListener('DOMContentLoaded', () => {
    const $ = (id) => document.getElementById(id);

    const elements = {
        uploadSection: $('upload-section'),
        fileInput: $('file-input'),
        fileNameDisplay: $('file-name-display'),

        previewContainer: $('preview-container'),
        previewImage: $('preview-image'),
        predictBtn: $('predict-btn'),
        clearBtn: $('clear-btn'),
        btnText: $('btn-text'),
        btnLoader: $('btn-loader'),

        emptyState: $('empty-state'),
        resultsLoaderContainer: $('results-loader-container'),
        predictionResults: $('prediction-results'),
        verdict: $('verdict'),
        diagnosisValue: $('diagnosis-value'),
        diagnosisSub: $('diagnosis-sub'),
        confidenceValue: $('confidence-value'),

        gradcamContainer: $('gradcam-container'),
        gradcamImage: $('gradcam-image'),
        originalImage: $('original-image'),
        probBars: $('prob-bars'),

        toast: $('toast')
    };

    const DEFAULT_PROMPT = 'Drop an image here or <span>browse</span>';

    // Display metadata for each model class (names must match backend CLASSES)
    const CLASS_INFO = {
        'No Tumor':  { tone: 'good', sub: 'Brain scan · no tumor detected' },
        'Tumor':     { tone: 'bad',  sub: 'Brain scan · tumor detected' },
        'Normal':    { tone: 'good', sub: 'Breast scan · no abnormality detected' },
        'Benign':    { tone: 'warn', sub: 'Breast scan · benign growth' },
        'Malignant': { tone: 'bad',  sub: 'Breast scan · malignant growth' }
    };

    let currentFile = null;
    let camUrl = null;
    let toastTimer = null;

    // --- Helpers ---

    const showToast = (message) => {
        elements.toast.textContent = message;
        elements.toast.classList.remove('hidden');
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => elements.toast.classList.add('hidden'), 5000);
    };

    const revokeCam = () => {
        if (camUrl) {
            URL.revokeObjectURL(camUrl);
            camUrl = null;
        }
    };

    const resetResults = () => {
        elements.emptyState.classList.remove('hidden');
        elements.resultsLoaderContainer.classList.add('hidden');
        elements.predictionResults.classList.add('hidden');
        elements.gradcamContainer.classList.add('hidden');

        revokeCam();
        elements.gradcamImage.removeAttribute('src');
        elements.diagnosisValue.textContent = '-';
        elements.diagnosisSub.textContent = '';
        elements.confidenceValue.textContent = '-';
        elements.probBars.innerHTML = '';
    };

    const clearAll = () => {
        currentFile = null;
        elements.fileInput.value = '';
        elements.fileNameDisplay.innerHTML = DEFAULT_PROMPT;
        elements.previewImage.removeAttribute('src');
        elements.originalImage.removeAttribute('src');
        elements.previewContainer.classList.add('hidden');
        resetResults();
    };

    const handleFileSelect = (file) => {
        if (!file.type.startsWith('image/')) {
            showToast('Please choose a valid image file (JPG, PNG, etc.).');
            return;
        }

        currentFile = file;
        elements.fileNameDisplay.textContent = file.name;

        const reader = new FileReader();
        reader.onload = (e) => {
            elements.previewImage.src = e.target.result;
            elements.originalImage.src = e.target.result;
            elements.previewContainer.classList.remove('hidden');
        };
        reader.readAsDataURL(file);

        resetResults();
    };

    const toggleLoading = (isLoading) => {
        elements.predictBtn.disabled = isLoading;
        elements.clearBtn.disabled = isLoading;
        elements.btnText.textContent = isLoading ? 'Analyzing…' : 'Analyze scan';
        elements.btnLoader.classList.toggle('hidden', !isLoading);
        if (isLoading) {
            elements.emptyState.classList.add('hidden');
            elements.resultsLoaderContainer.classList.remove('hidden');
            elements.predictionResults.classList.add('hidden');
        } else {
            elements.resultsLoaderContainer.classList.add('hidden');
        }
    };

    const renderProbs = (probs, predicted) => {
        elements.probBars.innerHTML = '';
        const fills = [];

        Object.entries(probs)
            .sort((a, b) => b[1] - a[1])
            .forEach(([name, val]) => {
                const pct = (val * 100).toFixed(1);

                const row = document.createElement('div');
                row.className = 'prob-row';

                const label = document.createElement('span');
                label.className = 'class-name';
                label.textContent = name;

                const track = document.createElement('div');
                track.className = 'bar-track';
                const fill = document.createElement('div');
                fill.className = 'bar-fill' + (name === predicted ? ' accent' : '');
                track.appendChild(fill);
                fills.push([fill, pct]);

                const value = document.createElement('span');
                value.className = 'pct';
                value.textContent = `${pct}%`;

                row.append(label, track, value);
                elements.probBars.appendChild(row);
            });

        // Animate bars after they are in the DOM
        requestAnimationFrame(() => fills.forEach(([fill, pct]) => { fill.style.width = `${pct}%`; }));
    };

    // --- Events ---

    elements.uploadSection.addEventListener('click', () => elements.fileInput.click());
    elements.uploadSection.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            elements.fileInput.click();
        }
    });

    elements.fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFileSelect(e.target.files[0]);
    });

    ['dragover', 'dragleave', 'drop'].forEach((evt) => {
        elements.uploadSection.addEventListener(evt, (e) => {
            e.preventDefault();
            elements.uploadSection.classList.toggle('dragover', evt === 'dragover');
            if (evt === 'drop' && e.dataTransfer.files.length > 0) {
                handleFileSelect(e.dataTransfer.files[0]);
            }
        });
    });

    elements.clearBtn.addEventListener('click', clearAll);

    elements.predictBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        toggleLoading(true);

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const [predRes, camRes] = await Promise.all([
                fetch('/predict', { method: 'POST', body: formData }),
                fetch('/gradcam', { method: 'POST', body: formData })
            ]);

            if (!predRes.ok) throw new Error('Prediction request failed.');

            const data = await predRes.json();
            const info = CLASS_INFO[data.prediction] || { tone: 'neutral', sub: '' };

            elements.predictionResults.classList.remove('hidden');
            elements.verdict.dataset.tone = info.tone;
            elements.diagnosisValue.textContent = data.prediction;
            elements.diagnosisSub.textContent = info.sub;
            elements.confidenceValue.textContent = data.confidence;

            renderProbs(data.all_probs, data.prediction);

            // Grad-CAM is optional: still show the prediction if it fails
            if (camRes.ok) {
                revokeCam();
                camUrl = URL.createObjectURL(await camRes.blob());
                elements.gradcamImage.src = camUrl;
                elements.gradcamContainer.classList.remove('hidden');
            } else {
                showToast('Prediction succeeded, but the Grad-CAM heatmap could not be generated.');
            }
        } catch (err) {
            console.error('Inference Error:', err);
            showToast('Analysis failed. Please try again or check the server logs.');
            resetResults();
        } finally {
            toggleLoading(false);
        }
    });
});
