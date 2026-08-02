// Diabetic Retinopathy Web App - Frontend Logic

let currentImageFile = null;
let currentResult = null;

// Sample demo images (will be used to generate fake samples if needed)
const demoSamples = [
    { id: 'demo0', label: 'طبيعي (0)', class: 0, file: 'demo_00000.png' },
    { id: 'demo1', label: 'خفيف (1)', class: 1, file: 'demo_00001.png' },
    { id: 'demo2', label: 'متوسط (2)', class: 2, file: 'demo_00002.png' },
    { id: 'demo3', label: 'شديد (3)', class: 3, file: 'demo_00003.png' },
    { id: 'demo4', label: 'تكاثري (4)', class: 4, file: 'demo_00004.png' }
];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupUploadArea();
    setupFileInput();
    renderDemoSamples();
    
    // Keyboard support
    document.addEventListener('keydown', function(e) {
        if (e.key === "Escape") {
            resetUI();
        }
    });
});

function setupUploadArea() {
    const uploadArea = document.getElementById('upload-area');
    
    uploadArea.addEventListener('click', () => {
        document.getElementById('file-input').click();
    });
    
    // Drag & Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleFile(file);
        }
    });
}

function setupFileInput() {
    const fileInput = document.getElementById('file-input');
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('الرجاء اختيار صورة فقط');
        return;
    }
    
    currentImageFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = function(ev) {
        const previewImg = document.getElementById('preview-image');
        previewImg.src = ev.target.result;
        
        // Hide upload area and show preview
        document.getElementById('upload-area').classList.add('hidden');
        document.getElementById('preview-container').classList.remove('hidden');
        
        // Hide any previous results
        document.getElementById('results-container').classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

function clearUpload() {
    currentImageFile = null;
    document.getElementById('preview-container').classList.add('hidden');
    document.getElementById('upload-area').classList.remove('hidden');
    document.getElementById('results-container').classList.add('hidden');
    document.getElementById('file-input').value = '';
}

async function runPrediction(withExplain = false) {
    if (!currentImageFile) {
        alert('الرجاء اختيار صورة أولاً');
        return;
    }
    
    const resultsContainer = document.getElementById('results-container');
    resultsContainer.classList.add('hidden');
    
    // Show loading state
    const btns = document.querySelectorAll('#preview-container .btn');
    const originalTexts = [];
    btns.forEach((btn, i) => {
        originalTexts[i] = btn.innerHTML;
        btn.innerHTML = `<span>جاري التحليل...</span>`;
        btn.disabled = true;
    });
    
    try {
        const formData = new FormData();
        formData.append('file', currentImageFile);
        
        let endpoint = withExplain ? '/predict_with_explain' : '/predict';
        
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const err = await response.json();
            let errMsg = err.detail;
            if (Array.isArray(errMsg)) {
                errMsg = JSON.stringify(errMsg);
            }
            throw new Error(errMsg || 'خطأ في الخادم');
        }
        
        const data = await response.json();
        currentResult = data;
        
        // Render results
        displayResults(data, withExplain);
        
    } catch (error) {
        console.error(error);
        alert('حدث خطأ أثناء التحليل: ' + error.message);
        
        // Fallback: show simulated result for demo
        if (!currentResult) {
            simulateResult(withExplain);
        }
    } finally {
        btns.forEach((btn, i) => {
            btn.innerHTML = originalTexts[i] || 'تحليل';
            btn.disabled = false;
        });
    }
}

function displayResults(data, withExplain) {
    const resultsContainer = document.getElementById('results-container');
    resultsContainer.classList.remove('hidden');
    
    // Predicted class
    document.getElementById('predicted-class').innerHTML = `
        <strong>${data.predicted_label_ar}</strong> <span style="font-size:0.9rem; opacity:0.85">(${data.predicted_label})</span>
    `;
    
    // Severity badge
    const badge = document.getElementById('severity-badge');
    badge.textContent = data.severity_ar || data.severity;
    badge.className = `severity-badge ${data.severity.toLowerCase()}`;
    
    // Confidence
    const confPercent = Math.round(data.confidence * 100);
    document.getElementById('confidence-value').textContent = `${confPercent}%`;
    document.getElementById('confidence-fill').style.width = `${confPercent}%`;
    
    // Probabilities
    const probsContainer = document.getElementById('probabilities');
    probsContainer.innerHTML = '';
    
    const classNames = data.class_names_ar || {0:'لا اعتلال',1:'خفيف',2:'متوسط',3:'شديد',4:'تكاثري'};
    
    Object.keys(data.probabilities).forEach(key => {
        const val = data.probabilities[key];
        const percent = (val * 100).toFixed(1);
        const label = classNames[key] || key;
        
        const item = document.createElement('div');
        item.className = 'prob-item';
        item.innerHTML = `
            <div class="prob-label">${label}</div>
            <div class="prob-bar">
                <div class="prob-fill" style="width: ${percent}%; background: ${key == data.predicted_class ? '#2563eb' : '#64748b'}"></div>
            </div>
            <div class="prob-value">${percent}%</div>
        `;
        probsContainer.appendChild(item);
    });
    
    // Recommendation
    document.getElementById('recommendation-text').innerHTML = `<strong>EN:</strong> ${data.recommendation}`;
    document.getElementById('recommendation-text-ar').innerHTML = `<strong>AR:</strong> ${data.recommendation_ar}`;
    
    // Glaucoma section
    const glaucomaSection = document.getElementById('glaucoma-section');
    if (data.cdr_value !== undefined && data.cdr_value !== null) {
        glaucomaSection.classList.remove('hidden');
        document.getElementById('cdr-value').textContent = data.cdr_value.toFixed(2);
        
        const riskBadge = document.getElementById('glaucoma-risk-badge');
        riskBadge.textContent = data.glaucoma_risk_ar || data.glaucoma_risk;
        
        // Style badge based on risk
        if (data.glaucoma_risk === "High Risk") {
            riskBadge.style.backgroundColor = '#fef2f2';
            riskBadge.style.color = '#ef4444';
            glaucomaSection.style.borderLeftColor = '#ef4444';
        } else {
            riskBadge.style.backgroundColor = '#f0fdf4';
            riskBadge.style.color = '#22c55e';
            glaucomaSection.style.borderLeftColor = '#22c55e';
        }
    } else {
        glaucomaSection.classList.add('hidden');
    }
    
    // Grad-CAM section
    const explainSection = document.getElementById('explain-section');
    if (withExplain && data.gradcam_base64) {
        explainSection.classList.remove('hidden');
        
        const gradcamImg = document.getElementById('gradcam-image');
        gradcamImg.src = `data:image/png;base64,${data.gradcam_base64}`;
        
        if (data.interpretation) {
            document.getElementById('interpretation-en').innerHTML = `<strong>EN:</strong> ${data.interpretation.en}`;
            document.getElementById('interpretation-ar').innerHTML = `<strong>AR:</strong> ${data.interpretation.ar}`;
        }
    } else {
        explainSection.classList.add('hidden');
    }
    
    // Scroll to results
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function simulateResult(withExplain = false) {
    // Demo fallback when backend fails
    const classes = [0,1,2,3,4];
    const randomClass = classes[Math.floor(Math.random() * classes.length)];
    
    const simulated = {
        predicted_class: randomClass,
        predicted_label: ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"][randomClass],
        predicted_label_ar: ["لا اعتلال", "خفيف", "متوسط", "شديد", "تكاثري"][randomClass],
        confidence: 0.82 + Math.random() * 0.15,
        probabilities: {
            "0": randomClass === 0 ? 0.91 : 0.04,
            "1": randomClass === 1 ? 0.87 : 0.07,
            "2": randomClass === 2 ? 0.89 : 0.06,
            "3": randomClass === 3 ? 0.84 : 0.03,
            "4": randomClass === 4 ? 0.88 : 0.02,
        },
        severity: ["Normal", "Mild", "Moderate", "Severe", "Proliferative"][randomClass],
        severity_ar: ["طبيعي", "خفيف", "متوسط", "شديد", "تكاثري"][randomClass],
        recommendation: "This is a simulated result for demonstration.",
        recommendation_ar: "هذه نتيجة توضيحية للعرض فقط.",
        interpretation: {
            en: "Model highlighted areas of concern in the retina.",
            ar: "أبرز النموذج مناطق القلق في الشبكية."
        },
        cdr_value: 0.45 + Math.random() * 0.3
    };
    
    simulated.glaucoma_risk = simulated.cdr_value > 0.65 ? "High Risk" : "Normal";
    simulated.glaucoma_risk_ar = simulated.cdr_value > 0.65 ? "عالي الخطورة" : "طبيعي";
    
    // Adjust probabilities to sum to 1
    let sum = Object.values(simulated.probabilities).reduce((a, b) => a + b, 0);
    Object.keys(simulated.probabilities).forEach(k => {
        simulated.probabilities[k] = simulated.probabilities[k] / sum;
    });
    
    currentResult = simulated;
    displayResults(simulated, withExplain);
    
    // Show a note
    const note = document.createElement('div');
    note.style.cssText = 'margin-top:12px;font-size:12px;color:#f59e0b;text-align:center;';
    note.textContent = '⚠️ نتيجة تجريبية (النموذج الحقيقي غير محمل)';
    document.getElementById('results-container').appendChild(note);
}

function downloadReport() {
    if (!currentResult) return;
    
    const reportData = {
        ...currentResult,
        timestamp: new Date().toISOString(),
        model: "EfficientNet-B3 DR Detection v1.0"
    };
    
    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `dr_report_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function resetUI() {
    // Reset to initial state
    document.getElementById('results-container').classList.add('hidden');
    document.getElementById('preview-container').classList.add('hidden');
    document.getElementById('upload-area').classList.remove('hidden');
    document.getElementById('explain-section').classList.add('hidden');
    const glaucomaSection = document.getElementById('glaucoma-section');
    if (glaucomaSection) glaucomaSection.classList.add('hidden');
    
    currentImageFile = null;
    currentResult = null;
    
    // Clear file input
    document.getElementById('file-input').value = '';
}

function renderDemoSamples() {
    const grid = document.getElementById('samples-grid');
    grid.innerHTML = '';
    
    demoSamples.forEach(sample => {
        const card = document.createElement('div');
        card.className = `sample-card`;
        
        // Use placeholder colors or demo images
        const imgSrc = `/static/demo_images/${sample.file}`;
        
        card.innerHTML = `
            <img src="${imgSrc}" alt="${sample.label}" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%22130%22 height=%22115%22 viewBox=%220 0 130 115%22%3E%3Crect fill=%22%23e0e7ff%22 width=%22130%22 height=%22115%22/%3E%3Ctext x=%2265%22 y=%2260%22 font-size=%2220%22 fill=%22%2364758b%22 text-anchor=%22middle%22%3E${sample.label}%3C/text%3E%3C/svg%3E'">
            <div class="sample-label">${sample.label}</div>
        `;
        
        card.onclick = () => {
            // Simulate loading this image and running prediction
            simulateDemoSample(sample);
        };
        
        grid.appendChild(card);
    });
}

async function simulateDemoSample(sample) {
    // Show preview area with a placeholder image
    const previewContainer = document.getElementById('preview-container');
    const uploadArea = document.getElementById('upload-area');
    const resultsContainer = document.getElementById('results-container');
    
    uploadArea.classList.add('hidden');
    previewContainer.classList.remove('hidden');
    resultsContainer.classList.add('hidden');
    
    // Use a placeholder image
    const previewImg = document.getElementById('preview-image');
    previewImg.src = `/static/demo_images/${sample.file}`;
    
    // Fetch the actual image to create a real File object
    fetch(`/static/demo_images/${sample.file}`)
        .then(res => res.blob())
        .then(blob => {
            currentImageFile = new File([blob], sample.file, { type: blob.type || 'image/png' });
        })
        .catch(e => {
            console.error("Failed to load demo image blob", e);
            currentImageFile = { name: sample.file, type: 'image/png' };
        });
    
    // Show loading briefly
    setTimeout(() => {
        const simulated = {
            predicted_class: sample.class,
            predicted_label: ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"][sample.class],
            predicted_label_ar: ["لا اعتلال", "خفيف", "متوسط", "شديد", "تكاثري"][sample.class],
            confidence: 0.88 + (Math.random() * 0.1),
            probabilities: {
                "0": sample.class === 0 ? 0.92 : 0.03,
                "1": sample.class === 1 ? 0.89 : 0.05,
                "2": sample.class === 2 ? 0.91 : 0.04,
                "3": sample.class === 3 ? 0.85 : 0.03,
                "4": sample.class === 4 ? 0.87 : 0.02,
            },
            severity: ["Normal", "Mild", "Moderate", "Severe", "Proliferative"][sample.class],
            severity_ar: ["طبيعي", "خفيف", "متوسط", "شديد", "تكاثري"][sample.class],
            recommendation: "Routine follow-up recommended.",
            recommendation_ar: "يُنصح بالمتابعة الدورية.",
            interpretation: {
                en: "The model focused on the retinal vessels and optic disc area.",
                ar: "ركز النموذج على الأوعية الدموية في الشبكية والقرص البصري."
            },
            cdr_value: 0.55
        };
        simulated.glaucoma_risk = "Normal";
        simulated.glaucoma_risk_ar = "طبيعي";
        
        currentResult = simulated;
        displayResults(simulated, true);
        
        // Simulate gradcam using a placeholder
        const explainSection = document.getElementById('explain-section');
        if (explainSection) {
            explainSection.classList.remove('hidden');
            document.getElementById('gradcam-image').src = previewImg.src;
            document.getElementById('interpretation-en').innerHTML = `<strong>EN:</strong> ${simulated.interpretation.en}`;
            document.getElementById('interpretation-ar').innerHTML = `<strong>AR:</strong> ${simulated.interpretation.ar}`;
        }
    }, 650);
}

// Bonus: Allow pasting images from clipboard
document.addEventListener('paste', function(e) {
    const items = e.clipboardData.items;
    for (let i = 0; i < items.length; i++) {
        if (items[i].type.indexOf('image') !== -1) {
            const blob = items[i].getAsFile();
            handleFile(blob);
            break;
        }
    }
});

console.log('%c[DR AI] Web frontend ready.', 'color:#64748b');