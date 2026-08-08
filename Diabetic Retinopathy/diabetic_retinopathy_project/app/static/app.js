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
        
        // Auto-run prediction with explanation
        runPrediction(true);
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
        // Handle Mask Image
        const maskContainer = document.getElementById('cdr-mask-container');
        if (data.cdr_mask_base64) {
            maskContainer.classList.remove('hidden');
            document.getElementById('cdr-mask-image').src = `data:image/png;base64,${data.cdr_mask_base64}`;
        } else {
            maskContainer.classList.add('hidden');
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
        cdr_value: 0.45 + Math.random() * 0.3,
        cdr_mask_base64: null // Cannot easily simulate a dummy mask dynamically
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

function getBase64Image(imgElement) {
    if (!imgElement || !imgElement.src) return '';
    if (imgElement.src.startsWith('data:')) return imgElement.src;
    
    try {
        const canvas = document.createElement('canvas');
        canvas.width = imgElement.naturalWidth || imgElement.width || 256;
        canvas.height = imgElement.naturalHeight || imgElement.height || 256;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(imgElement, 0, 0);
        return canvas.toDataURL('image/png');
    } catch (e) {
        console.error('Error converting image to base64:', e);
        return imgElement.src;
    }
}

function downloadReport() {
    // ============================================================
    // NUCLEAR OPTION: Abandon html2canvas entirely.
    // Use browser's native rendering via window.print() in a
    // new window. This ALWAYS works — Arabic, images, everything.
    // ============================================================

    // 1. Collect data
    const previewImg = document.getElementById('preview-image');
    const gradcamImg = document.getElementById('gradcam-image');
    const unetImg = document.getElementById('cdr-mask-image');

    const img1 = getBase64Image(previewImg);
    const img2 = getBase64Image(gradcamImg);
    const img3 = getBase64Image(unetImg);

    const drResult = document.getElementById('predicted-class') 
        ? document.getElementById('predicted-class').innerText.trim() : 'N/A';
    const drConfidence = document.getElementById('confidence-value') 
        ? document.getElementById('confidence-value').innerText.trim() : 'N/A';
    const vcdrEl = document.getElementById('cdr-value') || document.getElementById('vcdr-value');
    const vcdrValue = vcdrEl ? vcdrEl.innerText.trim() : 'N/A';

    const now = new Date();
    const dateStr = now.toLocaleDateString('ar-EG');
    const timeStr = now.toLocaleTimeString('ar-EG');

    // 2. Build a COMPLETE standalone HTML page
    const fullHTML = `<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>تقرير فحص شبكية العين</title>
    <style>
        @media print {
            body { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
            .no-print { display: none !important; }
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Arial, sans-serif;
            color: #333;
            background: #fff;
            direction: rtl;
            padding: 30px;
            max-width: 800px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            border-bottom: 3px solid #0056b3;
            padding-bottom: 20px;
            margin-bottom: 25px;
        }
        .header h1 { color: #0056b3; font-size: 22px; margin-bottom: 8px; }
        .header p { color: #666; font-size: 13px; }
        .images-table { width: 100%; border-collapse: collapse; margin-bottom: 30px; }
        .images-table td { width: 33%; text-align: center; vertical-align: top; padding: 8px; }
        .images-table img { width: 200px; height: 200px; object-fit: cover; border-radius: 8px; border: 1px solid #ddd; }
        .img-title { font-size: 14px; font-weight: bold; margin-bottom: 5px; color: #333; }
        .img-subtitle { font-size: 11px; color: #777; margin-bottom: 8px; }
        .result-box {
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .result-box.dr { background-color: #f8f9fa; border-right: 5px solid #d9534f; }
        .result-box.gl { background-color: #eef1f5; border-right: 5px solid #0275d8; }
        .result-box h2 { font-size: 17px; margin-bottom: 12px; }
        .result-box.dr h2 { color: #d9534f; }
        .result-box.gl h2 { color: #0275d8; }
        .data-table { width: 100%; border-collapse: collapse; }
        .data-table td { padding: 8px 5px; font-size: 15px; }
        .data-table .label { font-weight: bold; width: 40%; }
        .footer { margin-top: 40px; text-align: center; font-size: 11px; color: #aaa; border-top: 1px solid #eee; padding-top: 15px; }
        .print-btn-container { text-align: center; margin: 20px 0; }
        .print-btn {
            background: #0056b3; color: #fff; border: none; padding: 12px 40px;
            font-size: 16px; border-radius: 8px; cursor: pointer; font-family: inherit;
        }
        .print-btn:hover { background: #003d82; }
    </style>
</head>
<body>
    <div class="header">
        <h1>تقرير فحص شبكية العين بالذكاء الاصطناعي</h1>
        <p>تاريخ الفحص: ${dateStr} - ${timeStr}</p>
    </div>

    <table class="images-table">
        <tr>
            <td>
                <div class="img-title">الصورة الأصلية</div>
                <img src="${img1}" />
            </td>
            <td>
                <div class="img-title">تشخيص السكري</div>
                <div class="img-subtitle">(Grad-CAM)</div>
                <img src="${img2}" />
            </td>
            <td>
                <div class="img-title">فحص المياه الزرقاء</div>
                <div class="img-subtitle">(U-Net Mask)</div>
                <img src="${img3}" style="background:#000;" />
            </td>
        </tr>
    </table>

    <div class="result-box dr">
        <h2>نتائج اعتلال الشبكية السكري</h2>
        <table class="data-table">
            <tr><td class="label">النتيجة:</td><td>${drResult}</td></tr>
            <tr><td class="label">نسبة ثقة النموذج:</td><td>${drConfidence}</td></tr>
        </table>
    </div>

    <div class="result-box gl">
        <h2>نتائج فحص الجلوكوما</h2>
        <table class="data-table">
            <tr><td class="label">نسبة التقعر للقرص (vCDR):</td><td>${vcdrValue}</td></tr>
        </table>
    </div>

    <div class="footer">
        <p>هذا التقرير تم إنشاؤه آلياً بواسطة نظام Digilians</p>
        <p>يجب مراجعة هذا التقرير من قبل طبيب مختص.</p>
    </div>

    <div class="print-btn-container no-print">
        <button class="print-btn" onclick="window.print()">📥 طباعة / حفظ كـ PDF</button>
        <p style="margin-top:10px; font-size:13px; color:#888;">اختر "Save as PDF" أو "حفظ كـ PDF" من نافذة الطباعة</p>
    </div>
</body>
</html>`;

    // 3. Open a new window and write the report into it
    const reportWindow = window.open('', '_blank');
    if (!reportWindow) {
        alert('يرجى السماح بالنوافذ المنبثقة (Pop-ups) في المتصفح لتحميل التقرير.');
        return;
    }
    reportWindow.document.write(fullHTML);
    reportWindow.document.close();

    // 4. Auto-trigger print after images load
    reportWindow.onload = function() {
        setTimeout(() => {
            reportWindow.print();
        }, 500);
    };
}

function resetUI() {
    // Reset to initial state
    document.getElementById('results-container').classList.add('hidden');
    document.getElementById('preview-container').classList.add('hidden');
    document.getElementById('upload-area').classList.remove('hidden');
    document.getElementById('explain-section').classList.add('hidden');
    const glaucomaSection = document.getElementById('glaucoma-section');
    if (glaucomaSection) glaucomaSection.classList.add('hidden');
    const maskContainer = document.getElementById('cdr-mask-container');
    if (maskContainer) maskContainer.classList.add('hidden');
    
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
    try {
        // Show loading state if needed, but handleFile does this anyway
        const response = await fetch(`/static/demo_images/${sample.file}`);
        const blob = await response.blob();
        
        // Convert the blob from the demo image into a real File object
        const file = new File([blob], sample.file, { type: blob.type || 'image/png' });
        
        // Pass it to the normal flow as if the user uploaded it
        handleFile(file);
    } catch (e) {
        console.error("Failed to load demo image blob", e);
        alert('فشل في تحميل عينة التجربة. يرجى التأكد من اتصال الشبكة.');
    }
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