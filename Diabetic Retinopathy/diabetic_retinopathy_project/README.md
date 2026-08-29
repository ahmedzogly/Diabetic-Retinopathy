<div align="center">
  <h1>👁️ Diabetic Retinopathy & Glaucoma Detection AI</h1>
  <p><i>نظام ذكاء اصطناعي طبي متكامل لتشخيص اعتلال الشبكية السكري وفحص المياه الزرقاء (الجلوكوما) من صور قاع العين</i></p>
  <p><i>A comprehensive, production-ready AI pipeline & web platform for automated fundus image screening</i></p>
  
  <p>
    <img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C.svg?logo=pytorch" alt="PyTorch">
    <img src="https://img.shields.io/badge/FastAPI-0.103.1-009688.svg?logo=fastapi" alt="FastAPI">
    <img src="https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker" alt="Docker">
    <img src="https://img.shields.io/badge/Explainable_AI-Grad--CAM-orange.svg" alt="Grad-CAM">
    <img src="https://img.shields.io/badge/Segmentation-U--Net-purple.svg" alt="U-Net">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  </p>
</div>

<br>

---

## 📸 معرض واجهة المستخدم وسير العمل (Application Interface & Workflow)

يوفر النظام واجهة مستخدم ويب حديثة وسلسة باللغتين العربية والإنجليزية، تدعم التحليل الفوري لصور قاع العين مع تقارير طبية قابلة للطباعة:

---

### 1. واجهة الفحص والرفع (Hero Section & Image Upload)
<div align="center">
  <img src="docs/screenshots/01_hero_and_upload.png" alt="واجهة الفحص والرفع" width="850">
</div>

* **الوصف (AR):** الواجهة الرئيسية تتيح للطبيب أو أخصائي العيون رفع صور قاع العين (`Fundus Images`) بصيغ متعددة (JPG / PNG) بالسحب والإفلات أو التحديد المباشر مع معاينة فورية للصورة قبل الفحص.
* **Description (EN):** The interactive landing screen allows ophthalmologists to drag-and-drop or select high-resolution retinal fundus images with instant visual preview prior to AI inference.

---

### 2. نتائج التشخيص السريري والتوصيات (Clinical Diagnostics & Probability Breakdown)
<div align="center">
  <img src="docs/screenshots/02_diagnostic_results.png" alt="نتائج التشخيص السريري" width="850">
</div>

* **الوصف (AR):** عرض دقيق لدرجة اعتلال الشبكية السكري (طبيعي / خفيف / متوسط / شديد / تكاثري) مع نسبة ثقة النموذج، وتوزيع احتمالي تفصيلي لجميع الدرجات، بالإضافة إلى التوصية الإكلينيكية المعتمدة والفحص السريع لمؤشر الجلوكوما ($vCDR$).
* **Description (EN):** Real-time severity classification across 5 clinical stages (No DR, Mild, Moderate, Severe, Proliferative) paired with model confidence bars, multiclass probability distributions, actionable clinical guidelines, and vertical Cup-to-Disc Ratio ($vCDR$).

---

### 3. التفسيرية البصرية بالذكاء الاصطناعي وفحص الجلوكوما (Explainable AI & Glaucoma Segmentation)
<div align="center">
  <img src="docs/screenshots/03_xai_and_glaucoma.png" alt="التفسيرية البصرية وفحص الجلوكوما" width="850">
</div>

* **الوصف (AR):** 
  * **الخريطة الحرارية التفسيرية (Grad-CAM):** تبرز المناطق الدقيقة في الشبكية والأوعية الدموية التي استند إليها النموذج في اتخاذ القرار الطبي.
  * **تقطيع القرص والتقعر البصري (U-Net Segmentation Mask):** استخراج دقيق للقرص البصري (Optic Disc) والتقعر (Optic Cup) لحساب نسبة الجلوكوما بدقة.
  * **عينات تجريبية:** شريط سفلي يتيح اختبار عينات فورية مصنفة مسبقاً لجميع درجات المرض بنقرة واحدة.
* **Description (EN):** Highlights critical retinal regions and microvascular lesions via **Grad-CAM heatmaps**, generates semantic segmentation masks for the Optic Disc & Cup using **U-Net**, and offers one-click preloaded demo benchmark samples.

---

### 4. التقرير الطبي المعتمد الجاهز للطباعة والـ PDF (Automated Medical PDF Report)
<div align="center">
  <img src="docs/screenshots/04_printable_medical_report.png" alt="التقرير الطبي المعتمد" width="850">
</div>

* **الوصف (AR):** توليد تقرير طبي رسمي متكامل بنقرة واحدة، يجمع الصورة الأصلية وخريطة Grad-CAM وقناع U-Net مع تفاصيل التشخيص الثنائية (عربي/إنجليزي)، مصمم للطباعة المباشرة وحفظه كملف `PDF` عالي الجودة لملفات المرضى.
* **Description (EN):** Generates an instant, publication-quality bilingual medical report displaying the input fundus image, explainability heatmaps, segmentation masks, and clinical findings, fully printable and downloadable as a PDF.

---

### 5. خلفية المشروع والأثر الصحي (Project Impact & Clinical Significance)
<div align="center">
  <img src="docs/screenshots/05_project_info_and_impact.png" alt="خلفية المشروع والأثر الصحي" width="850">
</div>

* **الوصف (AR):** تسليط الضوء على الأهمية المجتمعية والطبية للمشروع، خاصة في مصر والعالم العربي لمواجهة نقص أطباء العيون في المناطق النائية وتقليل مخاطر فقدان البصر الناتج عن مضاعفات السكري.
* **Description (EN):** Contextual overview highlighting the technological foundation (Deep Learning, U-Net, Grad-CAM) and the socioeconomic healthcare impact in addressing diabetic complications and ophthalmologist shortages.

---

## ✨ المميزات الرئيسية (Key Features)

* 🧠 **Deep Learning Diagnosis:** مبني على نموذج **EfficientNet-B3** مع Transfer Learning على بيانات APTOS العالمية. محققاً نتائج فائقة: **Quadratic Weighted Kappa (QWK): 0.88**، ودقة عامة متقدمة.
* 🔍 **Explainable AI (Grad-CAM):** توفير شفافية كاملة للقرارات الطبية عبر الخرائط الحرارية لمساعدة الأطباء في التحقق من الآفات الشبكية.
* 👁️ **Glaucoma Screening ($vCDR$):** خوارزمية ذكية مدمجة مع شبكة تقطيع **U-Net** لحساب نسبة التقعر للقرص البصري وكشف مؤشرات المياه الزرقاء المبكرة.
* 📄 **Instant PDF Reports:** تصدير وطباعة تقارير طبية رسمية منسقة تدعم اللغة العربية والإنجليزية بجودة عالية.
* 🐳 **Production & Docker Ready:** تطبيق مهيأ بالكامل للنشر السحابي والمحلي عبر **Docker** و **FastAPI** بخفة وسرعة استجابة فائقة.

---

## 🔬 المعالجة الطبية للصور (Medical Preprocessing Pipeline)

لضمان أعلى دقة في التنبؤ، تمر كل صورة قاع عين بخطوات معالجة طبية صارمة:
1. **Dark Border Cropping:** إزالة الحواف السوداء غير المفيدة المحيطة بالصورة تلقائياً.
2. **Ben Graham's Method:** تطبيق Gaussian Blur معادل لتطبيع الإضاءة وإبراز التباين في الشعيرات الدموية والارتشاحات الدقيقة.

---

## 🛠️ حزمة التقنيات (Tech Stack)

* **Deep Learning:** PyTorch, Torchvision, PyTorch Grad-CAM, Segmentation Models
* **Computer Vision:** OpenCV, Albumentations, PIL, NumPy
* **Backend & API:** FastAPI, Uvicorn, Pydantic
* **Frontend:** Vanilla JavaScript, HTML5, Modern CSS (Responsive Glassmorphism & Medical Theme)
* **Reporting:** Native High-Fidelity Browser Print-to-PDF Pipeline
* **Deployment:** Docker, Uvicorn ASGI Server

---

## 🚀 التشغيل السريع (Quick Start with Docker)

أسرع وأسهل طريقة لتشغيل النظام بدون الحاجة لتثبيت أي بيئات عمل يدوياً:

**1. استنساخ المستودع (Clone Repo):**
```bash
git clone https://github.com/ahmedzogly/Diabetic-Retinopathy.git
cd "Diabetic-Retinopathy/Diabetic Retinopathy/diabetic_retinopathy_project"
```

**2. بناء حاوية الدوكر (Build Docker Image):**
```bash
docker build -t diabetic-retinopathy-api .
```

**3. تشغيل الحاوية (Run Container):**
```bash
docker run -d -p 8000:8000 --name dr-api diabetic-retinopathy-api
```

🌐 التطبيق متاح ومباشر الآن عبر المتصفح: **[http://localhost:8000](http://localhost:8000)**

---

## 📁 هيكل المشروع (Project Structure)

```text
Diabetic Retinopathy/
├── app/                  # تطبيق الويب وواجهة المستخدم
│   ├── static/           # ملفات CSS والتفاعل JS والصور التجريبية
│   ├── templates/        # قوالب HTML (index.html)
│   └── main.py           # خادم FastAPI ونقاط الـ API
├── docs/                 # التوثيق والصور المعمارية
│   └── screenshots/      # لقطات شاشة واجهة النظام والتقارير
├── src/                  # النماذج والمنطق الرياضي للذكاء الاصطناعي
│   ├── classifier_model.py # معمارية EfficientNet-B3
│   ├── explain.py        # محرك توليد خرائط Grad-CAM
│   ├── inference.py      # خط المعالجة والاستدلال الموحد
│   ├── segmentation_model.py # نموذج U-Net لتقطيع القرص والتقعر
│   └── cdr_calculator.py # حساب نسبة vCDR للجلوكوما
├── models/               # الأوزان المدربة (best_model.pth)
├── scripts/              # سكربتات التدريب والتقييم والاختبار
├── Dockerfile            # بناء بيئة الإنتاج الموحدة
├── requirements.txt      # الحزم والمتطلبات
└── run_server.bat        # تشغيل فوري وسريع للـ Backend محلياً
```

---

## 🎓 فريق العمل والإشراف (Team & Acknowledgements)

تم تطوير هذا النظام كمشروع تخرج تحت الإشراف الكريم للدكتور:
**Dr. Mohamed Elhadad** 👨‍🏫


**فريق التطوير والبحث (Development Team):**
* 👨‍💻 **Ahmed Shehta Zoghli**
* 👨‍💻 **Eslam Tag Elser**
* 👨‍💻 **Mohamed Hassan Ahmed**
* 👨‍💻 **Osama Mohamed Kamel**
* 👨‍💻 **Ahmed Zain Elabiden**


> *"مكرس لتطوير حلول رعاية صحية ذكية ومتاحة للجميع باستخدام الذكاء الاصطناعي مفتوح المصدر."*
