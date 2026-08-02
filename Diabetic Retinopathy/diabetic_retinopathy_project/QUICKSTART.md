# 🚀 QUICKSTART - Diabetic Retinopathy

## 🐳 Docker Deployment (Production)

The project is fully containerized for production deployment.

1. **Build the Docker Image**:
   ```bash
   docker build -t diabetic-retinopathy-api .
   ```

2. **Run the Docker Container**:
   ```bash
   docker run -d -p 8000:8000 --name dr-api diabetic-retinopathy-api
   ```

The API and Web UI will be available at `http://localhost:8000`.

## 📁 Project Structure

## ✅ المشروع مكتمل بالكامل

تم بناء كل شيء:
- نموذج AI (EfficientNet-B0 + Transfer Learning)
- بايبلاين تدريب كامل
- تفسيرية (Grad-CAM)
- تطبيق ويب متكامل (FastAPI + واجهة عربية)

---

## تشغيل فوري (في أقل من 60 ثانية)

### 1. تشغيل التطبيق الويب

```bash
cd diabetic_retinopathy_project

# تشغيل الخادم
cd app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

ثم افتح في المتصفح:
**http://localhost:8000**

### 2. تشغيل البايبلاين التجريبي الكامل (مرة واحدة فقط)

```bash
cd diabetic_retinopathy_project
python scripts/run_demo_pipeline.py
```

هذا يدرب نموذجًا ويُنشئ:
- نموذج مدرب (`models/best_model.pth`)
- Grad-CAM تفسيرات
- تقارير التقييم

---

## كيفية استخدام التطبيق

1. ارفع صورة قاع عين (fundus image)
2. اضغط **"تحليل الصورة"** أو **"تحليل + تفسير"**
3. احصل على:
   - الدرجة (0-4)
   - احتماليات
   - التوصية الطبية (عربي + إنجليزي)
   - خريطة Grad-CAM (شرح النموذج)

---

## الملفات الرئيسية

| المسار                        | الوصف                              |
|-------------------------------|------------------------------------|
| `app/main.py`                 | خادم FastAPI                       |
| `app/templates/index.html`    | واجهة المستخدم العربية            |
| `src/inference.py`            | محرك التنبؤ الجاهز                |
| `models/best_model.pth`       | النموذج المدرب (ديمو)             |
| `scripts/run_demo_pipeline.py`| تشغيل كل شيء من الصفر             |

---

## الخطوات التالية الموصى بها

1. **الآن**: شغّل الويب (`uvicorn`)
2. **لاحقًا**: حمل بيانات APTOS الحقيقية:
   ```bash
   python scripts/download_data.py
   python scripts/prepare_data.py
   ```
3. **Train Model** (Trains an EfficientNet-B3 model):
   ```bash
   python scripts/train_full.py --epochs 15 --batch-size 32
   ```
   *(Expected Real Validation QWK: ~0.83, Accuracy: ~68%)*

---

**المشروع جاهز للعرض، التطوير، أو النشر.**