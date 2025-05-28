
# 📈 Stock Recommender Ultimate

هذا المشروع عبارة عن نظام توصية أسهم ذكي يعتمد على تقنيات الذكاء الاصطناعي الحديثة، مثل:
- LSTM (Long Short-Term Memory) للتنبؤ بالسلاسل الزمنية
- Prophet (من Meta) للتوقعات الموسمية
- المؤشرات الفنية (RSI, MACD, Bollinger Bands)
- تحسين المعلمات باستخدام Optuna
- وحدة اختبارات للتأكد من جودة النماذج
- تكامل CI/CD باستخدام GitHub Actions
- نشر باستخدام Docker

## 🏗️ هيكل المشروع

```
/
├── dashboard/
│   └── app.py
├── data/
│   └── feature_engineering.py
├── models/
│   ├── advanced_lstm.py
│   └── prophet_model.py
├── tests/
│   ├── test_advanced_lstm.py
│   └── test_prophet_model.py
├── hyperparameter_tuning.py
├── requirements-upgraded.txt
├── Dockerfile
└── .github/
    └── workflows/
        └── ci.yml
```

## ⚙️ المتطلبات

- Python 3.10+
- مكتبات: pandas, numpy, tensorflow, prophet, ta, optuna, streamlit, plotly

لتثبيت:
```
pip install -r requirements-upgraded.txt
```

## 🚀 تشغيل المشروع

```
streamlit run dashboard/app.py
```

## 🧪 تشغيل الاختبارات

```
python -m unittest discover tests
```

## 🐳 تشغيل باستخدام Docker

```
docker build -t stock-recommender .
docker run -p 8501:8501 stock-recommender
```

## 🔧 تحسينات مقترحة

✅ دمج نماذج إضافية مثل XGBoost أو RandomForest  
✅ إضافة تحليل معنوي (sentiment analysis) من الأخبار والتغريدات  
✅ استخدام transformers لتوقعات السلاسل الزمنية  
✅ إنشاء لوحة قيادة تفاعلية بواجهة حديثة

