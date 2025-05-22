## 🏡 Real Estate Data Engineering Using Multi-Cloud & LLM

![Banner](https://img.shields.io/badge/Cloud-AWS%20%7C%20GCP-blue?style=flat-square)
![Badge](https://img.shields.io/badge/AI-LLM%20%7C%20ChatGPT%20%7C%20Gemini-critical?style=flat-square)

> An AI-powered real estate analytics platform that leverages **AWS**, **Google Cloud**, **PySpark**, and **LLMs** (ChatGPT & Gemini) to deliver intelligent property insights, rent predictions, and personalized recommendations.

---

### 📽️ Demo Video

[![Watch the video](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=JA1kmpehzBM?si=tJ0FdqtEcheyiEew)

Replace `YOUR_VIDEO_ID` with your actual YouTube video ID.

---

### 📌 Features

✅ Real-time rental and buy price prediction using ML models
✅ Multi-cloud infrastructure (AWS S3, SageMaker, QuickSight + GCP Vertex AI)
✅ AI-powered recommendations using Gemini & ChatGPT APIs
✅ Interactive dashboards with Amazon QuickSight
✅ Streamlit-based intuitive UI for customers and owners
✅ Scalable ETL with PySpark

---

### 🧠 Tech Stack

| Layer               | Tools/Services                |
| ------------------- | ----------------------------- |
| **Frontend**        | Streamlit                     |
| **AI APIs**         | Google Gemini API, OpenAI GPT |
| **ML Platform**     | Google Vertex AI              |
| **Data Processing** | PySpark on AWS SageMaker      |
| **Storage**         | Amazon S3                     |
| **Visualization**   | Amazon QuickSight             |

---

### 📊 Architecture

![Architecture](https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME/blob/main/architecture_diagram.png) <sub>*(Upload `architecture_diagram.png` to your repo first if you have it in your report)*</sub>

---

### 📂 Project Structure

```
.
├── data/                         # Datasets & CSVs
├── notebooks/                   # SageMaker notebooks
├── models/                      # Pre-trained ML models
├── streamlit_customer_app.py    # Customer-facing app
├── streamlit_owner_app.py       # Owner-facing app
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

### 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR_USERNAME/real-estate-ai-platform.git
   cd real-estate-ai-platform
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   streamlit run streamlit_customer_app.py
   ```

4. (Optional) To use the owner dashboard:

   ```bash
   streamlit run streamlit_owner_app.py
   ```

---

### 📈 Sample Output

#### Customer Recommendation (via Gemini)

```
✅ Recommended: RE29940196 in Adyar
💡 Insights: 50% larger area, lower rent, superior amenities.
🧠 Advice: Negotiate 5–7% rent reduction.
```

#### Owner Prediction (via OpenAI)

```
💰 Predicted Price/Sqft: ₹13,760
📊 Estimated Total: ₹13,759,720
✨ GPT Suggestion: Increase by ₹1,240,000 based on features.
```

---

### 🔐 Security & Access

* AWS IAM roles for secure service interaction
* API keys stored via AWS Secrets Manager & Vertex secret integration
* Owner contact data removed from visualization for privacy

---

### 📚 Project Report

Refer to the full PDF report [here](./realstate_project_report.pdf) for technical details and implementation explanation.

---

### 🙌 Contributors

* Jujjavarapu Sujan Chowdary
* Gandam Sai Ram
* Dileep Sai Ande
* Naveen Kumar Reddy

Under the guidance of **Dr. S. Prabakeran**, SRM IST

---

Would you like me to generate the `README.md` file and upload it here directly for convenience?
