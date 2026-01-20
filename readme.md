# Artificial Intelligent Investor (AII) 📈🤖

![Python](https://img.shields.io/badge/Python-3.13-blue) ![Django](https://img.shields.io/badge/Django-5.2-green) ![AI](https://img.shields.io/badge/AI-Gemini%203-orange) ![Status](https://img.shields.io/badge/Status-Live-success)

A comprehensive full-stack stock analysis platform that combines **real-time financial data** with **Generative AI** to provide institutional-grade investment insights.

**🔗 Live Demo:** [https://artificial-intelligent-investor.onrender.com](https://artificial-intelligent-investor.onrender.com)

---

## 📖 Overview

**Artificial Intelligent Investor** solves the problem of information overload in stock research. Instead of opening multiple tabs for charts, news, and financial statements, AII aggregates everything into a single dashboard and uses Google's Gemini AI to synthesize the data into a clear, human-readable qualitative analysis.

This project was built to simulate a real-world fintech environment, handling challenges like API rate limiting, efficient search algorithms, and cloud deployment constraints.

---

## ✨ Key Features

* **🤖 AI-Powered Analysis:** Generates "Pros," "Cons," and a "Summary" for any stock using **Google Gemini 2.5**, grounded in live financial data.
* **📊 Interactive Charts:** Dynamic 5-year price history charts with 1M/3M/6M/1Y/5Y timeframes using **Plotly**.
* **⚡ Instant Search:** Custom **Trie Data Structure** implementation for O(L) time complexity search (autocomplete) across 2000+ stocks.
* **📉 Financial Deep Dive:**
    * Quarterly & Annual Profit/Loss Statements.
    * Balance Sheets & Cash Flow Statements.
    * Key Ratios (P/E, ROE, ROCE, Debt-to-Equity).
* **👥 Peer Comparison:** Automatically benchmarks the stock against top competitors in the same industry.
* **📰 Smart News:** Aggregates recent news headlines and performs **Sentiment Analysis** (Positive/Negative/Neutral) using NLTK.
* **🔐 Secure Auth:** Complete user authentication system with **Google OAuth** integration.

---

## 🛠️ Tech Stack

### Backend
* **Framework:** Django 5.2 (Python 3.13)
* **Database:** PostgreSQL (Production), SQLite (Dev)
* **AI Model:** Google Gemini 2.5 Flash & Pro via `google-genai` SDK
* **Financial Data:** Yahoo Finance API (`yfinance`) with `curl_cffi` for bot-detection bypass.

### Frontend
* **Templating:** Django Templates (DTL)
* **Styling:** Custom CSS (Responsive Design)
* **Visualization:** Plotly.js
* **Interactivity:** HTMX (for lazy loading components)

### Infrastructure
* **Deployment:** Render Cloud
* **Server:** Gunicorn with WhiteNoise for static files.
* **Caching:** Django `LocMemCache` (Local Memory Cache).

---

## ⚙️ Technical Architecture & Challenges

### 1. Robust API Retrieval (The "Rate Limit" Problem)
Fetching charts, ratios, news, and financials simultaneously triggered Yahoo Finance's rate limits immediately.
* **Solution:** I engineered a custom `CachedTicker` wrapper with exponential backoff and retry logic. It uses **Django's Local Memory Cache** to store data for 30 minutes, ensuring we never hit the API unnecessarily.
* **Outcome:** Reduced API calls by ~90% and eliminated 500-error crashes during high traffic.

### 2. Efficient Search (The "O(N)" Problem)
Searching through thousands of stock tickers using standard database queries (`LIKE %query%`) was becoming slow.
* **Solution:** I implemented a **Trie (Prefix Tree)** data structure from scratch in Python. It loads stock tickers into memory at server startup.
* **Outcome:** Search complexity reduced from O(N) (scanning the DB) to O(L) (length of the search term), resulting in near-instant autocomplete results.

### 3. Memory Optimization (The "OOM" Problem)
Heavy libraries like Pandas, NLTK, and Google GenAI were causing Out-Of-Memory (OOM) crashes on the free-tier cloud instance during startup.
* **Solution:** Implemented **Lazy Loading** patterns. Heavy libraries are only imported inside the specific view functions that need them, drastically reducing the initial memory footprint of the application.

---

## 🚀 Installation & Setup

Follow these steps to run the project locally.

### Prerequisites
* Python 3.10+
* Git

### 1. Clone the Repository
```bash
git clone [https://github.com/FireStorm3444/artificial-intelligent-investor.git](https://github.com/yourusername/artificial-intelligent-investor.git)
cd artificial-intelligent-investor
```

### 2. Create a Virtual Environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

4. Set Up Environment Variables
Create a .env file in the root directory and add the following keys:
```
DEBUG=True
SECRET_KEY=your_secret_key_here
GEMINI_API_KEY=your_google_gemini_api_key
# Optional (for Google Login)
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
```

5. Run Migrations & Load Data
```bash
python manage.py migrate
```

6. Run the Server
```bash
python manage.py runserver
```
Visit http://127.0.0.1:8000 in your browser.

## Project Structure
```
├── aii/                # Project Settings
├── core/               # Main Application Logic
│   ├── views.py        # Controllers (Stock Logic)
│   ├── utils.py        # Caching & Retry Logic
│   ├── trie.py         # Custom Search Algorithm
│   └── models.py       # Database Schema
├── users/              # Authentication & User Profiles
├── templates/          # HTML Templates
└── static/             # CSS & Images
```

## 🤝 Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements.

- Fork the Project
- Create your Feature Branch (```git checkout -b feature/AmazingFeature```)
- Commit your Changes (```git commit -m 'Add some AmazingFeature'```)
- Push to the Branch (```git push origin feature/AmazingFeature```)
- Open a Pull Request

📄 License
Distributed under the MIT License. See LICENSE for more information.

Built with ❤️ by Shekhar Kumar [LinkedIn](https://www.linkedin.com/in/shekhar-coder)