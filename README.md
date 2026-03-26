# 🧠 Nigerian Tech Job Market Analyser
### A Multi-Agent GenAI System powered by Azure OpenAI

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nigerian-job-market-analyser.streamlit.app)

[streamlit-app-2026-03-26-23-41-20.webm](https://github.com/user-attachments/assets/3a6eb885-7f92-476c-b351-e2034ef6516a)


A multi-agent AI system that analyses the LinkedIn Job Postings dataset to surface in-demand skills, top-paying roles, and actionable career strategies — with a built-in chat interface to ask follow-up questions about the data.

---

## 📌 Project Overview

This project was built as a capstone for a Generative AI course, demonstrating how multiple specialised AI agents can work together in a sequential pipeline to solve a real-world data science problem.

The system ingests LinkedIn job postings data, runs it through four specialised agents, produces a structured JSON career strategy report with data visualisations, and exposes everything through a Streamlit web app where users can chat with the agents about the findings.

---

## 🏗️ Architecture

```
DataLoaderAgent → SkillsAnalystAgent → TrendForecasterAgent → ReportWriterAgent
```

The system follows a **Sequential Pipeline** pattern — each agent completes its task and passes its output downstream to the next agent.

### Agents

| Agent | Role | Tools Used |
|---|---|---|
| `DataLoaderAgent` | Fetches and summarises raw skills and salary data | `get_top_skills`, `get_top_salaries` |
| `SkillsAnalystAgent` | Identifies the top 3 "Power Skills" for 2026 | `get_top_skills` |
| `TrendForecasterAgent` | Finds roles with the best salary-to-entry-barrier ratio | `get_top_salaries` |
| `ReportWriterAgent` | Synthesises all agent outputs into a final JSON report | None (works from inputs) |

All agents inherit from a shared base `Agent` class that handles the OpenAI function calling loop, tool dispatch, error handling, and `max_iterations` safety guard.

---

## 🛠️ Tech Stack

- **LLM:** Azure OpenAI (GPT-4o)
- **Multi-agent framework:** Custom `Agent` base class with function calling API
- **Dataset:** [LinkedIn Job Postings — Kaggle (arshkon)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)
- **Data processing:** pandas
- **Visualisation:** matplotlib, seaborn
- **UI:** Streamlit
- **Deployment:** Streamlit Cloud

---

## ✅ Assignment Requirements Coverage

| Requirement | Implementation |
|---|---|
| Real dataset (500+ rows) | LinkedIn Job Postings — 5,000 postings loaded |
| 3+ specialised agents with distinct system prompts | DataLoaderAgent, SkillsAnalystAgent, TrendForecasterAgent, ReportWriterAgent |
| Agents implemented as `Agent` class or subclass | All 4 agents subclass the base `Agent` class |
| Multi-agent pattern | Sequential pipeline |
| Azure OpenAI as LLM | `AzureOpenAI` client used throughout |
| At least 2 custom tools with function calling | `get_top_skills` and `get_top_salaries` with full OpenAI tool schemas |
| Final structured output | JSON career strategy report |
| Data visualisation | 2 seaborn charts — top skills bar chart and top salaries bar chart |
| `max_iterations` safety guard | Implemented in base `Agent.run()` |
| Graceful error handling | try/except in every agent call |
| Streamlit UI (bonus) | Full Streamlit app with analysis + chat interface |

---

## 🔧 Custom Tools

### `get_top_skills(n: int)`
Merges the `job_skills` and `skills` mapping CSVs from the LinkedIn dataset and returns the top N most frequently required skills across all job postings.

```python
get_top_skills(n=10)
# Returns: {"Information Technology": 4523, "Sales": 3201, ...}
```

### `get_top_salaries(n: int)`
Merges salary data with job postings and returns the top N highest paying job titles by median maximum salary.

```python
get_top_salaries(n=10)
# Returns: {"Senior Database Administrator": 1300000, ...}
```

Both tools are registered with the OpenAI function calling API using full JSON schemas, allowing agents to decide autonomously when and how to call them.

---

## 🚀 Running the App

### On Streamlit Cloud (recommended)
The app is deployed and accessible at:
👉 **[nigerian-job-market-analyser.streamlit.app](https://nigerian-job-market-analyser.streamlit.app)**

You will need your own Azure OpenAI API key to run the analysis.

### Locally

**1. Clone the repo**
```bash
git clone https://github.com/visionbyangelic/nigerian-job-market-analyser.git
cd nigerian-job-market-analyser
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

**4. In the sidebar:**
- Enter your Azure OpenAI API key
- Enter your Azure endpoint
- Enter your deployment name
- Click **Run Analysis**

---

## 🔑 Azure OpenAI Setup

You need an Azure OpenAI resource with a GPT-4o deployment to use this app.

1. Go to [Azure AI Studio](https://ai.azure.com)
2. Create a deployment (e.g. `gpt-4o`)
3. Copy your endpoint and API key
4. Paste them into the app sidebar

> Each user provides their own API key. No keys are stored or shared.

---

## 📊 Output

The system produces:

**1. Skills Analysis** — written breakdown of the top 3 power skills for Nigerian tech job seekers in 2026

**2. Salary Forecast** — top 3 roles ranked by salary-to-entry-barrier ratio with reasoning

**3. Data Visualisations** — two bar charts showing top in-demand skills and top paying roles

**4. JSON Career Strategy Report** — structured output with this shape:
```json
{
  "summary": "...",
  "priority_skills": [
    {"name": "...", "why": "..."}
  ],
  "target_roles": [
    {"title": "...", "median_max_salary": 0, "why": "..."}
  ],
  "action_plan": ["step 1", "step 2", "step 3"]
}
```

**5. Chat interface** — ask follow-up questions grounded in the analysis, e.g.:
- *"What skill should I learn first if I'm just starting out?"*
- *"Which role has the best salary for someone without a degree?"*
- *"Give me a 3-month action plan to break into data science."*

---

## 📁 Project Structure

```
nigerian-job-market-analyser/
│
├── app.py                  # Streamlit web app
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

The core notebook (used for development and submission) contains:
```
job-market-analyser.ipynb   # Full pipeline with all agent classes
```

---

## 📦 Dependencies

```
openai
kagglehub
pandas
matplotlib
seaborn
streamlit>=1.32.0
altair>=5.0.0
```

---

## 👩🏾‍💻 Author

**Angelic Charles**
Data Scientist | Software Engineering Student

- GitHub: [@visionbyangelic](https://github.com/visionbyangelic)
- Portfolio: [visionbyangelic.github.io](https://visionbyangelic.github.io)
- ORCID: [0009-0008-7279-9663](https://orcid.org/0009-0008-7279-9663)

---

## 📄 Dataset

**LinkedIn Job Postings** by Arsh Kon  
[kaggle.com/datasets/arshkon/linkedin-job-postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)

Downloaded via `kagglehub` at runtime — no data is stored in this repository.
