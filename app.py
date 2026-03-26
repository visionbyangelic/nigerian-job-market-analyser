import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import kagglehub
from openai import AzureOpenAI

st.set_page_config(page_title="Nigerian Tech Job Market Analyser", layout="wide")
st.title("🧠 Nigerian Tech Job Market Analyser")
st.markdown("Multi-agent GenAI system powered by **Azure OpenAI GPT-4o** and the LinkedIn Job Postings dataset.")

# Sidebar config
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key    = st.text_input("Azure OpenAI API Key", type="password")
    endpoint   = st.text_input("Azure Endpoint", value="https://df202526.openai.azure.com/")
    deployment = st.text_input("Deployment Name", value="gpt-4o")
    n_skills   = st.slider("Top N Skills",   5, 20, 12)
    n_salaries = st.slider("Top N Salaries", 5, 20, 10)
    run_btn    = st.button("🚀 Run Analysis")

# --- Session state init ---
if "report" not in st.session_state:
    st.session_state.report = None
if "skills_out" not in st.session_state:
    st.session_state.skills_out = None
if "salary_out" not in st.session_state:
    st.session_state.salary_out = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "client" not in st.session_state:
    st.session_state.client = None
if "deployment" not in st.session_state:
    st.session_state.deployment = "gpt-4o"

# --- Run analysis ---
if run_btn:
    if not api_key:
        st.error("Please enter your Azure OpenAI API key.")
        st.stop()

    client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version="2024-02-01")
    st.session_state.client = client
    st.session_state.deployment = deployment

    with st.spinner("📡 Downloading dataset..."):
        BASE = kagglehub.dataset_download('arshkon/linkedin-job-postings')
        postings   = pd.read_csv(f"{BASE}/postings.csv").head(5000)
        job_skills = pd.read_csv(f"{BASE}/jobs/job_skills.csv")
        skills_map = pd.read_csv(f"{BASE}/mappings/skills.csv")
        salaries   = pd.read_csv(f"{BASE}/jobs/salaries.csv")

    def get_top_skills(n=10):
        sf = job_skills.merge(skills_map, on='skill_abr')
        return sf['skill_name'].value_counts().head(n).to_dict()

    def get_top_salaries(n=10):
        m = salaries.merge(postings[['job_id', 'title']], on='job_id')
        return m.groupby('title')['max_salary'].median().sort_values(ascending=False).head(n).to_dict()

    TOOLS = [
        {"type": "function", "function": {"name": "get_top_skills",   "description": "Returns top N in-demand skills.",  "parameters": {"type": "object", "properties": {"n": {"type": "integer", "default": 10}}, "required": []}}},
        {"type": "function", "function": {"name": "get_top_salaries", "description": "Returns top N paying roles.",      "parameters": {"type": "object", "properties": {"n": {"type": "integer", "default": 10}}, "required": []}}}
    ]
    TOOL_MAP = {"get_top_skills": get_top_skills, "get_top_salaries": get_top_salaries}

    def run_agent(system_prompt, user_message, tools=None, max_iterations=5):
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]
        for _ in range(max_iterations):
            kwargs = {"model": deployment, "messages": messages}
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            resp = client.chat.completions.create(**kwargs)
            msg  = resp.choices[0].message
            if not msg.tool_calls:
                return msg.content
            messages.append(msg)
            for tc in msg.tool_calls:
                result = TOOL_MAP[tc.function.name](**json.loads(tc.function.arguments))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
        return "Max iterations reached."

    col1, col2 = st.columns(2)

    with col1:
        with st.spinner("🧠 SkillsAnalyst running..."):
            skills_out = run_agent(
                "You are a tech skills analyst. Use get_top_skills to fetch data, then identify the top 3 Power Skills for 2026 for Nigerian job seekers.",
                f"Fetch the top {n_skills} skills and analyse them.",
                tools=TOOLS
            )
        st.session_state.skills_out = skills_out
        st.subheader("🧠 Skills Analysis")
        st.write(skills_out)

        skills_data = get_top_skills(n_skills)
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        sd = pd.DataFrame(list(skills_data.items()), columns=['Skill', 'Count'])
        sns.barplot(data=sd, x='Count', y='Skill', ax=ax1, palette='viridis')
        ax1.set_title('Top In-Demand Skills')
        st.pyplot(fig1)

    with col2:
        with st.spinner("📈 TrendForecaster running..."):
            salary_out = run_agent(
                "You are a salary strategist. Use get_top_salaries to fetch data, then pick the top 3 roles with the best salary-to-entry-barrier ratio.",
                f"Fetch the top {n_salaries} salaries and analyse them.",
                tools=TOOLS
            )
        st.session_state.salary_out = salary_out
        st.subheader("📈 Salary Forecast")
        st.write(salary_out)

        salary_data = get_top_salaries(n_salaries)
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        sal = pd.DataFrame(list(salary_data.items()), columns=['Role', 'Median Max Salary'])
        sns.barplot(data=sal, x='Median Max Salary', y='Role', ax=ax2, palette='magma')
        ax2.set_title('Top Paying Roles')
        st.pyplot(fig2)

    with st.spinner("📝 ReportWriter generating final report..."):
        report_str = run_agent(
            'You are a report writer. Return ONLY valid raw JSON: {"summary":"","priority_skills":[{"name":"","why":""}],"target_roles":[{"title":"","median_max_salary":0,"why":""}],"action_plan":[]}',
            f"Skills analysis:\n{skills_out}\n\nSalary forecast:\n{salary_out}"
        )

    try:
        report = json.loads(report_str.replace('```json', '').replace('```', '').strip())
        st.session_state.report = report
    except:
        st.session_state.report = {"raw": report_str}

    st.subheader("📋 Final Career Strategy Report")
    st.json(st.session_state.report)
    st.download_button("⬇️ Download Report", json.dumps(st.session_state.report, indent=2), "career_report.json", "application/json")

    # Reset chat when re-running analysis
    st.session_state.chat_history = []

# --- Chat interface (shows after analysis runs) ---
if st.session_state.report:
    st.markdown("---")
    st.subheader("💬 Ask the Agents")
    st.markdown("Ask anything about the job market analysis — skills, salaries, career advice, next steps.")

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    user_input = st.chat_input("e.g. What skill should I learn first? Which role suits a junior?")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                system = f"""You are a career advisor with access to a LinkedIn job market analysis report.
Use the following context to answer the user's questions accurately and practically.

Skills Analysis:
{st.session_state.skills_out}

Salary Forecast:
{st.session_state.salary_out}

Final Report:
{json.dumps(st.session_state.report, indent=2)}

Be concise, direct, and practical. If the question is outside the data, say so honestly."""

                messages = [{"role": "system", "content": system}]
                for msg in st.session_state.chat_history:
                    messages.append({"role": msg["role"], "content": msg["content"]})

                try:
                    resp = st.session_state.client.chat.completions.create(
                        model=st.session_state.deployment,
                        messages=messages,
                        max_tokens=500
                    )
                    answer = resp.choices[0].message.content
                except Exception as e:
                    answer = f"Error: {e}"

                st.write(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
