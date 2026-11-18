
import os
from pathlib import Path
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process, LLM
from ingest import search_research_papers, save_papers


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env", override=True)

API_KEY = os.getenv("GROQ_API_KEY", "").strip()
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()

REPORT_DIR = BASE_DIR / "generated_reports"
REPORT_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="Research Nexus AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# 2. CUSTOM UI DESIGN - PROFESSIONAL LIGHT THEME
# ==========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    color-scheme: dark;
    --bg: #090D14;
    --surface: #111824;
    --surface-2: #172231;
    --surface-3: #1D2A3B;
    --border: #29384B;
    --text: #EAF0F8;
    --muted: #A8B5C6;
    --muted-2: #8292A7;
    --accent: #55D6BE;
    --accent-2: #72A7FF;
    --gold: #E8C779;
    --danger: #FF7F86;
    --shadow: 0 14px 38px rgba(0,0,0,.24);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    color: var(--text);
    background:
        radial-gradient(ellipse at 88% 0%, rgba(85,214,190,.10), transparent 28%),
        radial-gradient(ellipse at 0% 35%, rgba(114,167,255,.07), transparent 32%),
        linear-gradient(145deg, #090D14 0%, #0D1420 55%, #0A1019 100%);
}
[data-testid="stHeader"] {
    background: rgba(9,13,20,.78);
}
[data-testid="stToolbar"] { right: 1rem; }
.block-container {
    max-width: 1500px;
    padding-top: 1.7rem;
    padding-bottom: 3rem;
}
h1, h2, h3, h4, h5, h6,
.brand, .hero h1, .section-title {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text) !important;
    letter-spacing: -.025em;
}
p, li, label, span, strong, em {
    color: inherit;
}
.stMarkdown, .stMarkdown p, .stMarkdown li,
.stMarkdown strong, .stMarkdown label,
[data-testid="stWidgetLabel"] p,
[data-testid="stCaptionContainer"] p {
    color: var(--text);
}
.stCaption, small, [data-testid="stCaptionContainer"] {
    color: var(--muted) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111A27 0%, #0C131E 100%);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] > div {
    padding-top: 1.4rem;
}
[data-testid="stSidebar"] h3 {
    color: var(--gold) !important;
    font-size: .86rem;
    letter-spacing: .09em;
}
.brand {
    color: #F2F6FC !important;
    font-size: 24px;
    line-height: 1.1;
    font-weight: 800;
    letter-spacing: -.06em;
}
.brand span { color: var(--accent) !important; }
.tagline {
    color: var(--muted) !important;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 2.1px;
    margin-top: 7px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 5px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #C2CDDB !important;
    border: 1px solid transparent;
    border-radius: 11px;
    padding: 9px 11px;
    transition: .18s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(85,214,190,.08);
    border-color: rgba(85,214,190,.18);
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(100deg, rgba(85,214,190,.16), rgba(114,167,255,.10));
    border: 1px solid rgba(85,214,190,.32);
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    color: inherit !important;
    font-weight: 600;
}

/* Main hero and section headings */
.hero {
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(circle at 90% 15%, rgba(85,214,190,.16), transparent 28%),
        linear-gradient(120deg, #172536 0%, #142031 55%, #1B2635 100%);
    border: 1px solid rgba(114,167,255,.25);
    border-radius: 24px;
    padding: 38px 40px;
    margin: 4px 0 28px;
    box-shadow: var(--shadow);
}
.hero:after {
    content: "";
    position: absolute;
    width: 270px;
    height: 270px;
    right: -85px;
    top: -145px;
    border: 1px solid rgba(85,214,190,.28);
    border-radius: 50%;
    box-shadow: 0 0 0 32px rgba(85,214,190,.045), 0 0 0 68px rgba(85,214,190,.025);
    pointer-events: none;
}
.hero-label {
    color: var(--accent) !important;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 2.3px;
}
.hero h1 {
    color: #F3F7FC !important;
    font-size: clamp(30px, 4vw, 47px);
    line-height: 1.12;
    margin: 13px 0 12px;
}
.hero p {
    color: #B7C5D6 !important;
    font-size: 15px;
    line-height: 1.8;
    max-width: 780px;
    margin-bottom: 0;
}
.section-title {
    color: #F0F5FC !important;
    font-size: 23px;
    font-weight: 700;
    margin: 14px 0 18px;
}

/* Dashboard metric cards */
.metric-card {
    position: relative;
    overflow: hidden;
    min-height: 128px;
    padding: 22px;
    background: linear-gradient(145deg, #172333, #111A27);
    border: 1px solid var(--border);
    border-radius: 17px;
    box-shadow: 0 8px 24px rgba(0,0,0,.16);
    transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease;
}
.metric-card:after {
    content: "";
    position: absolute;
    width: 90px;
    height: 90px;
    right: -35px;
    bottom: -48px;
    border: 1px solid rgba(85,214,190,.18);
    border-radius: 50%;
    box-shadow: 0 0 0 15px rgba(85,214,190,.035);
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(85,214,190,.48);
    box-shadow: 0 14px 30px rgba(0,0,0,.24);
}
.metric-label {
    color: var(--muted) !important;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.4px;
}
.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 34px;
    line-height: 1.2;
    font-weight: 700;
    color: var(--accent) !important;
    margin-top: 12px;
}

/* Agent and paper cards */
.agent-card {
    background: linear-gradient(150deg, #172333, #111A27);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 19px 16px;
    min-height: 178px;
    margin-bottom: 10px;
    box-shadow: 0 7px 20px rgba(0,0,0,.15);
    transition: .2s ease;
}
.agent-card:hover {
    transform: translateY(-3px);
    border-color: rgba(85,214,190,.45);
    box-shadow: 0 13px 27px rgba(0,0,0,.23);
}
.agent-number {
    color: var(--accent) !important;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.5px;
}
.agent-name {
    color: #EDF3FA !important;
    font-size: 15px;
    font-weight: 700;
    margin-top: 9px;
    line-height: 1.4;
}
.agent-description {
    color: #A8B7C9 !important;
    font-size: 11px;
    line-height: 1.65;
    margin-top: 8px;
}
.paper-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 12px;
}
.paper-title { color: #EDF3FA !important; font-size: 16px; font-weight: 700; }
.paper-meta { color: var(--muted) !important; font-size: 12px; margin-top: 8px; }

/* Buttons */
div.stButton > button,
div.stFormSubmitButton > button,
div.stDownloadButton > button {
    background: linear-gradient(110deg, #35BFA9 0%, #55D6BE 100%);
    color: #071411 !important;
    border: 1px solid #55D6BE;
    border-radius: 11px;
    min-height: 44px;
    font-weight: 800;
    box-shadow: 0 6px 18px rgba(85,214,190,.14);
    transition: .18s ease;
}
div.stButton > button p,
div.stFormSubmitButton > button p,
div.stDownloadButton > button p {
    color: #071411 !important;
    font-weight: 800;
}
div.stButton > button:hover,
div.stFormSubmitButton > button:hover,
div.stDownloadButton > button:hover {
    background: #78E5D1;
    border-color: #78E5D1;
    color: #06110F !important;
    transform: translateY(-2px);
    box-shadow: 0 9px 23px rgba(85,214,190,.22);
}

/* Inputs, dropdowns and sliders */
.stTextInput label, .stNumberInput label, .stSlider label,
.stSelectbox label, .stTextArea label, .stMultiSelect label,
[data-testid="stWidgetLabel"] {
    color: #D5DFEA !important;
    font-weight: 600;
}
.stTextInput input, .stNumberInput input, .stTextArea textarea,
[data-baseweb="input"] input, [data-baseweb="textarea"] textarea {
    background: #111B29 !important;
    color: #F0F5FC !important;
    -webkit-text-fill-color: #F0F5FC !important;
    border: 1px solid #35465B !important;
    border-radius: 10px !important;
    caret-color: var(--accent);
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder {
    color: #8292A7 !important;
    -webkit-text-fill-color: #8292A7 !important;
    opacity: 1;
}
[data-baseweb="select"] > div {
    background: #111B29 !important;
    color: #F0F5FC !important;
    border-color: #35465B !important;
    border-radius: 10px !important;
}
[data-baseweb="select"] span, [data-baseweb="select"] input {
    color: #F0F5FC !important;
    -webkit-text-fill-color: #F0F5FC !important;
}
[data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {
    background: #172333 !important;
    border: 1px solid #35465B !important;
}
[role="option"] {
    background: #172333 !important;
    color: #EAF0F8 !important;
}
[role="option"]:hover, [aria-selected="true"] {
    background: #26394D !important;
    color: #FFFFFF !important;
}
.stSlider [data-baseweb="slider"] { color: var(--accent); }
[data-testid="stSlider"] [role="slider"] { background: var(--accent) !important; }
[data-testid="stCheckbox"] label, [data-testid="stRadio"] label {
    color: #D5DFEA !important;
}

/* Forms, tabs, expanders and status blocks */
div[data-testid="stExpander"], div[data-testid="stForm"] {
    background: rgba(17,24,36,.92);
    border: 1px solid var(--border);
    border-radius: 15px;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: #EAF0F8 !important;
    font-weight: 600;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 7px;
    background: #111B29;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 5px;
}
.stTabs [data-baseweb="tab"] {
    color: #AEBBCD !important;
    border-radius: 9px;
}
.stTabs [data-baseweb="tab"] p { color: inherit !important; }
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    background: #203144 !important;
}
[data-testid="stAlert"] {
    background: #172333;
    color: #EAF0F8;
    border: 1px solid #35465B;
    border-radius: 12px;
}
[data-testid="stStatusWidget"] {
    background: #111B29;
    border: 1px solid var(--border);
    border-radius: 13px;
}
[data-testid="stProgress"] > div > div { background: var(--accent); }

/* Tables, code, links and separators */
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border: 1px solid var(--border);
    border-radius: 12px;
}
.stCode, pre, code {
    color: #DDE8F5 !important;
    background: #0D1520 !important;
}
a, .stMarkdown a {
    color: #7DDCCB !important;
    text-decoration-color: rgba(125,220,203,.45) !important;
}
a:hover, .stMarkdown a:hover { color: #B1F2E7 !important; }
hr, [data-testid="stDivider"] {
    border-color: var(--border) !important;
}
footer { visibility: hidden; }

/* Streamlit popovers and tooltips */
[data-testid="stPopover"] button {
    color: var(--text) !important;
    border-color: var(--border) !important;
}
[data-testid="stTooltipContent"] {
    background: #1D2A3B !important;
    color: #F0F5FC !important;
}

/* Responsive layout */
@media (max-width: 800px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .hero { padding: 27px 23px; }
    .hero h1 { font-size: 30px; }
}
</style>
""", unsafe_allow_html=True)


# ==========================================================
# 3. SESSION STATE
# ==========================================================

defaults = {
    "papers": [],
    "report": "",
    "topic": "",
    "workflow_completed": False,
    "workflow_error": ""
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==========================================================
# 4. AGENT INFORMATION
# ==========================================================

AGENTS = [
    {
        "number": "01",
        "name": "Discovery Agent",
        "description": "Discovers and organizes authentic academic research papers.",
        "icon": "🔎"
    },
    {
        "number": "02",
        "name": "Summarization Agent",
        "description": "Analyzes research objectives, methods and findings.",
        "icon": "📚"
    },
    {
        "number": "03",
        "name": "Literature Review Agent",
        "description": "Compares research themes and methodologies.",
        "icon": "📊"
    },
    {
        "number": "04",
        "name": "Research Gap Agent",
        "description": "Identifies limitations and future research opportunities.",
        "icon": "💡"
    },
    {
        "number": "05",
        "name": "Report Agent",
        "description": "Creates a structured academic research report.",
        "icon": "📄"
    }
]


def show_agents():

    columns = st.columns(5)

    for column, agent in zip(columns, AGENTS):

        with column:

            st.markdown(
                f"""
                <div class="agent-card">
                    <div class="agent-number">
                        AGENT {agent['number']}
                    </div>
                    <div style="font-size:25px;margin-top:8px">
                        {agent['icon']}
                    </div>
                    <div class="agent-name">
                        {agent['name']}
                    </div>
                    <div class="agent-description">
                        {agent['description']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ==========================================================
# 5. CREWAI RESEARCH WORKFLOW
# ==========================================================

def run_research_crew(topic, papers):

    # Use CrewAI's native OpenAI-compatible client for Groq. This strips
    # CrewAI-internal message metadata that Groq does not accept.

    llm = LLM(
        model=MODEL,
        api_key=API_KEY,
        base_url="https://api.groq.com/openai/v1",
        provider="openai",
        temperature=0.3,
        max_tokens=1200
    )

    # Keep prompts compact so the five sequential calls fit Groq's TPM limit.

    paper_information = []
    for index, paper in enumerate(papers, start=1):
        abstract = str(paper.get("abstract") or "Abstract not available.")
        if len(abstract) > 500:
            abstract = abstract[:500].rsplit(" ", 1)[0] + "..."
        paper_information.append(
            f"{index}. {paper['title']} ({paper['year']}); "
            f"{', '.join((paper.get('authors') or [])[:3])}; "
            f"DOI: {paper['doi'] or 'Not available'}; "
            f"Venue: {paper['venue']}; Source: {paper['url']}\n"
            f"Abstract: {abstract}"
        )
    paper_information = "\n\n".join(paper_information)

    # ------------------------------------------------------
    # AGENT 1: DISCOVERY
    # ------------------------------------------------------

    discovery_agent = Agent(
        role="Research Paper Discovery Specialist",

        goal="Validate and organize authentic academic research papers.",

        backstory=(
            "You are an academic research discovery specialist. "
            "You work only with supplied scholarly database records. "
            "You never invent titles, authors, DOIs or publication details."
        ),

        llm=llm,
        verbose=True,
        max_iter=1
    )

    # ------------------------------------------------------
    # AGENT 2: SUMMARIZATION
    # ------------------------------------------------------

    summarization_agent = Agent(
        role="Academic Research Summarization Expert",

        goal="Summarize research objectives, methodologies and findings.",

        backstory=(
            "You are an academic researcher skilled in understanding "
            "research objectives, methodologies, findings and limitations. "
            "You distinguish abstract-based summaries from full-text analysis."
        ),

        llm=llm,
        verbose=True,
        max_iter=1
    )

    # ------------------------------------------------------
    # AGENT 3: LITERATURE REVIEW
    # ------------------------------------------------------

    literature_agent = Agent(
        role="Literature Review Specialist",

        goal="Compare research themes, methodologies and findings.",

        backstory=(
            "You are an experienced academic literature review specialist. "
            "You identify similarities, differences and research trends "
            "using the supplied academic papers."
        ),

        llm=llm,
        verbose=True,
        max_iter=1
    )

    # ------------------------------------------------------
    # AGENT 4: RESEARCH GAP
    # ------------------------------------------------------

    gap_agent = Agent(
        role="Research Gap Analysis Expert",

        goal="Identify evidence-supported research gaps and limitations.",

        backstory=(
            "You are a research gap analysis expert. "
            "You identify limitations and future research directions. "
            "You never present unsupported assumptions as established facts."
        ),

        llm=llm,
        verbose=True,
        max_iter=1
    )

    # ------------------------------------------------------
    # AGENT 5: REPORT
    # ------------------------------------------------------

    report_agent = Agent(
        role="Academic Research Report Writer",

        goal="Prepare a structured academic research assistance report.",

        backstory=(
            "You are an academic report writer. "
            "You prepare clear, structured academic reports "
            "with accurate references and source links."
        ),

        llm=llm,
        verbose=True,
        max_iter=1
    )

    # ======================================================
    # TASK 1: DISCOVERY
    # ======================================================

    discovery_task = Task(

        description=f"""
        Review and organize the following research papers.

        Research Topic: {topic}

        Retrieved Records:
        {paper_information}

        Return:
        1. Paper title
        2. Authors
        3. Publication year
        4. DOI
        5. Publication venue
        6. Source URL

        Do not invent or modify references.
        """,

        expected_output=(
            "An organized list of actual research papers "
            "with original metadata and references."
        ),

        agent=discovery_agent
    )

    # ======================================================
    # TASK 2: SUMMARIZATION
    # ======================================================

    summarization_task = Task(

        description=f"""
        Summarize each supplied paper using its title and abstract:

        {paper_information}

        For each, provide at most three concise bullets for objective,
        method/findings, and reported limitations. Mark unavailable details.
        Treat abstracts as abstracts, not full-text evidence.
        """,

        expected_output=(
            "A concise, source-grounded summary for each supplied paper."
        ),

        agent=summarization_agent,

        context=[discovery_task]
    )

    # ======================================================
    # TASK 3: LITERATURE REVIEW
    # ======================================================

    literature_task = Task(

        description="""
        Prepare a comparative literature review.

        Compare:

        1. Research themes
        2. Methodologies
        3. Technologies and datasets, where reported
        4. Findings
        5. Similarities and differences

        Support comparisons using supplied paper references.
        """,

        expected_output="A concise comparative literature review (under 200 words).",

        agent=literature_agent,

        context=[summarization_task]
    )

    # ======================================================
    # TASK 4: RESEARCH GAP
    # ======================================================

    gap_task = Task(

        description="""
        Analyze the literature review and identify research gaps.

        Include:

        1. Reported limitations
        2. Areas needing further investigation
        3. Unresolved research problems
        4. Possible future research directions

        Link proposed gaps to the supplied research papers.

        Clearly distinguish reported evidence from suggestions.
        """,

        expected_output="Up to five evidence-linked gaps and future directions.",

        agent=gap_agent,

        context=[literature_task]
    )

    # ======================================================
    # TASK 5: FINAL REPORT
    # ======================================================

    report_task = Task(

        description=f"""
        Prepare a complete academic research assistance report.

        Research Topic: {topic}

        Write a concise report (no more than about 800 words) with these
        sections:

        1. Research Topic and Introduction
        2. List of Discovered Research Papers
        3. Individual Paper Summaries
        4. Comparative Literature Review
        5. Research Gaps and Supporting Evidence
        6. Future Research Directions
        7. References with DOI and source links

        Use only the supplied research records and task outputs.

        Do not fabricate references, quotations or findings.

        Mention when conclusions are based only on abstracts.
        """,

        expected_output="A concise, structured Markdown research report.",

        agent=report_agent,

        context=[
            summarization_task,
            literature_task,
            gap_task
        ]
    )

    # ======================================================
    # CREATE CREW
    # ======================================================

    research_crew = Crew(

        agents=[
            discovery_agent,
            summarization_agent,
            literature_agent,
            gap_agent,
            report_agent
        ],

        tasks=[
            discovery_task,
            summarization_task,
            literature_task,
            gap_task,
            report_task
        ],

        process=Process.sequential,

        max_rpm=10,

        verbose=True
    )

    return research_crew.kickoff()


# ==========================================================
# 6. SIDEBAR NAVIGATION
# ==========================================================

with st.sidebar:

    st.markdown(
        '<div class="brand">RESEARCH<span>NEXUS</span></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="tagline">AI RESEARCH INTELLIGENCE</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### WORKSPACE")

    page = st.radio(
        "Navigation",
        [
            "✦  Dashboard",
            "▤  Research Library",
            "◈  AI Agent Workspace",
            "↗  Report Studio"
        ],
        label_visibility="collapsed"
    )
    page = {
        "✦  Dashboard": "Dashboard",
        "▤  Research Library": "Research Library",
        "◈  AI Agent Workspace": "AI Agent Workspace",
        "↗  Report Studio": "Report Studio"
    }[page]

    st.divider()

    st.markdown("### AI CONFIGURATION")

    if API_KEY:
        st.success("Groq API Connected")
    else:
        st.error("Groq API Key Missing")

    st.caption(f"Model: {MODEL}")

    st.divider()

    st.markdown("### DATA SOURCE")

    st.markdown("🟢 OpenAlex + Crossref")

    st.caption("Academic research metadata")

    st.divider()

    st.caption("RESEARCH NEXUS AI")
    st.caption("CrewAI Multi-Agent Research Assistant")


# ==========================================================
# 7. APPLICATION HEADER
# ==========================================================

st.html("""
<div class="hero">

    <div class="hero-label">
        ✦ INTELLIGENT ACADEMIC RESEARCH PLATFORM
    </div>

    <h1>Research smarter.<br>Discover what matters.</h1>

    <p>
        Your AI-powered research workspace — discover authentic academic papers,
        understand findings, compare methodologies and uncover evidence-based
        research gaps with five specialized CrewAI agents.
    </p>

</div>
""")


# ==========================================================
# 8. DASHBOARD
# ==========================================================

if page == "Dashboard":

    st.markdown(
        '<div class="section-title">Research Command Center</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">PAPERS DISCOVERED</div>
                <div class="metric-value">
                    {len(st.session_state.papers)}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">SPECIALIZED AI AGENTS</div>
            <div class="metric-value">05</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        report_status = (
            "READY"
            if st.session_state.report
            else "PENDING"
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">REPORT STATUS</div>
                <div class="metric-value" style="font-size:23px">
                    {report_status}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown(
        '<div class="section-title">Start New Research</div>',
        unsafe_allow_html=True
    )

    with st.form("research_form"):

        topic = st.text_input(
            "Research Topic",
            placeholder="Example: Generative AI in Healthcare"
        )

        col1, col2 = st.columns(2)

        with col1:

            number = st.slider(
                "Number of Research Papers",
                min_value=1,
                max_value=20,
                value=5
            )

        with col2:

            start_year = st.number_input(
                "Publication Starting Year",
                min_value=1900,
                max_value=datetime.now().year,
                value=2020
            )

        submitted = st.form_submit_button(
            "🚀 Discover & Analyze Research",
            use_container_width=True
        )

    # ======================================================
    # START RESEARCH
    # ======================================================

    if submitted:

        if not topic.strip():

            st.warning("Please enter a research topic.")

        elif not API_KEY:

            st.error("GROQ_API_KEY is missing from your .env file.")

        else:

            st.session_state.topic = topic
            st.session_state.papers = []
            st.session_state.report = ""
            st.session_state.workflow_completed = False
            st.session_state.workflow_error = ""

            try:

                with st.status(
                    "Research workflow running...",
                    expanded=True
                ) as status:

                    # Step 1: OpenAlex

                    st.write("🔎 Discovering academic research papers...")

                    papers = search_research_papers(
                        topic,
                        number,
                        int(start_year)
                    )

                    if not papers:

                        raise RuntimeError(
                            f"No research papers matched '{topic}' from "
                            f"{int(start_year)} onward. Try broader search "
                            "terms or an earlier starting year."
                        )

                    st.session_state.papers = papers

                    save_papers(
                        papers,
                        str(BASE_DIR / "research_papers.txt")
                    )

                    st.write(
                        f"Retrieved {len(papers)} paper records."
                    )

                    # Step 2: CrewAI

                    st.write("🤖 Initializing five CrewAI agents...")

                    st.write("📚 Starting sequential research analysis...")

                    result = run_research_crew(topic, papers)
                    result_text = str(result).strip()

                    if not result_text or result_text == "None":
                        raise RuntimeError(
                            "The AI workflow returned an empty report. "
                            "Check the CrewAI/Groq output above and try again."
                        )

                    # Step 3: Save report

                    report_file = REPORT_DIR / "research_report.md"

                    report_content = (
                        f"# Research Report: {topic}\n\n"
                        f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n\n"
                        f"{result_text}"
                    )

                    report_file.write_text(
                        report_content,
                        encoding="utf-8"
                    )

                    st.session_state.report = report_content

                    st.session_state.workflow_completed = True

                    status.update(
                        label="Research analysis completed!",
                        state="complete",
                        expanded=False
                    )

                st.success("Your research report is ready!")

                st.rerun()

            except Exception as error:

                st.session_state.workflow_error = str(error)

                error_text = str(error)
                if "429" in error_text or "rate_limit_exceeded" in error_text:
                    st.error(
                        "Groq's token-per-minute limit was reached during "
                        "analysis. Your papers were retrieved, but the report "
                        "was not completed. Wait about one minute and retry; "
                        "selecting fewer papers can also reduce token use."
                    )
                    st.caption(f"Provider details: {error_text}")
                else:
                    st.error(f"Research workflow failed: {error_text}")

    # Show agent team

    st.markdown("---")

    st.markdown(
        '<div class="section-title">Your AI Research Team</div>',
        unsafe_allow_html=True
    )

    show_agents()

    if st.session_state.workflow_error:

        st.error(st.session_state.workflow_error)


# ==========================================================
# 9. RESEARCH LIBRARY
# ==========================================================

elif page == "Research Library":

    st.markdown(
        '<div class="section-title">Academic Research Library</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.papers:

        st.info(
            "No research papers available. Start a research project first."
        )

    else:

        st.caption(
            f"Topic: {st.session_state.topic} | "
            f"Total Papers: {len(st.session_state.papers)}"
        )

        search_text = st.text_input(
            "Search discovered papers",
            placeholder="Search by title or author..."
        )

        for index, paper in enumerate(
            st.session_state.papers,
            start=1
        ):

            searchable = (
                paper["title"] + " " +
                " ".join(paper["authors"])
            ).lower()

            if search_text.lower() not in searchable:
                continue

            with st.expander(
                f"📄 {index:02d} | {paper['title']}"
            ):

                st.markdown(
                    f"**Authors:** {', '.join(paper['authors'])}"
                )

                st.markdown(
                    f"**Publication Year:** {paper['year']}"
                )

                st.markdown(
                    f"**Publication Venue:** {paper['venue']}"
                )

                st.markdown(
                    f"**DOI:** {paper['doi'] or 'Not available'}"
                )

                st.markdown(
                    f"[Open paper source]({paper['url']})"
                )

                st.markdown("### Abstract")

                st.write(paper["abstract"])


# ==========================================================
# 10. AI AGENT WORKSPACE
# ==========================================================

elif page == "AI Agent Workspace":

    st.markdown(
        '<div class="section-title">Multi-Agent Intelligence Workspace</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Five specialized CrewAI agents collaborate to complete "
        "the academic research workflow."
    )

    show_agents()

    st.markdown("---")

    st.markdown("### Sequential Agent Workflow")

    workflow_steps = [
        "Discovery Agent – Research paper discovery",
        "Summarization Agent – Academic paper summaries",
        "Literature Review Agent – Research comparison",
        "Research Gap Agent – Gap analysis",
        "Report Agent – Final report preparation"
    ]

    for step in workflow_steps:
        st.markdown(f"✓ {step}")

    if st.session_state.workflow_completed:

        st.success("Latest research workflow completed.")

    else:

        st.info(
            "To run the agents and generate a report, select **Dashboard** "
            "in the sidebar, enter a research topic, and click "
            "**Discover & Analyze Research**."
        )


# ==========================================================
# 11. REPORT STUDIO
# ==========================================================

elif page == "Report Studio":

    st.markdown(
        '<div class="section-title">Research Report Studio</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.report:

        st.info(
            "No report generated yet. Select **Dashboard** in the sidebar, "
            "enter a research topic, and click **Discover & Analyze Research**."
        )

    else:

        st.caption(
            f"Research Topic: {st.session_state.topic}"
        )

        tab1, tab2 = st.tabs(
            ["📖 Report Preview", "⬇ Download Report"]
        )

        with tab1:

            st.markdown(st.session_state.report)

        with tab2:

            st.download_button(
                label="Download Research Report",
                data=st.session_state.report,
                file_name="Research_Nexus_Report.md",
                mime="text/markdown",
                use_container_width=True
            )

            metadata_file = BASE_DIR / "research_papers.txt"

            if metadata_file.exists():

                st.download_button(
                    label="Download Research Paper Metadata",
                    data=metadata_file.read_text(encoding="utf-8"),
                    file_name="Research_Papers.txt",
                    mime="text/plain",
                    use_container_width=True
                )


# ==========================================================
# 12. FOOTER
# ==========================================================

st.markdown("---")

st.markdown("""
<div style="
    text-align:center;
    color:#7890A8;
    font-size:12px;
    padding:15px;
">
    RESEARCH NEXUS AI
    <br>
    BUILT WITH CREWAI  ·  GROQ  ·  OPENALEX
    <br><br>
    Academic findings should be validated against original research papers.
</div>
""", unsafe_allow_html=True)
