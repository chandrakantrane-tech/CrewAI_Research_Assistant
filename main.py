
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
# 2. CUSTOM UI DESIGN - RESEARCH NEXUS DARK INTERFACE
# ==========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    color-scheme: dark;
    --bg: #080D18;
    --bg-soft: #0D1424;
    --panel: #111B2D;
    --panel-raised: #162238;
    --panel-hover: #1B2A43;
    --line: #263650;
    --line-soft: rgba(125, 155, 195, .16);
    --text: #F1F5FC;
    --text-soft: #C3CFE0;
    --muted: #91A2BA;
    --accent: #63D6C5;
    --accent-2: #7CA7FF;
    --accent-deep: #163E4B;
    --warning: #F2C879;
    --danger: #FF858D;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 82% 0%, rgba(71, 123, 190, .17), transparent 34%),
        radial-gradient(ellipse at 4% 38%, rgba(55, 190, 170, .08), transparent 32%),
        linear-gradient(145deg, #080D18 0%, #0B1220 52%, #080D18 100%);
    color: var(--text);
}

header[data-testid="stHeader"] {
    background: rgba(8, 13, 24, .78);
    backdrop-filter: blur(14px);
}
[data-testid="stToolbar"] { right: 1rem; }
[data-testid="stDecoration"] { background: var(--accent); }

.block-container {
    max-width: 1540px;
    padding: 2rem 2.4rem 3.5rem;
}

h1, h2, h3, h4, h5,
.brand, .hero h1, .section-title {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text) !important;
    letter-spacing: -.035em;
}
h1 { font-weight: 700 !important; }
h2, h3, h4 { font-weight: 650 !important; }

.stMarkdown, .stMarkdown p, .stMarkdown li,
.stMarkdown strong, .stMarkdown span,
[data-testid="stText"], [data-testid="stCaptionContainer"] {
    color: var(--text-soft);
}
.stMarkdown strong { color: var(--text) !important; }
p, li { line-height: 1.72; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101B2D 0%, #0B1322 100%);
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] > div {
    padding: 1.55rem 1rem 1.5rem;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: var(--text-soft);
}
[data-testid="stSidebar"] h3 {
    color: #AFC5E4 !important;
    font-size: .78rem !important;
    letter-spacing: .12em;
    text-transform: uppercase;
}
.brand {
    font-size: 1.55rem;
    font-weight: 800;
    color: #F4F7FD !important;
    letter-spacing: -.07em;
    white-space: nowrap;
}
.brand span { color: var(--accent) !important; }
.tagline {
    color: #8FA6C4;
    font-size: .62rem;
    letter-spacing: .19em;
    margin-top: .35rem;
    font-weight: 800;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > label {
    color: var(--muted) !important;
    font-size: .75rem;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
    gap: .35rem;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    border: 1px solid transparent;
    border-radius: 12px;
    padding: .65rem .72rem;
    margin: 0;
    color: #B7C6DB !important;
    font-weight: 650;
    transition: all .18s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(99, 214, 197, .08);
    border-color: rgba(99, 214, 197, .2);
    color: #E9FFFB !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(100deg, rgba(99, 214, 197, .16), rgba(124, 167, 255, .10));
    border: 1px solid rgba(99, 214, 197, .34);
    color: #E9FFFB !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    color: inherit !important;
}

.hero {
    position: relative;
    overflow: hidden;
    isolation: isolate;
    background:
        radial-gradient(circle at 88% 18%, rgba(99, 214, 197, .16), transparent 25%),
        linear-gradient(118deg, #14243A 0%, #142238 53%, #172A3B 100%);
    border: 1px solid rgba(115, 159, 201, .28);
    border-radius: 25px;
    padding: 2.7rem 3rem;
    margin: .2rem 0 2rem;
    box-shadow: 0 24px 65px rgba(0, 0, 0, .25);
}
.hero:before {
    content: "";
    position: absolute;
    z-index: -1;
    width: 340px;
    height: 340px;
    right: -100px;
    top: -190px;
    border: 1px solid rgba(99, 214, 197, .26);
    border-radius: 50%;
    box-shadow: 0 0 0 32px rgba(99, 214, 197, .035),
                0 0 0 72px rgba(124, 167, 255, .025);
}
.hero-label {
    color: var(--accent);
    font-size: .68rem;
    font-weight: 800;
    letter-spacing: .2em;
}
.hero h1 {
    color: #F5F8FF !important;
    font-size: clamp(2rem, 4vw, 3.15rem);
    line-height: 1.1;
    margin: 1rem 0 .8rem;
}
.hero p {
    color: #B9C9DD !important;
    font-size: 1rem;
    line-height: 1.8;
    max-width: 780px;
    margin: 0;
}

.section-title {
    color: #F0F5FD !important;
    font-size: 1.48rem;
    font-weight: 700;
    margin: 1.15rem 0 1.15rem;
}
.metric-card {
    position: relative;
    overflow: hidden;
    min-height: 138px;
    padding: 1.35rem 1.45rem;
    background: linear-gradient(145deg, rgba(23, 37, 59, .98), rgba(15, 26, 43, .98));
    border: 1px solid var(--line);
    border-radius: 18px;
    box-shadow: 0 12px 30px rgba(0, 0, 0, .16);
    transition: transform .2s ease, border-color .2s ease;
}
.metric-card:after {
    content: "";
    position: absolute;
    width: 110px;
    height: 110px;
    right: -45px;
    bottom: -60px;
    border: 1px solid rgba(99, 214, 197, .2);
    border-radius: 50%;
    box-shadow: 0 0 0 18px rgba(99, 214, 197, .025);
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(99, 214, 197, .48);
}
.metric-label {
    color: #9DB0C9;
    font-size: .67rem;
    font-weight: 800;
    letter-spacing: .13em;
}
.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.15rem;
    line-height: 1.2;
    font-weight: 700;
    color: var(--accent);
    margin-top: .75rem;
}

.agent-card {
    background: linear-gradient(155deg, #17263C, #101B2D);
    border: 1px solid var(--line);
    border-radius: 17px;
    padding: 1.15rem 1rem;
    min-height: 190px;
    margin-bottom: .65rem;
    box-shadow: 0 10px 26px rgba(0, 0, 0, .16);
    transition: transform .2s ease, border-color .2s ease;
}
.agent-card:hover {
    transform: translateY(-4px);
    border-color: rgba(99, 214, 197, .48);
}
.agent-number {
    color: var(--accent);
    font-size: .65rem;
    font-weight: 800;
    letter-spacing: .15em;
}
.agent-name {
    color: #F0F5FD;
    font-size: .94rem;
    font-weight: 750;
    margin-top: .55rem;
    line-height: 1.4;
}
.agent-description {
    color: #9EADC2;
    font-size: .75rem;
    line-height: 1.65;
    margin-top: .5rem;
}
.paper-card {
    background: linear-gradient(145deg, #17263C, #101B2D);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 1.15rem;
    margin-bottom: .8rem;
}
.paper-title { color: #F0F5FD; font-size: 1rem; font-weight: 750; }
.paper-meta { color: #9EADC2; font-size: .78rem; margin-top: .5rem; }

div.stButton > button,
div.stFormSubmitButton > button,
div.stDownloadButton > button {
    background: linear-gradient(105deg, #55CDBB 0%, #7CA7FF 100%);
    color: #07131C !important;
    font-weight: 800;
    border: 1px solid transparent;
    border-radius: 11px;
    min-height: 2.85rem;
    padding: .55rem 1.1rem;
    box-shadow: 0 8px 22px rgba(76, 184, 177, .16);
    transition: transform .18s ease, filter .18s ease;
}
div.stButton > button:hover,
div.stFormSubmitButton > button:hover,
div.stDownloadButton > button:hover {
    filter: brightness(1.08);
    color: #06111A !important;
    transform: translateY(-2px);
    border-color: rgba(255,255,255,.25);
}
div.stButton > button:focus,
div.stFormSubmitButton > button:focus,
div.stDownloadButton > button:focus {
    box-shadow: 0 0 0 .2rem rgba(99, 214, 197, .24);
}

.stTextInput input, .stNumberInput input, .stTextArea textarea,
.stDateInput input, .stTimeInput input,
.stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"] {
    background: #0D1728 !important;
    color: #F1F5FC !important;
    border: 1px solid #344761 !important;
    border-radius: 10px !important;
    caret-color: var(--accent);
}
.stTextInput input:focus, .stNumberInput input:focus,
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder {
    color: #8192AA !important;
    opacity: 1;
}
.stTextInput label, .stNumberInput label, .stSlider label,
.stSelectbox label, .stTextArea label, .stMultiSelect label,
.stDateInput label, .stTimeInput label,
[data-testid="stCheckbox"] label, [data-testid="stRadio"] > label {
    color: #D5DFED !important;
    font-weight: 650 !important;
}
[data-baseweb="select"] * { color: #F1F5FC !important; }
[data-baseweb="popover"], [data-baseweb="menu"],
[data-baseweb="list"] {
    background: #17243A !important;
    border-color: #344761 !important;
}
[role="option"] { color: #EAF0F9 !important; }
[role="option"]:hover { background: #263B59 !important; }
.stSlider [data-baseweb="slider"] { color: var(--accent); }
.stSlider [data-testid="stTickBar"] { background: #42536B; }

div[data-testid="stExpander"], div[data-testid="stForm"] {
    background: rgba(17, 27, 45, .82);
    border: 1px solid var(--line);
    border-radius: 15px;
}
[data-testid="stExpander"] summary {
    color: #E5EDF8 !important;
    font-weight: 700;
}
[data-testid="stExpander"] summary:hover { color: var(--accent) !important; }

.stTabs [data-baseweb="tab-list"] {
    gap: .45rem;
    background: #101B2D;
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: .35rem;
}
.stTabs [data-baseweb="tab"] {
    color: #AFC0D6 !important;
    border-radius: 9px;
    font-weight: 650;
}
.stTabs [aria-selected="true"] {
    color: #07131C !important;
    background: var(--accent) !important;
}
.stTabs [data-baseweb="tab-highlight"] { background: var(--accent); }

[data-testid="stAlert"] {
    background: #142238;
    border: 1px solid #344761;
    border-radius: 12px;
    color: #EAF0F9;
}
[data-testid="stAlert"] p { color: #EAF0F9 !important; }
[data-testid="stStatusWidget"] {
    background: #111D30;
    border: 1px solid var(--line);
    border-radius: 14px;
}
[data-testid="stProgress"] > div > div { background: var(--accent); }

a, .stMarkdown a {
    color: #79DCCF !important;
    font-weight: 650;
    text-decoration: none;
}
a:hover, .stMarkdown a:hover {
    color: #B1FFF3 !important;
    text-decoration: underline;
}
hr {
    border-color: rgba(130, 157, 190, .22);
    margin: 1.4rem 0;
}
small, .stCaption, [data-testid="stCaptionContainer"] {
    color: #91A2BA !important;
}
code {
    color: #A8F0E5 !important;
    background: #101C2D !important;
}
pre {
    background: #0B1423 !important;
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border: 1px solid var(--line);
    border-radius: 12px;
}
[data-testid="stFileUploader"] {
    background: #111B2D;
    border: 1px dashed #425875;
    border-radius: 13px;
}
[data-testid="stFileUploader"] * { color: #DCE6F3; }
[data-testid="stChatInput"] textarea {
    background: #0D1728 !important;
    color: #F1F5FC !important;
}
[data-testid="stMarkdownContainer"] table {
    color: #DCE6F3;
}
[data-testid="stMarkdownContainer"] th {
    background: #1B2A43;
    color: #F1F5FC;
}
[data-testid="stMarkdownContainer"] td {
    background: #111B2D;
    color: #DCE6F3;
}
button[kind="secondary"] {
    color: #EAF0F9 !important;
    background: #1A2940 !important;
    border: 1px solid #3B506D !important;
}
button[kind="secondary"]:hover {
    background: #263B59 !important;
}
footer { visibility: hidden; }

@media (max-width: 900px) {
    .block-container { padding: 1.2rem 1rem 2.5rem; }
    .hero { padding: 2rem 1.45rem; border-radius: 19px; }
    .hero h1 { font-size: 2rem; }
    .hero p { font-size: .9rem; }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        transition-duration: .01ms !important;
        animation-duration: .01ms !important;
    }
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
        abstract = paper["abstract"]
        if len(abstract) > 500:
            abstract = abstract[:500].rsplit(" ", 1)[0] + "..."
        paper_information.append(
            f"{index}. {paper['title']} ({paper['year']}); "
            f"{', '.join(paper['authors'][:3])}; "
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

st.markdown("""
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
