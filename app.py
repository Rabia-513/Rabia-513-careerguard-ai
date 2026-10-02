
import re
from urllib.parse import urlparse

import pymupdf
import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="CareerGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM UI
# =========================================================
st.markdown("""
<style>
:root {
    --cg-navy: #17324D;
    --cg-blue: #246BFD;
    --cg-teal: #1C9AB7;
    --cg-bg: #F5F8FC;
    --cg-card: #FFFFFF;
    --cg-border: #E3EAF2;
    --cg-text: #1D2A36;
    --cg-muted: #607080;
}

.stApp {
    background: var(--cg-bg);
}

.block-container {
    padding-top: 1.7rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

section[data-testid="stSidebar"] {
    background: #10273B;
    border-right: 0;
}

section[data-testid="stSidebar"] * {
    color: #F4F8FC;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: rgba(255,255,255,0.04);
    padding: 0.55rem 0.7rem;
    border-radius: 10px;
    margin-bottom: 0.2rem;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.10);
}

.cg-hero {
    background: linear-gradient(115deg, #17324D 0%, #195879 55%, #1C9AB7 100%);
    border-radius: 22px;
    padding: 2rem 2.2rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 12px 30px rgba(20, 58, 90, 0.14);
}

.cg-hero h1 {
    color: white !important;
    font-size: 2.35rem;
    margin: 0 0 .45rem 0;
}

.cg-hero p {
    color: #E8F4F8 !important;
    font-size: 1.02rem;
    max-width: 850px;
    margin: 0;
    line-height: 1.6;
}

.cg-card {
    background: white;
    border: 1px solid var(--cg-border);
    border-radius: 16px;
    padding: 1.1rem 1.15rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 18px rgba(25, 55, 85, 0.05);
}

.cg-card-title {
    color: var(--cg-navy);
    font-weight: 700;
    font-size: 1.05rem;
    margin-bottom: .35rem;
}

.cg-muted {
    color: var(--cg-muted);
    font-size: .92rem;
}

.cg-step {
    display: inline-block;
    background: #EAF2FF;
    color: #2353A5;
    padding: .28rem .6rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: .78rem;
    margin-bottom: .6rem;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--cg-border);
    padding: 0.95rem 1rem;
    border-radius: 14px;
    box-shadow: 0 4px 15px rgba(25,55,85,.05);
}

.stButton > button {
    border-radius: 10px;
    font-weight: 700;
    padding: .55rem 1rem;
}

.stTextArea textarea,
.stTextInput input {
    border-radius: 10px !important;
}

.cg-footer {
    color: #6C7C8B;
    font-size: .84rem;
    text-align: center;
    padding: 1rem 0 .3rem 0;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# DATA
# =========================================================
SKILLS = [
    "Python", "C++", "Java", "JavaScript", "Dart", "SQL",
    "Machine Learning", "Deep Learning", "Computer Vision",
    "Natural Language Processing", "NLP", "Generative AI",
    "LLM", "RAG", "TensorFlow", "Keras", "PyTorch",
    "Scikit-learn", "Pandas", "NumPy", "Matplotlib",
    "FastAPI", "Flask", "Streamlit", "LangChain",
    "FAISS", "Hugging Face", "Docker", "Git", "GitHub",
    "REST API", "AWS", "Azure", "GCP", "PostgreSQL",
    "MySQL", "MongoDB", "OpenCV", "Kubernetes"
]

SKILL_ALIASES = {
    "scikit learn": "Scikit-learn",
    "sklearn": "Scikit-learn",
    "tensorflow keras": "TensorFlow",
    "natural language processing": "NLP",
    "large language model": "LLM",
    "large language models": "LLM",
    "retrieval augmented generation": "RAG",
    "restful api": "REST API",
    "rest api": "REST API",
    "huggingface": "Hugging Face",
    "computer vision": "Computer Vision",
    "machine learning": "Machine Learning",
    "deep learning": "Deep Learning",
    "k8s": "Kubernetes",
}

RED_FLAGS = {
    "Money request": [
        "registration fee", "processing fee", "application fee",
        "pay money", "send payment", "deposit fee",
        "security deposit", "training fee"
    ],
    "Sensitive information": [
        "send otp", "share otp", "bank password", "bank login",
        "credit card", "debit card", "pin code", "password"
    ],
    "Unusual hiring process": [
        "guaranteed job", "instant hiring", "hired without interview",
        "no interview required", "telegram only", "contact on telegram"
    ],
    "Equipment scam pattern": [
        "purchase equipment", "buy equipment", "buy laptop",
        "we will send you a cheque", "check for equipment"
    ]
}

PROJECT_MAP = {
    "RAG": "Build a document-based RAG assistant using Python, Sentence Transformers, FAISS and FastAPI.",
    "LLM": "Build a small LLM-powered support or document assistant and expose it through a web API.",
    "FastAPI": "Wrap one of your ML models inside a FastAPI service with prediction and health-check endpoints.",
    "Docker": "Containerize one of your existing ML or RAG projects using Docker and document how to run it.",
    "AWS": "Deploy a small ML API or Streamlit application on AWS and document the deployment architecture.",
    "PyTorch": "Rebuild one of your TensorFlow image-classification projects in PyTorch and compare results.",
    "OpenCV": "Build a computer-vision mini project using OpenCV for detection, preprocessing or tracking.",
    "PostgreSQL": "Add PostgreSQL storage to one of your AI applications for users, jobs or analysis history.",
    "REST API": "Create a production-style REST API around an existing ML model using FastAPI.",
    "Kubernetes": "Containerize a small AI service and deploy it locally on Kubernetes or Minikube."
}


# =========================================================
# AI MODEL
# =========================================================
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    # Lazy import keeps the UI fast at startup.
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


# =========================================================
# HELPERS
# =========================================================
def extract_pdf_text(uploaded_file):
    data = uploaded_file.read()
    doc = pymupdf.open(stream=data, filetype="pdf")
    return "\n".join(page.get_text("text") for page in doc)


def normalize(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_skills(text):
    low = normalize(text)
    found = set()

    for skill in SKILLS:
        if skill.lower() in low:
            found.add(skill)

    for alias, canonical in SKILL_ALIASES.items():
        if alias in low:
            found.add(canonical)

    return sorted(found)


def extract_education(text):
    patterns = [
        ("BS Software Engineering", r"\b(bs|bsc|bachelor).*software engineering"),
        ("BS Computer Science", r"\b(bs|bsc|bachelor).*computer science"),
        ("BS Artificial Intelligence", r"\b(bs|bsc|bachelor).*artificial intelligence"),
        ("FSc", r"\bfsc\b"),
        ("Matriculation", r"\bmatric"),
    ]

    result = []
    low = text.lower()

    for label, pattern in patterns:
        if re.search(pattern, low, flags=re.I):
            result.append(label)

    return result


def extract_experience_level(text):
    low = text.lower()

    if any(x in low for x in ["fresh graduate", "entry level", "0-1 year", "0–1 year"]):
        return "Fresh Graduate / Entry Level"

    if "intern" in low or "internship" in low:
        return "Internship / Entry Level"

    return "Not clearly stated"


def extract_email(text):
    m = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    return m.group(0) if m else None


def extract_urls(text):
    return re.findall(r'https?://[^\s<>"\']+', text)


def domain_from_email(email):
    return email.split("@")[-1].lower() if email and "@" in email else None


def domain_from_url(url):
    if not url:
        return None

    try:
        netloc = urlparse(url).netloc.lower()
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:
        return None


def semantic_skill_match(candidate_skills, job_skills, threshold=0.72):
    from sentence_transformers import util

    matched = {}
    missing = []

    if not job_skills:
        return matched, missing

    if not candidate_skills:
        return matched, list(job_skills)

    model = load_embedding_model()

    cand_emb = model.encode(candidate_skills, convert_to_tensor=True)

    for job_skill in job_skills:
        job_emb = model.encode(job_skill, convert_to_tensor=True)
        scores = util.cos_sim(job_emb, cand_emb)[0]

        best_idx = int(np.argmax(scores.cpu().numpy()))
        best_score = float(scores[best_idx])
        best_skill = candidate_skills[best_idx]

        if best_score >= threshold:
            matched[job_skill] = {
                "candidate_skill": best_skill,
                "similarity": round(best_score, 2)
            }
        else:
            missing.append(job_skill)

    return matched, missing


def job_requirement_categories(job_text):
    lines = [x.strip() for x in job_text.splitlines() if x.strip()]
    mandatory, preferred = [], []

    for line in lines:
        low = line.lower()
        skills = extract_skills(line)

        if not skills:
            continue

        if any(w in low for w in ["preferred", "plus", "nice to have", "bonus", "familiarity"]):
            preferred.extend(skills)
        else:
            mandatory.extend(skills)

    all_skills = extract_skills(job_text)
    categorized = set(mandatory + preferred)
    mandatory.extend([s for s in all_skills if s not in categorized])

    return sorted(set(mandatory)), sorted(set(preferred))


def recruiter_risk_analysis(message, claimed_company_domain=None):
    message_low = message.lower()
    findings = []

    for category, phrases in RED_FLAGS.items():
        hits = [p for p in phrases if p in message_low]

        if hits:
            findings.append({
                "type": category,
                "severity": "high",
                "detail": ", ".join(hits)
            })

    email = extract_email(message)
    sender_domain = domain_from_email(email)

    urls = extract_urls(message)
    url_domains = [domain_from_url(u) for u in urls if domain_from_url(u)]

    free_domains = {
        "gmail.com", "yahoo.com", "outlook.com",
        "hotmail.com", "proton.me"
    }

    if sender_domain in free_domains:
        findings.append({
            "type": "Sender domain",
            "severity": "medium",
            "detail": f"Recruiter is using a free email domain: {sender_domain}"
        })

    if claimed_company_domain and sender_domain:
        claimed_company_domain = claimed_company_domain.lower().replace("www.", "")

        if sender_domain != claimed_company_domain:
            findings.append({
                "type": "Domain mismatch",
                "severity": "medium",
                "detail": (
                    f"Sender domain ({sender_domain}) does not match "
                    f"claimed company domain ({claimed_company_domain})."
                )
            })

    for d in url_domains:
        if re.search(r"\d+\.\d+\.\d+\.\d+", d):
            findings.append({
                "type": "Suspicious link",
                "severity": "high",
                "detail": f"Link uses a raw IP address: {d}"
            })

    high = sum(1 for f in findings if f["severity"] == "high")
    medium = sum(1 for f in findings if f["severity"] == "medium")

    score = max(0, 100 - high * 30 - medium * 12)

    if score >= 80:
        label = "Low observed risk"
    elif score >= 55:
        label = "Needs verification"
    else:
        label = "High observed risk"

    return {
        "score": score,
        "label": label,
        "email": email,
        "sender_domain": sender_domain,
        "urls": urls,
        "findings": findings
    }


def inspect_url(url):
    result = {
        "url": url,
        "domain": domain_from_url(url),
        "https": url.startswith("https://"),
        "reachable": False,
        "title": None,
        "final_url": None
    }

    try:
        r = requests.get(
            url,
            timeout=8,
            allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        result["reachable"] = r.ok
        result["final_url"] = r.url

        soup = BeautifulSoup(r.text[:300000], "html.parser")
        result["title"] = (
            soup.title.string.strip()
            if soup.title and soup.title.string
            else None
        )
    except Exception:
        pass

    return result


def project_recommendations(missing_skills):
    recs = []

    for skill in missing_skills:
        if skill in PROJECT_MAP:
            recs.append((skill, PROJECT_MAP[skill]))

    if not recs and missing_skills:
        joined = ", ".join(missing_skills[:4])
        recs.append((
            "Custom",
            f"Build one small project that demonstrates these missing requirements together: {joined}."
        ))

    return recs[:3]


def cv_optimization(candidate_skills, job_skills, cv_text):
    suggestions = []
    missing_from_cv = [s for s in job_skills if s not in candidate_skills]

    if len(candidate_skills) < 5:
        suggestions.append(
            "Make the technical-skills section clearer and group related AI/ML tools together."
        )

    if "github" not in cv_text.lower():
        suggestions.append(
            "Add a GitHub link so recruiters can verify your technical projects."
        )

    if "project" not in cv_text.lower():
        suggestions.append(
            "Add a dedicated Projects section with technologies, task, and measurable result."
        )

    if missing_from_cv:
        suggestions.append(
            "Do not add missing skills as keywords unless you have actually used them. "
            "Instead, build a project first and then add truthful evidence to the CV."
        )

    suggestions.append(
        "For each relevant skill, show evidence through a project, internship, "
        "certification, or measurable result."
    )

    return suggestions


# =========================================================
# SESSION STATE
# =========================================================
if "cv_text" not in st.session_state:
    st.session_state.cv_text = ""

if "candidate_skills" not in st.session_state:
    st.session_state.candidate_skills = []


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## 🛡️ CareerGuard AI")
st.sidebar.caption("AI job intelligence for early-career candidates")

page = st.sidebar.radio(
    "Navigate",
    [
        "1. Candidate Profile",
        "2. Job Match",
        "3. Recruiter Trust",
        "4. CV Optimization"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### How it works")
st.sidebar.markdown(
    "1. Upload CV  \n"
    "2. Analyze a job  \n"
    "3. Review skill gaps  \n"
    "4. Check recruiter signals  \n"
    "5. Improve your CV"
)

st.sidebar.markdown("---")
st.sidebar.caption("MVP • No generative LLM required")


# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="cg-hero">
    <h1>🛡️ CareerGuard AI</h1>
    <p>
        Turn your CV into a structured candidate profile, compare it with real job opportunities,
        identify missing skills, get project recommendations, and inspect recruiter messages for risk signals.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# PAGE 1
# =========================================================
if page == "1. Candidate Profile":
    st.markdown('<span class="cg-step">STEP 1</span>', unsafe_allow_html=True)
    st.header("Candidate Profile")

    st.markdown("""
    <div class="cg-card">
        <div class="cg-card-title">What this does</div>
        <div class="cg-muted">
            Upload a PDF CV. CareerGuard extracts its text, detects technical skills,
            education and experience level, then stores the profile for the other modules.
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CV (PDF)", type=["pdf"])

    if uploaded:
        with st.spinner("Reading CV and building candidate profile..."):
            cv_text = extract_pdf_text(uploaded)
            st.session_state.cv_text = cv_text
            st.session_state.candidate_skills = extract_skills(cv_text)

            education = extract_education(cv_text)
            level = extract_experience_level(cv_text)

        c1, c2, c3 = st.columns(3)
        c1.metric("Skills detected", len(st.session_state.candidate_skills))
        c2.metric("Education items", len(education))
        c3.metric("Experience level", level)

        st.subheader("Detected Skills")

        if st.session_state.candidate_skills:
            cols = st.columns(3)
            for idx, skill in enumerate(st.session_state.candidate_skills):
                cols[idx % 3].success(skill)
        else:
            st.warning("No supported skills were detected.")

        st.subheader("Education")
        st.write(", ".join(education) or "No supported education pattern detected.")

        with st.expander("View extracted CV text"):
            st.text(cv_text)

        st.success("Candidate profile is ready. Continue to Job Match.")


# =========================================================
# PAGE 2
# =========================================================
elif page == "2. Job Match":
    st.markdown('<span class="cg-step">STEP 2</span>', unsafe_allow_html=True)
    st.header("Job Match & Gap Analysis")

    st.markdown("""
    <div class="cg-card">
        <div class="cg-card-title">AI-powered semantic matching</div>
        <div class="cg-muted">
            CareerGuard uses the open-source all-MiniLM-L6-v2 Sentence Transformer model
            to compare job skills with skills detected from the candidate CV.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.cv_text:
        st.warning("Upload your CV on the Candidate Profile page first.")

    job_text = st.text_area(
        "Paste job description / LinkedIn post",
        height=300,
        placeholder="Paste the full job description here..."
    )

    with st.expander("Example job description"):
        st.code(
            """Junior AI Engineer

Requirements:
- Python
- Machine Learning
- Deep Learning
- RAG
- FastAPI
- Docker
- 0-2 years of experience
- BS in Software Engineering, Computer Science or AI

Docker is preferred."""
        )

    if st.button("Analyze Job", type="primary", use_container_width=True):
        if not st.session_state.cv_text:
            st.error("Please upload your CV first.")

        elif not job_text.strip():
            st.error("Please paste a job description.")

        else:
            mandatory, preferred = job_requirement_categories(job_text)
            job_skills = sorted(set(mandatory + preferred))

            with st.spinner("Running semantic skill matching..."):
                matched, missing = semantic_skill_match(
                    st.session_state.candidate_skills,
                    job_skills
                )

            match_score = (
                round((len(matched) / len(job_skills)) * 100)
                if job_skills else 0
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Role Match", f"{match_score}%")
            m2.metric("Matched Skills", len(matched))
            m3.metric("Missing Skills", len(missing))

            c1, c2 = st.columns(2)

            with c1:
                st.subheader("Matched Requirements")

                if matched:
                    for req, info in matched.items():
                        st.success(
                            f"{req} → {info['candidate_skill']} "
                            f"({info['similarity']:.2f} similarity)"
                        )
                else:
                    st.write("No matches found.")

            with c2:
                st.subheader("Missing Requirements")

                if missing:
                    for skill in missing:
                        st.error(skill)
                else:
                    st.success("No detected skill gaps.")

            st.subheader("Requirement Categories")
            cc1, cc2 = st.columns(2)

            with cc1:
                st.markdown("**Mandatory / Core**")
                st.write(", ".join(mandatory) or "None detected")

            with cc2:
                st.markdown("**Preferred / Optional**")
                st.write(", ".join(preferred) or "None detected")

            st.subheader("Recommended Projects")
            recs = project_recommendations(missing)

            if recs:
                for title, rec in recs:
                    st.info(f"**{title}:** {rec}")
            else:
                st.success(
                    "No project recommendation is needed from the detected skill gaps."
                )


# =========================================================
# PAGE 3
# =========================================================
elif page == "3. Recruiter Trust":
    st.markdown('<span class="cg-step">STEP 3</span>', unsafe_allow_html=True)
    st.header("Recruiter & Opportunity Trust Check")

    st.markdown("""
    <div class="cg-card">
        <div class="cg-card-title">Explainable recruiter-risk analysis</div>
        <div class="cg-muted">
            This module checks observable warning signals such as money requests,
            OTP/password requests, free email domains, domain mismatch and suspicious links.
            It does not claim with certainty that an opportunity is fraudulent or legitimate.
        </div>
    </div>
    """, unsafe_allow_html=True)

    recruiter_message = st.text_area(
        "Paste recruiter email/message",
        height=250,
        placeholder="Paste the recruiter email or message..."
    )

    c1, c2 = st.columns(2)

    with c1:
        claimed_domain = st.text_input(
            "Official company domain (optional)",
            placeholder="example.com"
        )

    with c2:
        job_link = st.text_input(
            "Job/company link (optional)",
            placeholder="https://example.com/careers/job"
        )

    with st.expander("Example suspicious recruiter message"):
        st.code(
            """From: hiring.company@gmail.com

Congratulations. You have been selected for the AI Engineer position without an interview.
Please pay a registration fee of Rs. 8,000 today.
Contact our HR manager only through Telegram.
Send your bank details and OTP for verification.
You must also purchase a laptop from our approved vendor."""
        )
        st.caption("Example official company domain: examplecompany.com")

    if st.button("Check Opportunity", type="primary", use_container_width=True):
        if not recruiter_message.strip():
            st.error("Please paste a recruiter message first.")
        else:
            report = recruiter_risk_analysis(
                recruiter_message,
                claimed_company_domain=claimed_domain or None
            )

            c1, c2 = st.columns(2)
            c1.metric("Trust Signal Score", f"{report['score']}/100")
            c2.metric("Assessment", report["label"])

            st.markdown("### Sender details")
            st.write("**Detected sender:**", report["email"] or "Not found")
            st.write("**Sender domain:**", report["sender_domain"] or "Not found")

            if report["findings"]:
                st.subheader("Warning Signals")

                for f in report["findings"]:
                    if f["severity"] == "high":
                        st.error(f"**{f['type']}** — {f['detail']}")
                    else:
                        st.warning(f"**{f['type']}** — {f['detail']}")
            else:
                st.success("No obvious warning signals detected in the supplied message.")

            if job_link:
                st.subheader("Link Inspection")
                info = inspect_url(job_link)

                d1, d2, d3 = st.columns(3)
                d1.metric("HTTPS", "Yes" if info["https"] else "No")
                d2.metric("Reachable", "Yes" if info["reachable"] else "Could not verify")
                d3.metric("Domain", info["domain"] or "Unknown")

                st.write("**Final URL:**", info["final_url"] or "Not available")
                st.write("**Page title:**", info["title"] or "Not available")


# =========================================================
# PAGE 4
# =========================================================
elif page == "4. CV Optimization":
    st.markdown('<span class="cg-step">STEP 4</span>', unsafe_allow_html=True)
    st.header("CV Optimization")

    st.markdown("""
    <div class="cg-card">
        <div class="cg-card-title">Truthful CV improvement</div>
        <div class="cg-muted">
            CareerGuard compares the target job with your detected CV skills and recommends
            presentation improvements without encouraging you to claim skills you have not demonstrated.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.cv_text:
        st.warning("Upload your CV first.")

    else:
        job_text = st.text_area(
            "Paste a target job description",
            height=260,
            placeholder="Paste the role you want to optimize your CV for..."
        )

        if st.button(
            "Generate CV Suggestions",
            type="primary",
            use_container_width=True
        ):
            if not job_text.strip():
                st.error("Please paste a target job description.")
            else:
                job_skills = extract_skills(job_text)

                suggestions = cv_optimization(
                    st.session_state.candidate_skills,
                    job_skills,
                    st.session_state.cv_text
                )

                st.subheader("Truthful Optimization Suggestions")

                for s in suggestions:
                    st.info(s)

                currently_missing = [
                    s for s in job_skills
                    if s not in st.session_state.candidate_skills
                ]

                if currently_missing:
                    st.subheader("Do not add these yet without evidence")
                    st.write(", ".join(currently_missing))
                    st.caption(
                        "Build a project or gain real experience first, "
                        "then add them to your CV with evidence."
                    )


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    """
    <div class="cg-footer">
        CareerGuard AI MVP • Semantic job matching • Skill-gap analysis •
        Recruiter-risk signals • Truthful CV optimization
    </div>
    """,
    unsafe_allow_html=True
)
