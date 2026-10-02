# CareerGuard AI

CareerGuard AI is an AI-powered job intelligence MVP designed for fresh graduates and entry-level candidates.

## What it does

CareerGuard helps a candidate:

- upload a PDF CV and build a structured profile
- compare CV skills with a job description
- identify matched and missing requirements
- receive project recommendations for missing skills
- inspect recruiter emails for observable risk signals
- improve a CV truthfully for a target role

## Where AI is used

The MVP uses the open-source `all-MiniLM-L6-v2` model from Sentence Transformers.

It converts candidate skills and job requirements into numerical embeddings and compares them using cosine similarity. This lets CareerGuard perform semantic matching rather than relying only on exact keyword matches.

The recruiter-risk and CV-optimization modules are intentionally rule-based so their outputs are explainable.

The MVP does **not** require a generative LLM such as Llama or Mistral.

## Features

### 1. Candidate Profile
Upload a PDF CV.

CareerGuard extracts:
- technical skills
- education
- experience level

### 2. Job Match & Gap Analysis
Paste a job description.

CareerGuard:
- detects technical requirements
- separates core and preferred skills
- uses semantic similarity to compare job requirements with CV skills
- calculates a role-match score
- identifies missing requirements

### 3. Project Recommendations
Missing requirements are mapped to practical project ideas.

Example:

Missing:
- RAG
- FastAPI
- Docker

Possible recommendation:
Build a document-based RAG assistant using FastAPI, FAISS, Sentence Transformers and Docker.

### 4. Recruiter Trust Analysis
Paste a recruiter message.

CareerGuard checks:
- free email domains
- company-domain mismatch
- requests for money
- requests for OTP/password/bank details
- unusual hiring language
- raw-IP links
- equipment-purchase patterns

CareerGuard reports risk signals. It does not guarantee that an opportunity is legitimate or fraudulent.

### 5. CV Optimization
Paste a target job description.

CareerGuard suggests truthful CV improvements and highlights skills that should not be added until the candidate has real evidence for them.

## Quick Test

### Job test

Paste:

```text
Junior AI Engineer

Requirements:
- Python
- Machine Learning
- Deep Learning
- RAG
- FastAPI
- Docker
- 0-2 years experience
- BS in Software Engineering, Computer Science or AI

Docker is preferred.
```

Expected:
- matching skills are shown from the uploaded CV
- missing skills are identified
- a match score is calculated
- project recommendations are generated

### Suspicious recruiter test

Paste:

```text
From: hiring.company@gmail.com

Congratulations. You have been selected for the AI Engineer position without an interview.
Please pay a registration fee of Rs. 8,000 today.
Contact our HR manager only through Telegram.
Send your bank details and OTP for verification.
You must also purchase a laptop from our approved vendor.
```

Official company domain:

```text
examplecompany.com
```

Expected:
- free email domain warning
- domain mismatch warning
- money request warning
- sensitive-information warning
- unusual hiring-process warning
- equipment-purchase warning

## Run locally

Create a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python -m streamlit run app.py
```

## Technology

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- all-MiniLM-L6-v2
- NumPy
- Requests
- BeautifulSoup

## Limitation

This is an MVP. Candidate skill extraction uses a controlled vocabulary and the recruiter module identifies warning signals rather than proving whether a recruiter or company is legitimate.
