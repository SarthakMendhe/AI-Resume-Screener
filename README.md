# AI Resume Screening System

A 100% client-side web app that ranks multiple resumes against a job description using **NLP + TF-IDF cosine similarity**. Runs entirely in your browser — no backend, no API keys, no installs.

**Live on GitHub Pages** — just push these files to a repo and enable Pages.

## ✨ Features
- Upload multiple resumes (PDF, DOCX, TXT)
- In-browser text extraction (pdf.js + mammoth.js)
- NLP cleaning: lowercasing, stopword removal, tokenization, bigrams
- TF-IDF + cosine similarity scoring
- Ranked results with match %, progress bar, gold/silver/bronze podium
- Matched vs missing keyword chips
- Auto-extracts email + phone
- Modern dark UI, fully responsive
- **No server. No Python. No dependencies to install.**

## 🚀 Deploy to GitHub Pages (step by step)

1. Create a new repo on GitHub (e.g. `ai-resume-screener`).
2. Upload **all four files** to the root of the repo:
   - `index.html`
   - `style.css`
   - `script.js`
   - `README.md`
3. Go to **Settings → Pages**.
4. Under **Source**, choose **Deploy from a branch**.
5. Select branch `main` (or `master`) and folder `/ (root)`.
6. Click **Save**. Wait ~1 minute.
7. Your site will be live at `https://<your-username>.github.io/<repo-name>/`.

## 🖥️ Run locally
Just open `index.html` in your browser — that's it.
(For PDF parsing to work via `file://`, some browsers may require serving locally. If so: `python3 -m http.server` then open `http://localhost:8000`.)

## 🧠 How it works
1. Each resume is parsed in-browser to plain text.
2. Job description + resumes are tokenized, stopwords removed, bigrams added.
3. TF-IDF vectors are built across all documents.
4. Cosine similarity between the JD vector and each resume vector → match score.
5. Results are ranked and rendered.

## 📄 License
MIT
