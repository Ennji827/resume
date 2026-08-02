# ResumeIntel+ 🧠📄

A resume parsing and job-matching system that compares a resume against a job
description and produces a match score with a matched/missing skills
breakdown — the kind of tool recruiters and ATS systems use, built to
understand how that scoring actually works under the hood.

## Problem

Job seekers rarely know *why* their resume did or didn't get a callback.
ResumeIntel+ makes that visible: it extracts a skills profile from a resume,
compares it against a target job description, and reports:

- An overall match score (0–100%)
- Text similarity between the two documents (TF-IDF + cosine similarity)
- Explicit skill coverage (which required skills are present vs. missing)
- Skills on the resume that aren't mentioned in the job post

## How it works

1. **Text extraction** — pulls raw text from a PDF (via `pypdf`) or plain
   text resume.
2. **Skill extraction** — scans both documents against a skills taxonomy
   using word-boundary regex matching (avoids false positives like matching
   "r" inside "reporting").
3. **Similarity scoring** — vectorizes both documents with TF-IDF
   (`scikit-learn`) and computes cosine similarity to capture overall
   semantic overlap, not just keyword matches.
4. **Blended score** — combines semantic similarity (60%) with explicit
   skill coverage (40%) into a single overall match percentage.

## Example output

```
=== ResumeIntel+ Match Report ===
Overall Match Score : 58.1%
Text Similarity     : 49.2%
Skill Coverage      : 71.4%

Matched skills (10):
  data analysis, earth observation, geospatial, gis, google earth engine,
  python, qgis, remote sensing, reporting, satellite imagery

Missing skills (4) — consider adding if genuine:
  communication, data visualization, sql, statistics

Extra skills on resume not mentioned in job post (2):
  machine learning, petroleum geoscience
```

## Usage

```bash
pip install -r requirements.txt

python resume_intel.py --resume sample_data/sample_resume.txt \
                        --job sample_data/job_description.txt

# Works with PDF resumes too:
python resume_intel.py --resume my_resume.pdf --job job_post.txt
```

## Project structure

```
resume-intel/
├── resume_intel.py        # Core parsing + matching logic
├── requirements.txt
├── sample_data/
│   ├── sample_resume.txt
│   └── job_description.txt
└── README.md
```

## Design notes / limitations

- The skills taxonomy in `SKILLS_TAXONOMY` is a starter list — real-world use
  would extend this significantly, or swap in a proper NER model (e.g. spaCy)
  trained on labeled resume data.
- TF-IDF + cosine similarity captures word overlap well but misses deeper
  semantic matches (e.g. "led a team" vs. "leadership") — a natural next step
  is swapping in sentence embeddings (e.g. `sentence-transformers`) for the
  similarity component.
- Currently single resume vs. single job description; a natural extension is
  batch-ranking multiple resumes against one job posting for recruiter use
  cases.

## Author

Erick Ng'ang'a Njihia — Petroleum Geoscience & Geospatial (GIS/Remote Sensing)
professional, building practical data science / ML skills alongside
fieldwork experience.
