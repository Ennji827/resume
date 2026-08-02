"""
ResumeIntel+ : Resume Parsing and Job Matching System
Author: Erick Ng'ang'a Njihia

Parses a resume (PDF or plain text), extracts a skills profile,
compares it against a job description, and produces a match score
with matched/missing skill breakdown.

Usage:
    python resume_intel.py --resume resume.pdf --job job_description.txt
"""

import argparse
import re
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


# ---------------------------------------------------------------------------
# A starter skills taxonomy. Extend this list to widen detection coverage.
# ---------------------------------------------------------------------------
SKILLS_TAXONOMY = [
    "python", "sql", "r", "java", "javascript", "c++",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "data analysis", "data visualization", "statistics",
    "gis", "remote sensing", "qgis", "google earth engine", "arcgis",
    "geospatial", "satellite imagery", "earth observation",
    "excel", "power bi", "tableau", "streamlit",
    "git", "github", "docker", "aws", "azure", "gcp",
    "project management", "hse", "drilling operations", "petroleum geoscience",
    "leadership", "communication", "reporting",
]


def extract_text_from_pdf(path: str) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf is not installed. Run: pip install pypdf")
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_document(path: str) -> str:
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        return extract_text_from_pdf(str(p))
    return p.read_text(encoding="utf-8", errors="ignore")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_skills(text: str, taxonomy=SKILLS_TAXONOMY) -> set:
    """Return the subset of the taxonomy found in the given text."""
    norm = normalize(text)
    found = set()
    for skill in taxonomy:
        # word-boundary match so 'r' doesn't match inside 'reporting', etc.
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, norm):
            found.add(skill)
    return found


def compute_similarity(resume_text: str, job_text: str) -> float:
    """TF-IDF cosine similarity between resume and job description (0-1)."""
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([resume_text, job_text])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return float(score)


def match_report(resume_path: str, job_path: str) -> dict:
    resume_text = load_document(resume_path)
    job_text = load_document(job_path)

    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_text)

    matched = sorted(resume_skills & job_skills)
    missing = sorted(job_skills - resume_skills)
    extra = sorted(resume_skills - job_skills)

    similarity = compute_similarity(resume_text, job_text)
    skill_coverage = len(matched) / len(job_skills) if job_skills else 0.0

    # Blend semantic similarity (60%) with explicit skill coverage (40%)
    overall_score = round((0.6 * similarity + 0.4 * skill_coverage) * 100, 1)

    return {
        "overall_score": overall_score,
        "text_similarity_pct": round(similarity * 100, 1),
        "skill_coverage_pct": round(skill_coverage * 100, 1),
        "matched_skills": matched,
        "missing_skills": missing,
        "extra_skills": extra,
    }


def print_report(report: dict) -> None:
    print("\n=== ResumeIntel+ Match Report ===")
    print(f"Overall Match Score : {report['overall_score']}%")
    print(f"Text Similarity     : {report['text_similarity_pct']}%")
    print(f"Skill Coverage      : {report['skill_coverage_pct']}%")
    print(f"\nMatched skills ({len(report['matched_skills'])}):")
    print("  " + (", ".join(report["matched_skills"]) or "none"))
    print(f"\nMissing skills ({len(report['missing_skills'])}) — consider adding if genuine:")
    print("  " + (", ".join(report["missing_skills"]) or "none"))
    print(f"\nExtra skills on resume not mentioned in job post ({len(report['extra_skills'])}):")
    print("  " + (", ".join(report["extra_skills"]) or "none"))
    print()


def main():
    parser = argparse.ArgumentParser(description="Resume vs job description matcher")
    parser.add_argument("--resume", required=True, help="Path to resume (.pdf or .txt)")
    parser.add_argument("--job", required=True, help="Path to job description (.txt)")
    args = parser.parse_args()

    report = match_report(args.resume, args.job)
    print_report(report)


if __name__ == "__main__":
    main()
