"""
Fetches open-access remote sensing / precision agriculture papers from arXiv.

Why arXiv instead of MDPI: MDPI's open-access policy allows reuse of content,
but they don't expose a documented, stable bulk-search API the way arXiv does
(https://info.arxiv.org/help/api/index.html). Under a tight deadline, a
well-documented API beats a "prestigious" source with an unreliable fetch
path. arXiv papers in eess.IV / cs.CV cover a large amount of remote sensing
and precision-agriculture work (crop classification, land-cover mapping,
UAV/satellite imagery analysis).

Usage:
    python -m ingestion.fetch_papers --max-results 80

This script needs real internet access - it will NOT work in a sandboxed
environment without outbound access to export.arxiv.org.
"""
import argparse
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import requests

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

# arXiv doesn't have a dedicated "remote sensing" category, so we search
# across the categories where this work actually gets posted, filtered by
# keywords in the abstract.
CATEGORIES = ["eess.IV", "cs.CV"]
KEYWORDS = [
    "remote sensing",
    "precision agriculture",
    "satellite imagery",
    "crop monitoring",
    "UAV imagery",
    "land cover classification",
]


@dataclass
class PaperMeta:
    arxiv_id: str
    title: str
    summary: str
    authors: list[str]
    published: str
    pdf_url: str


def build_query(categories: list[str], keywords: list[str]) -> str:
    cat_part = " OR ".join(f"cat:{c}" for c in categories)
    kw_part = " OR ".join(f'abs:"{k}"' for k in keywords)
    return f"({cat_part}) AND ({kw_part})"


def parse_atom_feed(xml_text: str) -> list[PaperMeta]:
    """Parse an arXiv Atom API response into a list of PaperMeta.
    Kept as a pure function (string in, objects out) so it's testable
    without hitting the network."""
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall("atom:entry", ATOM_NS):
        raw_id = entry.findtext("atom:id", default="", namespaces=ATOM_NS)
        # raw_id looks like http://arxiv.org/abs/2401.01234v1
        arxiv_id = raw_id.rsplit("/", 1)[-1]
        title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip().replace("\n", " ")
        summary = (entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or "").strip().replace("\n", " ")
        published = entry.findtext("atom:published", default="", namespaces=ATOM_NS) or ""
        authors = [
            a.findtext("atom:name", default="", namespaces=ATOM_NS)
            for a in entry.findall("atom:author", ATOM_NS)
        ]
        pdf_url = ""
        for link in entry.findall("atom:link", ATOM_NS):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
        if not pdf_url:
            # fall back to the predictable PDF URL pattern
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        papers.append(
            PaperMeta(
                arxiv_id=arxiv_id,
                title=title,
                summary=summary,
                authors=authors,
                published=published,
                pdf_url=pdf_url,
            )
        )
    return papers


def search_arxiv(max_results: int = 80, categories=None, keywords=None) -> list[PaperMeta]:
    categories = categories or CATEGORIES
    keywords = keywords or KEYWORDS
    query = build_query(categories, keywords)

    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    resp = requests.get(ARXIV_API_URL, params=params, timeout=30)
    resp.raise_for_status()
    return parse_atom_feed(resp.text)


def download_pdf(paper: PaperMeta, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{paper.arxiv_id}.pdf"
    if dest_path.exists():
        return dest_path
    resp = requests.get(paper.pdf_url, timeout=60)
    resp.raise_for_status()
    dest_path.write_bytes(resp.content)
    return dest_path


def main():
    parser = argparse.ArgumentParser(description="Fetch remote sensing / precision ag papers from arXiv")
    parser.add_argument("--max-results", type=int, default=80)
    parser.add_argument("--out-dir", type=str, default="data/raw_pdfs")
    parser.add_argument("--sleep", type=float, default=1.0, help="seconds between downloads (be polite to arXiv)")
    args = parser.parse_args()

    print(f"Searching arXiv (categories={CATEGORIES}, keywords={KEYWORDS})...")
    papers = search_arxiv(max_results=args.max_results)
    print(f"Found {len(papers)} papers.")

    out_dir = Path(args.out_dir)
    manifest = []
    for i, paper in enumerate(papers, 1):
        print(f"[{i}/{len(papers)}] {paper.arxiv_id}: {paper.title[:70]}")
        try:
            path = download_pdf(paper, out_dir)
            manifest.append({**paper.__dict__, "local_path": str(path)})
        except Exception as e:
            print(f"  FAILED: {e}")
        time.sleep(args.sleep)

    import json

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone. {len(manifest)} papers saved to {out_dir}/, manifest.json written.")


if __name__ == "__main__":
    main()
