"""
Catalog loading, indexing, and search for SHL assessments.
Provides both keyword-based and metadata-based filtering.
"""

import json
import os
from pathlib import Path
from typing import Optional


# Resolve catalog path relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CATALOG_PATH = _PROJECT_ROOT / "data" / "catalog.json"


class Assessment:
    """Represents a single SHL assessment from the catalog."""

    def __init__(self, data: dict):
        self.name: str = data.get("name", "")
        self.url: str = data.get("url", "")
        self.description: str = data.get("description", "")
        self.test_type: str = data.get("test_type", "")
        self.category: str = data.get("category", "")
        self.duration: str = data.get("duration", "")
        self.remote_testing: bool = data.get("remote_testing", True)
        self.adaptive_testing: bool = data.get("adaptive_testing", False)
        self.keywords: list[str] = data.get("keywords", [])
        self._raw = data

    @property
    def search_text(self) -> str:
        """Combined text used for embedding and keyword search."""
        parts = [
            self.name,
            self.description,
            self.category,
            f"Test type: {self._test_type_label}",
            f"Duration: {self.duration}",
            " ".join(self.keywords),
        ]
        return " ".join(parts)

    @property
    def _test_type_label(self) -> str:
        labels = {
            "K": "Knowledge",
            "P": "Personality",
            "A": "Ability/Cognitive",
            "S": "Skills/Simulation",
            "B": "Behavioral/SJT",
        }
        return labels.get(self.test_type, self.test_type)

    @property
    def summary(self) -> str:
        """Short summary for LLM context."""
        return (
            f"{self.name} [{self.test_type}] — {self.category} | "
            f"{self.duration} | {self.description[:200]}"
        )

    def to_recommendation(self) -> dict:
        """Convert to the API recommendation format."""
        return {
            "name": self.name,
            "url": self.url,
            "test_type": self.test_type,
        }

    def matches_keywords(self, query_lower: str) -> float:
        """Score how well this assessment matches keyword queries. Returns 0-1."""
        score = 0.0
        total_checks = 0

        query_words = set(query_lower.split())

        # Check name match
        name_lower = self.name.lower()
        for word in query_words:
            total_checks += 1
            if word in name_lower:
                score += 2.0  # Name matches are weighted heavily

        # Check keyword match
        kw_lower = {k.lower() for k in self.keywords}
        for word in query_words:
            total_checks += 1
            if any(word in kw for kw in kw_lower):
                score += 1.0

        # Check description match
        desc_lower = self.description.lower()
        for word in query_words:
            total_checks += 1
            if word in desc_lower:
                score += 0.5

        return score / max(total_checks, 1)


class Catalog:
    """Manages the SHL assessment catalog."""

    def __init__(self):
        self.assessments: list[Assessment] = []
        self._loaded = False

    def load(self, path: Optional[str] = None):
        """Load catalog from JSON file."""
        catalog_path = Path(path) if path else _CATALOG_PATH

        if not catalog_path.exists():
            raise FileNotFoundError(f"Catalog not found at {catalog_path}")

        with open(catalog_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assessments = [Assessment(item) for item in data]
        self._loaded = True
        print(f"Loaded {len(self.assessments)} assessments from catalog")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_all_names(self) -> list[str]:
        """Get all assessment names."""
        return [a.name for a in self.assessments]

    def get_by_name(self, name: str) -> Optional[Assessment]:
        """Find assessment by exact or fuzzy name match."""
        name_lower = name.lower().strip()
        # Exact match
        for a in self.assessments:
            if a.name.lower() == name_lower:
                return a
        # Partial match
        for a in self.assessments:
            if name_lower in a.name.lower() or a.name.lower() in name_lower:
                return a
        return None

    def filter_by_type(self, test_type: str) -> list[Assessment]:
        """Filter assessments by test type code."""
        return [a for a in self.assessments if a.test_type == test_type.upper()]

    def filter_by_category(self, category: str) -> list[Assessment]:
        """Filter assessments by category."""
        cat_lower = category.lower()
        return [a for a in self.assessments if cat_lower in a.category.lower()]

    def keyword_search(self, query: str, top_k: int = 20) -> list[Assessment]:
        """Simple keyword-based search across all assessments."""
        query_lower = query.lower()
        scored = []
        for a in self.assessments:
            score = a.matches_keywords(query_lower)
            if score > 0:
                scored.append((score, a))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [a for _, a in scored[:top_k]]

    def get_catalog_summary(self) -> str:
        """Generate a summary of the catalog for the LLM system prompt."""
        type_counts = {}
        cat_counts = {}
        for a in self.assessments:
            type_counts[a.test_type] = type_counts.get(a.test_type, 0) + 1
            cat_counts[a.category] = cat_counts.get(a.category, 0) + 1

        type_labels = {
            "K": "Knowledge Tests",
            "P": "Personality Questionnaires",
            "A": "Ability/Cognitive Tests",
            "S": "Skills Simulations",
            "B": "Behavioral/SJT Tests",
        }

        lines = [f"Total assessments: {len(self.assessments)}"]
        lines.append("\nBy test type:")
        for code, count in sorted(type_counts.items()):
            label = type_labels.get(code, code)
            lines.append(f"  {code} = {label}: {count}")

        lines.append("\nBy category:")
        for cat, count in sorted(cat_counts.items()):
            lines.append(f"  {cat}: {count}")

        return "\n".join(lines)


# Singleton instance
catalog = Catalog()
