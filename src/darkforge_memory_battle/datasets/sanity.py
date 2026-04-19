"""Track 0 — pipeline sanity set.

Ten hand-authored QA pairs used to validate the harness plumbing before any
real budget is spent on LongMemEval or Dark Forge. Deliberately boring.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QaPair:
    id: str
    question: str
    answer: str
    # One or more supporting passages that should be retrievable.
    passages: list[str]


SANITY_SET: list[QaPair] = [
    QaPair(
        id="s1",
        question="Who discovered penicillin?",
        answer="Alexander Fleming",
        passages=["Alexander Fleming discovered penicillin in 1928 at St. Mary's Hospital in London."],
    ),
    QaPair(
        id="s2",
        question="What is the capital of Australia?",
        answer="Canberra",
        passages=["Canberra is the capital city of Australia. It is not Sydney, despite Sydney's size."],
    ),
    QaPair(
        id="s3",
        question="What gas do plants use for photosynthesis?",
        answer="Carbon dioxide",
        passages=["Plants absorb carbon dioxide from the air and use it during photosynthesis to produce glucose."],
    ),
    QaPair(
        id="s4",
        question="Who wrote 'Pride and Prejudice'?",
        answer="Jane Austen",
        passages=["Jane Austen published Pride and Prejudice in 1813. It is one of her most enduring novels."],
    ),
    QaPair(
        id="s5",
        question="What is the chemical symbol for gold?",
        answer="Au",
        passages=["Gold has the chemical symbol Au, from the Latin 'aurum'. Its atomic number is 79."],
    ),
    QaPair(
        id="s6",
        question="Which planet has the Great Red Spot?",
        answer="Jupiter",
        passages=["Jupiter's Great Red Spot is a persistent anticyclonic storm larger than Earth."],
    ),
    QaPair(
        id="s7",
        question="What year did the Berlin Wall fall?",
        answer="1989",
        passages=["The Berlin Wall fell on November 9, 1989, ending nearly three decades of separation."],
    ),
    QaPair(
        id="s8",
        question="Who painted 'Starry Night'?",
        answer="Vincent van Gogh",
        passages=["Vincent van Gogh painted The Starry Night in June 1889 while at an asylum in Saint-Rémy."],
    ),
    QaPair(
        id="s9",
        question="What is the largest ocean on Earth?",
        answer="Pacific Ocean",
        passages=["The Pacific Ocean is the largest and deepest of Earth's oceans, covering about 63 million square miles."],
    ),
    QaPair(
        id="s10",
        question="What language was primarily spoken in the Roman Empire?",
        answer="Latin",
        passages=["Latin was the official language of the Roman Empire, though Greek was widely spoken in the east."],
    ),
]


def to_ingest_items() -> list[dict]:
    """Flatten sanity passages into contestant ingest format."""
    out = []
    for qa in SANITY_SET:
        for j, p in enumerate(qa.passages):
            out.append(
                {
                    "id": f"{qa.id}-p{j}",
                    "text": p,
                    "metadata": {"qa_id": qa.id},
                }
            )
    return out
