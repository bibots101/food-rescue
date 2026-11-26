"""Explaining a safety verdict with the source documents (the "generation" in RAG).

The rule engine has already decided (approve / warn / ask / reject). This step never changes that
decision: hard rules always win. It finds the passages that justify the decision and asks the LLM to
write a short explanation for the merchant that cites one of them.

Grounding check: the cited passage must be one of the retrieved passages, and the quote must really
appear in it. Otherwise the LLM's text is discarded and the rule's own explanation is used instead.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from foodrescue.llm.client import LLMClient, LLMError
from foodrescue.models import Donation
from foodrescue.rag.corpus import Passage
from foodrescue.rag.index import Retriever
from foodrescue.safety.rules import Decision, Finding, SafetyVerdict


class CitedExplanation(BaseModel):
    """What the LLM must return."""

    explanation: str = Field(description="1-2 short sentences for the merchant, plain words")
    passage_id: str = Field(description="id of the passage that supports the explanation, e.g. 'WHO-5K p19 #1'")
    quote: str = Field(description="a short exact quote (5 to 25 words) copied from that passage")


class GroundedExplanation(BaseModel):
    """What the safety agent attaches to its verdict."""

    text: str
    source: str  # e.g. "WHO-5K, page 19"
    quote: str
    passage_id: str
    grounded: bool  # False = the LLM's answer failed the check, the rule's explanation is used
    retrieval: str  # "embeddings" or "bm25"


SYSTEM_PROMPT = """\
You explain food safety decisions to shop owners who donate food. The decision is already made by
rules; you never change it. Using ONLY the passages given, write 1-2 short sentences in plain words
that explain the decision, then give the id of the passage you used and copy a short exact quote
(5 to 25 words) from it. If no passage supports the decision, explain with the rule text and pick the
closest passage. Match the decision exactly: if the food was refused, say why it is unsafe; if we only ask the
merchant a question, never say the food is unsafe, say what the merchant must confirm and why; if it was
accepted with a warning, say what to be careful about."""

MEANING = {Decision.REJECT: "refused as unsafe",
           Decision.NEEDS_HUMAN: "not refused: we ask the merchant to confirm before a driver is sent",
           Decision.WARN: "accepted, with a warning"}


def build_query(donation: Donation, finding: Finding) -> str:
    """The search query is built by our code from structured fields, in English, like the documents."""
    item = next((i for i in donation.items if i.name == finding.item), donation.items[0])
    return (f"{finding.title}. {item.name} ({item.category.value.replace('_', ' ')}), "
            f"stored {donation.storage_of(item).value}. {finding.explanation}")


def explain(verdict: SafetyVerdict, donation: Donation, retriever: Retriever, llm: LLMClient,
            k: int = 3) -> GroundedExplanation | None:
    """Explain the most severe finding of the verdict, citing the documents."""
    findings = [f for f in verdict.findings if f.decision is not Decision.APPROVE]
    if not findings:
        return None
    severity = {Decision.REJECT: 3, Decision.NEEDS_HUMAN: 2, Decision.WARN: 1}
    finding = max(findings, key=lambda f: severity[f.decision])
    hits = retriever.search(build_query(donation, finding), k)
    if not hits:
        return None
    passages = {p.id: p for p, _ in hits}
    context = "\n\n".join(f"[{p.id}]\n{p.text}" for p in passages.values())
    question = (f"Donation: {donation.describe()} (stored {donation.storage_text()}).\n"
                f"Decision: {MEANING[finding.decision]}, because of rule {finding.rule_id}: {finding.title}. "
                f"{finding.explanation}\n\nPassages:\n{context}")
    best = hits[0][0]
    fallback = GroundedExplanation(text=finding.explanation, source=best.citation(), quote="",
                                   passage_id=best.id, grounded=False, retrieval=retriever.last_method)
    try:
        answer = llm.chat_json([{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": question}], CitedExplanation)
    except LLMError:
        return fallback
    passage = passages.get(answer.passage_id.strip().strip("[]").strip())  # ids are shown as "[id]"
    if passage is None or not quote_in(answer.quote, passage):
        return fallback
    return GroundedExplanation(text=answer.explanation.strip(), source=passage.citation(),
                               quote=answer.quote.strip(), passage_id=passage.id, grounded=True,
                               retrieval=retriever.last_method)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9°]+", " ", text.lower()).strip()


def quote_in(quote: str, passage: Passage) -> bool:
    """The quote must appear in the passage (ignoring case, punctuation and line breaks)."""
    words = _normalize(quote)
    return len(words.split()) >= 4 and words in _normalize(passage.text)
