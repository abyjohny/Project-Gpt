"""
memory.py - PHASE 3: Conversation memory, so follow-up questions work.
=====================================================================

THE PROBLEM
    You:  "What payer response codes are there?"
    AI:   "1 APPROVED, 2 QUERIED, 3 REJECTED ..."
    You:  "and what about the second one?"

    A human understands "the second one" = QUERIED. But our SEARCH step does not
    remember anything: it embeds the words "and what about the second one?" and
    looks for chunks that mean... nothing in particular. Wrong chunks -> bad answer.

    Just giving the AI the chat history is NOT enough. The AI only reads what the
    SEARCH found - and the search was done with the vague question.

THE FIX: "query rewriting" (also called "question condensing")
    Before searching, we ask Qwen ONE small extra question:
        "Here is the chat so far. Rewrite the last question so it makes sense ON ITS OWN."
    "and what about the second one?"  ->  "What does payer response code 2 (QUERIED) mean?"

    Then:
        SEARCH with the rewritten question   (finds the right chunks)
        ANSWER the original question, with the chat history  (sounds natural)

        question --> [rewrite with history] --> search --> re-rank --> Qwen answers
                          ^                                             |
                          +------------- memory (last N turns) <--------+

COST
    One extra short call to Qwen (~1-1.7 s on a 4b model).

MEASURED (python evaluate.py followups, 8 conversations, hybrid + re-rank):
    no memory (search the follow-up as typed) ...... Hit@1 50%  MRR 0.60
    Qwen rewrite alone ............................. Hit@1 62-75%  (a 4b model sometimes
                                                     picks the WRONG item from a list!)
    previous question + follow-up .................. Hit@1 88%  MRR 0.94   <- free
    previous question + Qwen rewrite ............... Hit@1 88%  MRR 0.94   (+1.5 s)

    Lesson: the simple, free trick is as good as the clever, slow one. So the default
    mode is "context": glue the previous question in front of a follow-up.
    The re-ranker then reads the whole thing and works out what is meant.

CLARIFICATION REQUESTS ("didn't understand, can you clarify?")
    These are NOT new questions - they ask us to explain the LAST answer again.
    Searching for the words "didn't understand can you clarify" finds random chunks,
    the re-ranker rightly rejects them all, and the user gets "I could not find...".
    So every turn remembers its SOURCES, and a clarification re-uses them (no search).

Learn more: docs/learning/15_conversation_memory.md
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag import config, ollama_client

#   "context" -> follow-ups are searched as  previous question + follow-up   (free)
#   "auto"    -> follow-ups are searched as  previous question + Qwen rewrite (+~1.5 s)
#   "always"  -> like auto, but EVERY question is rewritten once there is history
#   "off"     -> no memory in the search at all
VALID_REWRITE_MODES = ("context", "auto", "always", "off")

# The glued search query never grows longer than this many words.
MAX_CONTEXT_WORDS = 60


@dataclass
class Turn:
    """One question + answer from the conversation."""

    question: str          # what the user typed
    search_query: str      # what we actually searched for (rewritten, or the same)
    answer: str
    sources: list = field(default_factory=list)   # the chunks the answer was based on


# ----------------------------------------------------------------
# Does this question depend on the earlier conversation?
# ----------------------------------------------------------------
# Words that POINT BACK at something said before.
_REFERENCE_WORDS = re.compile(
    r"\b(it|its|it's|they|them|their|theirs|this|that|these|those|he|she|his|her|"
    r"former|latter|above|previous|earlier|same|such|"
    r"first|second|third|fourth|fifth|last|next|other|another|"
    r"one|ones|both|either|neither|each)\b",
    re.IGNORECASE,
)
# Ways people START a follow-up.
_FOLLOW_UP_STARTS = re.compile(
    r"^\s*(and|also|but|so|then|or|what about|how about|and what|and how|why not|"
    r"same for|what if|more|explain more|tell me more|elaborate|example|why\??$)\b",
    re.IGNORECASE,
)

# Ways people say "I didn't get your LAST answer" - a request to explain it again.
_CLARIFY = re.compile(
    r"\b(clarify|clarification|didn'?t (get|understand)|did not (get|understand)|"
    r"don'?t (get|understand)|do not (get|understand)|not clear|unclear|confus\w*|"
    r"what do you mean|what does that mean|meaning of (that|this|it)|"
    r"simpler|simply|simple (words|terms|language)|plain (english|words|language)|layman|eli5|"
    r"rephrase|reword|explain (it|that|this|again|more|better|differently)|"
    r"elaborate|more detail\w*|in detail|an example|examples?\??$|"
    r"summari[sz]e|tl;?dr|shorter|in short|briefly|say (it|that) again|once more)\b",
    re.IGNORECASE,
)
# Words that carry no TOPIC. If a clarification has (almost) nothing else in it,
# it is about the previous answer - not a new question.
_FILLER_WORDS = set("""
a an the i me my you your we us our it its this that these those there here
can could would will please pls plz just again more bit little some any one
is are was were be do does did didnt dont not no yes ok okay so and but or
get got understand understood clarify clarification explain explained mean meaning
simple simpler simply words terms language plain english layman eli5 rephrase
reword elaborate detail details example examples summarise summarize short shorter
briefly brief tldr say once differently better clear unclear confused confusing
what how about with in on of for to give show tell
""".split())


def looks_like_follow_up(question: str) -> bool:
    """A cheap guess (no AI needed): does this question lean on the conversation?

        "What is the claim submission deadline?"   -> False  (complete on its own)
        "and the second one?"                      -> True   (starts with "and", says "second one")
        "why?"                                     -> True   (very short)
        "didnt understand can you clarify.."       -> True   (asks about the last answer)
    """
    words = re.findall(r"[\w'-]+", question)
    if len(words) <= 3:                      # "why?", "and code 5?", "an example?"
        return True
    if _FOLLOW_UP_STARTS.search(question) or _CLARIFY.search(question):
        return True
    return bool(_REFERENCE_WORDS.search(question))


def is_clarification(question: str) -> bool:
    """Is this ONLY a request to re-explain the last answer (no new topic)?

        "didnt understand can you clarify.."       -> True
        "explain that in simpler words please"     -> True
        "can you give an example?"                 -> True
        "give an example of a FHIR claim bundle"   -> False  (new topic: FHIR claim bundle)
        "I don't understand workflow 19"           -> False  (names a topic -> search for it)
    """
    if not _CLARIFY.search(question):
        return False
    words = [w.replace("'", "") for w in re.findall(r"[a-z0-9']+", question.lower())]
    topic_words = [w for w in words if w not in _FILLER_WORDS and not _REFERENCE_WORDS.fullmatch(w)]
    return len(topic_words) == 0


# ----------------------------------------------------------------
# The memory itself
# ----------------------------------------------------------------

def _shorten(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


class ConversationMemory:
    """Remembers the last few turns and turns follow-ups into stand-alone questions."""

    def __init__(self, max_turns: int = config.MEMORY_TURNS, rewrite_mode: str = config.QUERY_REWRITE):
        self.max_turns = max_turns
        self.rewrite_mode = rewrite_mode
        self.turns: list[Turn] = []

    # ---- basic bookkeeping ----
    def add(self, question: str, search_query: str, answer: str, sources: list | None = None) -> None:
        self.turns.append(Turn(question, search_query, answer, list(sources or [])))
        self.turns = self.turns[-self.max_turns:] if self.max_turns > 0 else []

    def previous_sources(self) -> list:
        """The chunks the LAST answer was based on (empty if there is none)."""
        return list(self.turns[-1].sources) if self.turns else []

    def clear(self) -> None:
        self.turns.clear()

    def __len__(self) -> int:
        return len(self.turns)

    def as_messages(self) -> list[dict]:
        """The history in the format the chat model expects.

        Old answers are SHORTENED: the AI only needs a reminder of what was said,
        and long histories would push the retrieved SOURCES out of the context window.
        """
        messages: list[dict] = []
        for turn in self.turns:
            messages.append({"role": "user", "content": turn.question})
            messages.append({"role": "assistant",
                             "content": _shorten(turn.answer, config.HISTORY_ANSWER_MAX_CHARS)})
        return messages

    # ---- query rewriting ----
    def needs_rewrite(self, question: str) -> bool:
        """Does this question need the conversation to be understood?"""
        if not self.turns or self.rewrite_mode == "off":
            return False
        if self.rewrite_mode == "always":
            return True
        return looks_like_follow_up(question)

    def prepare(self, question: str) -> tuple[str, str | None]:
        """Decide what to SEARCH for. Returns (search_query, hint_for_the_answering_ai).

            "context":  search "What diagnosis codes exist? What does the last one mean?"
                        hint   'This is a follow-up to "What diagnosis codes exist?" ...'
            "auto":     search "What diagnosis codes exist? What does 'final' mean?"
                        hint   "In this conversation, that means: What does 'final' mean?"

        The hint matters: without it, the AI may read "the last one" as
        "the last row of source [4]" instead of "the last item in MY previous answer".
        """
        if not self.needs_rewrite(question):
            return question, None
        previous = self.turns[-1]
        if self.rewrite_mode == "context":
            hint = (f'This is a follow-up to the previous question "{previous.question}". '
                    "Words like it / that / the last one refer to that conversation and your "
                    "previous answer, not to the order of the sources.")
            return _glue(previous.search_query, question), hint
        standalone = self.rewrite(question)
        hint = f"In this conversation, that means: {standalone}" if standalone != question else None
        return _glue(previous.search_query, standalone), hint

    def rewrite(self, question: str) -> str:
        """Ask Qwen for a stand-alone version of `question`.

        If anything goes wrong we simply return the question unchanged.
        """
        if not self.turns:
            return question
        try:
            reply = "".join(ollama_client.chat(
                [{"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                 {"role": "user", "content": self._rewrite_request(question)}],
                temperature=0.0,
                stream=False,
                extra_options={"num_predict": 80},   # a question is short - stop early
            ))
        except ollama_client.OllamaError:
            return question
        return _clean_rewrite(reply, fallback=question)

    def _rewrite_request(self, question: str) -> str:
        lines = ["CONVERSATION SO FAR:"]
        for turn in self.turns:
            lines.append(f"User: {turn.search_query}")
            # The answer matters: "the second one" usually points INTO the answer's list.
            lines.append(f"Assistant: {_shorten(turn.answer, config.HISTORY_ANSWER_MAX_CHARS)}")
        lines += ["", f"FOLLOW-UP QUESTION: {question}", "", "STAND-ALONE QUESTION:"]
        return "\n".join(lines)


REWRITE_SYSTEM_PROMPT = """You rewrite follow-up questions so they can be understood WITHOUT the conversation.

Rules:
- Replace words like "it", "that", "the second one", "those" with the exact thing they refer to.
  COPY that thing word-for-word from the conversation (names, codes, numbers, terms).
  Example: after a list "red, green, blue", "the green one" -> "green", never another item.
- Keep the user's intent and wording. Do NOT answer the question.
- Do NOT add words, products or topics the user did not mention.
- If the question is already complete on its own, repeat it unchanged.
- Reply with ONLY the rewritten question, on one line."""


def _glue(previous: str, question: str) -> str:
    """'What codes exist?' + 'and the second one?' -> 'What codes exist? and the second one?'

    Only the LAST words are kept, so a long chain of follow-ups can't grow forever.
    """
    words = f"{previous.strip()} {question.strip()}".split()
    return " ".join(words[-MAX_CONTEXT_WORDS:])


def _clean_rewrite(reply: str, fallback: str) -> str:
    """Models sometimes add labels or quotes - strip them. Reject obviously bad output."""
    lines = [line.strip() for line in reply.strip().splitlines() if line.strip()]
    if not lines:
        return fallback
    text = re.sub(r"^(stand-?alone question|rewritten question|question)\s*:\s*", "", lines[0], flags=re.IGNORECASE)
    text = text.strip().strip('"\'*`').strip()
    # Far too long = the model probably started ANSWERING instead of rewriting.
    if not text or len(text) > max(300, len(fallback) * 4):
        return fallback
    return text
