"""
intent.py - "Is this a real question, or is the user just being friendly?"
=========================================================================

THE PROBLEM
    You:  "hi how are you today.."
    AI:   "Based on the provided sources, here is an analysis of the phrase
           'hi how are you today': ### 1. Meaning and Usage ..."  [2][3]

    Every message went through the full RAG pipeline: search the documents,
    search the web, then "answer ONLY from the sources, cite everything".
    For a greeting that is exactly wrong. The search finds pages ABOUT the phrase,
    and the strict prompt forces Qwen to write a cited essay about them.

THE FIX: route small talk around RAG
    message --> is_small_talk()?  yes --> no search, friendly prompt, short human reply
                                  no  --> normal RAG pipeline (search, re-rank, cite)

    The check is a cheap word test (no AI call, ~0 ms): a message counts as small talk
    only if EVERY word is a "social" word. One topic word ("claims", "deadline",
    "FHIR") and it is treated as a real question, so we never skip a search the
    user actually needed.

        "hi"                              -> True
        "hi how are you today.."          -> True
        "thanks a lot!"                   -> True
        "who are you?"                    -> True
        "hi, what is the claim deadline?" -> False  ("claim", "deadline")
        "how are rejected claims handled" -> False

Learn more: docs/learning/15_conversation_memory.md
"""

from __future__ import annotations

import re

# Words that carry no TOPIC - only greetings, politeness, feelings and "who are you".
_SOCIAL_WORDS = set("""
hi hii hiii hello helo hey heya hiya yo sup hola namaste howdy greetings
good morning afternoon evening night day gm gn
how hows how's are r you u ya ye doing going today tonight there
what whats what's wassup wazzup up is it its it's all
i im i'm am me my we fine well great good ok okay okk k kk cool nice awesome perfect
not bad too also and so very really much lot a an the
thanks thank thx ty tysm thankyou appreciate appreciated cheers
bye goodbye byee cya see later soon take care night
nice meet meeting glad hope hoping your you're youre yourself
who can do your name made built created are
buddy friend mate bro dude sir madam man
oh ah hmm haha hehe lol yes yeah yep no nope sure alright
""".split())

# Short replies that are ONLY an acknowledgement ("ok", "great", "thanks") are
# also small talk - but never treat a lone "yes"/"no" as small talk mid-conversation:
# it may answer a question the assistant just asked. Those go through the normal path.
_NEEDS_CONTEXT = {"yes", "yeah", "yep", "no", "nope", "sure"}

_MAX_WORDS = 14   # longer messages are almost always real questions


def is_small_talk(message: str) -> bool:
    """True if the message is ONLY a greeting / thanks / goodbye / 'who are you'."""
    words = re.findall(r"[a-z']+", message.lower())
    if not words or len(words) > _MAX_WORDS:
        return False
    if all(w in _NEEDS_CONTEXT for w in words):
        return False
    return all(w in _SOCIAL_WORDS for w in words)
