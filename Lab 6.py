import csv
import json
import re
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    import numpy as np
    import sklearn
    import sentence_transformers
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Missing packages. Please run:")
    print("  pip install numpy scikit-learn sentence-transformers")
    raise SystemExit(1)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
WORD_BUDGET = 40          
TOP_K = 3                 
STRATEGIES = ["0overlap", "10overlap", "sentence"]
BACKENDS = ["tfidf", "embedding"]
PREVIEW_CHARS = 120

# Logging: mirror everything printed to run_output.txt

class TeeStream:
    """File-like object that writes to several streams at once."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


class RunLog:
    """Context manager: stdout -> console + run_output.txt for the duration of the run."""

    def __init__(self, path="run_output.txt"):
        self.path = path
        self.handle = None

    def __enter__(self):
        self.handle = open(self.path, "w", encoding="utf-8")
        sys.stdout = TeeStream(sys.__stdout__, self.handle)
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.stdout = sys.__stdout__
        if self.handle is not None:
            self.handle.close()
        return False



# Synthetic data (handbook documents, labeled evidence, queries, FAQ)
 
DOCS = {
    "late": (
        "University late submission policy applies to all undergraduate coursework in taught modules. "
        "Late work receives a 10 percent penalty for each calendar day late, up to three days, and is "
        "refused after that unless a formal extension was approved in writing before the deadline. "
        "Extensions may be granted for documented medical issues, family emergencies, or verified system "
        "outages, but students must email the module leader with a medical certificate before the due date. "
        "Late penalties are applied automatically by the grading system and appear on the feedback sheet. "
        "Group projects follow the same rules unless the brief states otherwise. Students who miss the "
        "absolute cutoff should contact student services to discuss deferred assessment options. Appeal "
        "requests must be filed within five working days of the grade release. Teaching assistants cannot "
        "waive late penalties without approval from the module leader. Keep copies of all submission "
        "receipts for your records. Weekend days still count as calendar days for the penalty clock. "
        "Resit coursework uses the same late rules unless the exam board publishes a different notice. "
        "Please note a conflicting portal notice elsewhere may mention a five day window, but this late "
        "policy document is the assessment office source for day based penalties on taught modules. "
        "Markers should never invent a local late rule that contradicts this assessment office text."
    ),
    "password": (
        "Password reset procedures for university accounts are handled through the IT self-service portal. "
        "Navigate to the login page and select Forgot Password, then enter your student email and complete "
        "an identity check with your student ID and date of birth. "
        "A reset link is sent to your registered personal email within ten minutes and the link expires "
        "after one hour. New passwords must be at least twelve characters and include upper case, lower "
        "case, a digit and a symbol, and reusing any of the last five passwords is not allowed. "
        "If you do not receive the email, check the spam folder first. Staff at the IT help desk can reset "
        "passwords in person after photo ID verification. Multi-factor authentication must be re-enrolled "
        "after a password change. Shared passwords between classmates violate the acceptable use policy "
        "and may lead to account suspension. Contact helpdesk@university.edu if the portal reports an "
        "error during reset. Temporary guest accounts cannot use the self-service reset form. Always "
        "sign out of shared lab machines after changing credentials so the old session cannot be reused. "
        "Store recovery codes offline and never send them by chat or email to anyone claiming to be IT support."
    ),
    "assignment": (
        "Assignment submission guidelines require all coursework to be uploaded through the virtual "
        "learning environment before the stated deadline. Accepted file formats are PDF for written "
        "reports and ZIP for programming projects. "
        "Filenames should follow the pattern ModuleCode_StudentID_Assignment so markers can match files "
        "to students without opening every archive by hand during busy marking weeks at the end of term "
        "when dozens of late uploads and resubmissions arrive in the same short window and support tickets "
        "already fill the help desk queue. "
        "The plagiarism checker runs automatically on every upload. Similarity scores above 25 percent "
        "are flagged for manual review by the module leader. Students may upload a draft once for "
        "feedback before the final submission. Only the last upload before the deadline is graded. "
        "Technical problems during upload should be reported immediately with screenshots to the help "
        "desk so a late penalty waiver can be considered. Printed hard copies are not accepted unless "
        "the brief specifically requests them. Group assignments need one designated member to submit "
        "on behalf of the team. Confirmation emails are sent after a successful upload. "
        "Important conflict note: the portal help page states late uploads are accepted for up to five "
        "calendar days with a flat five percent penalty applied once, which disagrees with the late "
        "policy day based ten percent rule, so always confirm with your module leader which source applies."
    ),
    "contractor": (
        "Contractor access policy covers external vendors working on university systems and buildings. "
        "Contractors receive temporary accounts limited to the systems listed in their statement of work. "
        "Access is reviewed every ninety days and removed when the contract ends. Contractors must "
        "complete security awareness training before credentials are issued. They cannot sponsor guest "
        "Wi-Fi for others and must not store university data on personal devices. Data export requires "
        "written approval from the information security office. Contractor accounts use multi-factor "
        "authentication and are monitored for unusual activity. Lost badges must be reported within "
        "twenty-four hours. Permanent staff accounts must never be shared with contractors. Requests "
        "for elevated privileges go through the change advisory board. Breach of these rules can "
        "terminate the contract and trigger a security incident review. "
        "Contractors can work remotely only if the signed contract explicitly permits remote work, and "
        "employee remote work rules do not apply to contractors even when both groups use the same VPN. "
        "The policy does not apply to full-time employees or enrolled students seeking temporary lab "
        "accounts, who must use the student account process instead of contractor forms."
    ),
    "printer": (
        "Campus printer and print quota rules are managed by IT services for students and visitors. "
        "Each student receives a monthly quota of 150 pages in black and white. Colour pages cost three "
        "times as much against the quota. Quotas reset on the first day of each month and unused pages "
        "do not roll over. Additional pages can be purchased through the student portal in packs of "
        "fifty. Printers are located in the library, lab buildings, and student centre. Jobs older than "
        "twenty-four hours in the print queue are deleted automatically. Paper jams should be reported "
        "using the help button on the printer panel. Do not attempt to open the printer casing yourself. "
        "Double-sided printing is the default to save paper. Large posters must use the dedicated "
        "plotter in the media lab and require staff booking. Guest printing is available with a prepaid "
        "card from the library desk. Misuse of printers for commercial work is prohibited under campus "
        "rules. For blank printer pages, check ink levels and run the cleaning utility, then print a "
        "test page afterwards before opening a hardware ticket. "
        "A related but conflicting notice in some lab doors still says the monthly student quota is 200 "
        "pages, but the IT services printer document here is the current source for the 150 page limit."
    ),
}


EV_LATE_RULE = (
    "Late work receives a 10 percent penalty for each calendar day late, up to three days, and is "
    "refused after that unless a formal extension was approved in writing before the deadline."
)
EV_LATE_ILLNESS = (
    "Extensions may be granted for documented medical issues, family emergencies, or verified system "
    "outages, but students must email the module leader with a medical certificate before the due date."
)
EV_PASSWORD_RESET = (
    "Navigate to the login page and select Forgot Password, then enter your student email and complete "
    "an identity check with your student ID and date of birth."
)
EV_PASSWORD_RULES = (
    "New passwords must be at least twelve characters and include upper case, lower "
    "case, a digit and a symbol, and reusing any of the last five passwords is not allowed."
)
EV_ASSIGNMENT_FORMATS = "Accepted file formats are PDF for written reports and ZIP for programming projects."
EV_CONTRACTOR_REMOTE = (
    "Contractors can work remotely only if the signed contract explicitly permits remote work, and "
    "employee remote work rules do not apply to contractors even when both groups use the same VPN."
)
EV_PRINTER_QUOTA = "Each student receives a monthly quota of 150 pages in black and white."
EV_PRINTER_BLANK = (
    "For blank printer pages, check ink levels and run the cleaning utility, then print a "
    "test page afterwards before opening a hardware ticket."
)


RULE_EXCEPTIONS = [
    {"doc": "late", "label": "late_penalty_with_extension", "evidence": EV_LATE_RULE},
    {"doc": "password", "label": "password_rules", "evidence": EV_PASSWORD_RULES},
    {"doc": "contractor", "label": "contractor_remote", "evidence": EV_CONTRACTOR_REMOTE},
]


def make_query(qid, text, doc, evidence, split):
    return {"id": qid, "query": text, "doc": doc, "evidence": evidence,
            "answerable": doc is not None, "split": split}


DEV_QUERIES = [
    make_query("dev1", "What is the late penalty and when is late work refused?", "late", EV_LATE_RULE, "dev"),
    make_query("dev2", "How do I reset my university password?", "password", EV_PASSWORD_RESET, "dev"),
    make_query("dev3", "What file formats are accepted for assignment upload?", "assignment",
               EV_ASSIGNMENT_FORMATS, "dev"),
    make_query("dev4", "Can contractors work remotely?", "contractor", EV_CONTRACTOR_REMOTE, "dev"),
]

TEST_QUERIES = [
    make_query("test1", "What proof is needed when I was ill and need more time?", "late",
               EV_LATE_ILLNESS, "test"),
    make_query("test2", "What are the rules for a new password?", "password", EV_PASSWORD_RULES, "test"),
    make_query("test3", "How many pages is the monthly student print quota?", "printer",
               EV_PRINTER_QUOTA, "test"),
    make_query("test4", "What should I do for blank printer pages?", "printer", EV_PRINTER_BLANK, "test"),
    make_query("unans1", "What are tomorrow's flight prices?", None, None, "test"),
    make_query("unans2", "Who won the football match?", None, None, "test"),
]

FAQ = [
    {"id": "faq:password", "question": "How do I reset my password?",
     "answer": "Use the recovery link on the login page.", "source": "it_help.txt"},
    {"id": "faq:submit", "question": "Where do I submit my assignment?",
     "answer": "Use the assignment upload page in the learning portal.", "source": "assessment.txt"},
    {"id": "faq:late", "question": "Can illness justify a late submission?",
     "answer": "Request an extension with medical evidence before the deadline.", "source": "handbook.txt"},
    {"id": "faq:printer", "question": "What is the monthly student print quota?",
     "answer": "Students get 150 black and white pages each month.", "source": "printer_policy.txt"},
    {"id": "faq:wifi", "question": "How do I connect to campus Wi-Fi?",
     "answer": "Join Eduroam and sign in with your university email and password.",
     "source": "network_guide.txt"},
]





def count_words(text):
    return len(text.split())


def shorten(text, limit):
    return text if len(text) <= limit else text[:limit] + "..."


def show(value):
    """Print empty strings as '' so blank input is visible in transcripts."""
    return repr(value) if value == "" else value



# Chunking (Task 2)

def windows(text, size=WORD_BUDGET, overlap=10, offset=0):
    """Fixed word windows -> list of (start_char, end_char). `offset` shifts the character positions."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    spans = [m.span() for m in re.finditer(r"\S+", text)]
    if not spans:
        return []

    stride = size - overlap
    out = []
    first = 0
    while True:
        last = min(first + size, len(spans))               
        out.append((offset + spans[first][0], offset + spans[last - 1][1]))
        if last >= len(spans):
            return out
        first += stride


def split_sentences(text):
    """Return [(start, end, word_count)] for each sentence; a boundary is . ! or ? plus whitespace."""
    pieces, begin = [], 0
    for m in re.finditer(r"[.!?]\s+", text):
        pieces.append((begin, m.end()))
        begin = m.end()
    if begin < len(text) and text[begin:].strip():
        pieces.append((begin, len(text)))

    sentences = []
    for a, b in pieces:
        b = a + len(text[a:b].rstrip())                    
        sentences.append((a, b, count_words(text[a:b])))
    return sentences


def sentence_chunks(text, size=WORD_BUDGET):
    """Pack whole sentences into chunks of at most `size` words.
    A sentence that is longer than the budget is cut with windows() (no overlap) as a fallback."""
    if size <= 0:
        raise ValueError("size must be positive")

    chunks = []
    open_start = open_end = None
    open_words = 0

    def close_chunk():
        nonlocal open_start, open_end, open_words
        if open_start is not None:
            chunks.append((open_start, open_end))
        open_start = open_end = None
        open_words = 0

    for a, b, n in split_sentences(text):
        if n > size:                                       
            close_chunk()
            chunks.extend(windows(text[a:b], size=size, overlap=0, offset=a))
        elif open_start is None:
            open_start, open_end, open_words = a, b, n
        elif open_words + n <= size:
            open_end, open_words = b, open_words + n
        else:
            close_chunk()
            open_start, open_end, open_words = a, b, n
    close_chunk()
    return chunks


def make_chunks(docs, strategy):
    """Turn {name: text} into chunk records {id, doc, start, end, text} for one strategy."""
    splitters = {
        "0overlap": lambda t: windows(t, size=WORD_BUDGET, overlap=0),
        "10overlap": lambda t: windows(t, size=WORD_BUDGET, overlap=10),
        "sentence": lambda t: sentence_chunks(t, size=WORD_BUDGET),
    }
    split_fn = splitters.get(strategy, splitters["sentence"])

    records = []
    for name, text in docs.items():
        for n, (a, b) in enumerate(split_fn(text)):
            records.append({"id": f"{name}:{n}", "doc": name, "start": a, "end": b, "text": text[a:b]})
    return records


def chunk_stats(records, docs):
    """Print chunk count, min/max/avg words, duplicated words and rule+exception coverage."""
    if not records:
        print("  No chunks.")
        return

    sizes = [count_words(r["text"]) for r in records]
    original = sum(count_words(t) for t in docs.values())

    print("  chunk_count:", len(records))
    print("  min_words:", min(sizes), "max_words:", max(sizes),
          "avg_words:", round(sum(sizes) / len(sizes), 2))
    print("  duplicated_words (chunk_total - original):", sum(sizes) - original)

    print("  rule+exception intact in one chunk?")
    for item in RULE_EXCEPTIONS:
        intact = any(r["doc"] == item["doc"] and item["evidence"] in r["text"] for r in records)
        print("   ", item["label"], "->", intact)



# Retrieval: TF-IDF or sentence embeddings behind one search() method
class Retriever:
    """Rank records for a query with either TF-IDF or MiniLM embeddings (cosine similarity)."""

    def __init__(self, records, backend="tfidf", model=None):
        self.records = records
        self.backend = backend
        texts = [r["text"] for r in records]

        if backend == "tfidf":
            self.vectorizer = TfidfVectorizer()
            self.matrix = self.vectorizer.fit_transform(texts)
            self.model = None
        else:
            self.model = model if model is not None else SentenceTransformer(MODEL_NAME, device="cpu")
            self.matrix = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
            self.vectorizer = None

    def score_all(self, query):
        if self.backend == "tfidf":
            q = self.vectorizer.transform([query])
            if q.nnz == 0:                                  
                return None
            return (self.matrix @ q.T).toarray().ravel()
        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        return np.dot(self.matrix, q)

    def search(self, query, k=TOP_K):
        """Top-k [(record, score)]; empty list for blank queries or zero TF-IDF overlap."""
        if not query.strip() or k <= 0:
            return []
        scores = self.score_all(query)
        if scores is None:
            return []
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.records[i], float(scores[i])) for i in order]


# FAQ chatbot (Task 1)

def build_faq_bot(backend="tfidf"):
    records = [dict(item, text=item["question"] + " " + item["answer"]) for item in FAQ]
    return Retriever(records, backend)


def faq_reply(engine, query):
    """Exact (case-insensitive) question -> stored answer; otherwise suggestions only."""
    if not query.strip():
        return "Please enter a question."

    wanted = query.strip().casefold()
    for item in FAQ:
        if wanted == item["question"].casefold():
            return f"{item['answer']} [source: {item['source']}]"

    hits = engine.search(query, k=3)
    if not hits:
        return "No matching FAQ. Please contact the help desk."

    lines = ["Possible matches; choose or rephrase your question:"]
    lines += [f"{rec['question']} (similarity {score:.3f})" for rec, score in hits]
    return "\n".join(lines)


def run_faq_chatbot_interactive():
    print("=" * 60)
    print("FAQ Chatbot interactive (TF-IDF). Type exit to stop.")
    print("=" * 60)
    engine = build_faq_bot("tfidf")
    while True:
        try:
            query = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting FAQ chatbot.")
            return
        if query.strip().lower() == "exit":
            return
        print("Bot:", faq_reply(engine, query))



# Evaluation helpers (Task 3)

def is_evidence_hit(record, query):
    """A hit needs the labeled document AND the full evidence string inside the chunk."""
    return record["doc"] == query["doc"] and query["evidence"] in record["text"]


def audit_token_lengths(model, texts, label):
    """Check that no chunk/query is longer than the encoder's max_seq_length (no silent truncation)."""
    longest = max((len(model.tokenizer(t, truncation=False)["input_ids"]) for t in texts), default=0)
    limit = model.max_seq_length
    print(f"  Token audit ({label}): max_tokens={longest}", f"limit={limit}", f"fits={longest <= limit}")
    if longest > limit:
        raise ValueError("Model limit exceeded; rechunk or shorten query")
    return longest


def score_matrix_for(backend, texts, query_texts, model):
    """Return (queries x chunks score matrix, encode_seconds, search_seconds)."""
    t0 = time.perf_counter()
    if backend == "tfidf":
        vec = TfidfVectorizer()
        chunk_vecs = vec.fit_transform(texts)
        query_vecs = vec.transform(query_texts)
        encode_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        matrix = (query_vecs @ chunk_vecs.T).toarray()
    else:
        chunk_vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        query_vecs = model.encode(query_texts, normalize_embeddings=True, convert_to_numpy=True)
        encode_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        matrix = query_vecs @ chunk_vecs.T
    return matrix, encode_s, time.perf_counter() - t1


def top_k_indices(row, k=TOP_K):
    return list(np.argsort(-row, kind="stable")[:k])


def split_metrics(matrix, records, queries, split):
    """Hit@3 and MRR@3 over the answerable queries of one split."""
    hits, reciprocal = [], []
    for qi, q in enumerate(queries):
        if q["split"] != split or not q["answerable"]:
            continue
        rank_of_hit = next((rank for rank, idx in enumerate(top_k_indices(matrix[qi]), 1)
                            if is_evidence_hit(records[idx], q)), None)
        hits.append(1 if rank_of_hit else 0)
        reciprocal.append(1.0 / rank_of_hit if rank_of_hit else 0.0)
    if not hits:
        return 0.0, 0.0
    return sum(hits) / len(hits), sum(reciprocal) / len(reciprocal)


# Task 1: tokenization + FAQ demo
def run_task1(model):
    print("=" * 60)
    print("TASK 1: Tokenization Comparison")
    print("=" * 60)
    print("MARKER_TOKEN_TABLE_START")

    samples = ["AI, search!", "I can't log in.", "help@example.com",
               "CS4944 version 2.1", "", "file_v2.1-final"]

    print("")
    print(f"{'Input':<22} {'Whitespace':<40} {'Regex':<40} {'Model subwords (2 EN)'}")
    print("-" * 140)

    for n, text in enumerate(samples):
        by_space = text.split()
        by_regex = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
        left = f"{by_space} n={len(by_space)}"
        middle = f"{by_regex} n={len(by_regex)}"
        right = str(model.tokenizer.tokenize(text)) if n < 2 else "-"      # subwords for 2 English inputs
        print(f"{text if text else '(empty)':<22} {left:<40} {middle:<40} {right}")

    print("MARKER_TOKEN_TABLE_END")
    print("")
    print("Notes:")
    print("- Whitespace keeps punctuation attached (e.g. 'AI,' and 'search!').")
    print("- Regex \\b\\w+\\b drops punctuation but splits emails and contractions like can't.")
    print("- Empty string gives zero tokens for both methods.")
    print("- Model subwords are different again; do not treat the regex as a full tokenizer.")
    print("")

    print("=" * 60)
    print("TASK 1: FAQ Chatbot (TF-IDF) demos")
    print("=" * 60)
    print("MARKER_FAQ_START")
    engine = build_faq_bot("tfidf")
    for q in ["How do I reset my password?", "password recovery", "", "volcano geology",
              "What is the monthly student print quota?", "How do I connect to campus Wi-Fi?"]:
        print("You:", show(q))
        print("Bot:", faq_reply(engine, q))
        print("")
    print("MARKER_FAQ_END")
    print("Exact questions return the stored answer plus source.")
    print("Other questions only show suggestions so we confirm before treating them as answers.")
    print("")


# Task 2: chunk tests, statistics, paraphrase retrieval
def expect_value_error(label, **kwargs):
    try:
        windows("a b c d", **kwargs)
    except ValueError:
        print(f"PASS: {label}")
        return
    raise AssertionError(f"{label}: invalid setting was accepted")


def run_chunk_tests():
    print("=" * 60)
    print("TASK 2: Chunking tests")
    print("=" * 60)
    print("MARKER_CHUNK_TESTS_START")

    assert windows("") == []
    print("PASS: windows('') -> []")

    one = windows("short text")
    assert len(one) == 1
    print("PASS: short document -> one window", one)

    expect_value_error("overlap >= size raises ValueError", size=2, overlap=2)
    expect_value_error("negative overlap raises ValueError", size=2, overlap=-1)

    long_text = " ".join(["word"] * 95) + "."
    pieces = sentence_chunks(long_text, size=40)
    sizes = [count_words(long_text[a:b]) for a, b in pieces]
    assert max(sizes) <= 40
    print("PASS: long sentence fallback, spans=", len(pieces), "max_words=", max(sizes))

    tiny = {"example": "Reset your password. Keep recovery codes private."}
    for strategy in STRATEGIES:
        for c in make_chunks(tiny, strategy):
            assert c["text"] == tiny[c["doc"]][c["start"]:c["end"]]
    print("PASS: make_chunks text == source[start:end]")
    print("MARKER_CHUNK_TESTS_END")
    print("")


def run_task2(model):
    run_chunk_tests()

    print("=" * 60)
    print("TASK 2: Chunk stats per strategy")
    print("=" * 60)
    print("MARKER_CHUNK_STATS_START")
    for name, text in DOCS.items():
        print("doc", name, "words=", count_words(text))
    for strategy in STRATEGIES:
        print("")
        print("Strategy:", strategy)
        chunk_stats(make_chunks(DOCS, strategy), DOCS)
    print("MARKER_CHUNK_STATS_END")
    print("")
    print("Sentence splitter note: abbreviations like Mr. or e.g. can still break on the period.")
    print("")

    print("=" * 60)
    print("TASK 2: Paraphrase retrieval (sentence chunks)")
    print("=" * 60)
    print("MARKER_PARAPHRASE_START")
    records = make_chunks(DOCS, "sentence")
    questions = ["How can I regain access to my account?",
                 "I was sick; can I hand work in tomorrow?",
                 "My printer has power but the pages come out blank."]

    for backend in BACKENDS:
        print("")
        print("Backend:", backend)
        engine = Retriever(records, backend=backend, model=model if backend == "embedding" else None)
        for q in questions:
            print("Query:", q)
            hits = engine.search(q, k=TOP_K)
            if not hits:
                print("  No matching records.")
            for rec, score in hits:
                print(" ", rec["id"], f"score={score:.3f}", "doc=" + rec["doc"],
                      f"offsets=({rec['start']}-{rec['end']})")
                print("   evidence:", shorten(rec["text"], 160))
    print("MARKER_PARAPHRASE_END")
    print("")

# Task 3: evaluation + document chatbot demo
def print_top3(strategy, backend, matrix, records, queries, logs, failures):
    """Print the top-3 chunks for every query, append to the CSV log and record misses."""
    print("")
    print("Top-3 details:", strategy, backend)
    for qi, q in enumerate(queries):
        print("Query:", q["query"], "| split=", q["split"], "| answerable=", q["answerable"])
        top = top_k_indices(matrix[qi])
        matched = False
        for rank, idx in enumerate(top, 1):
            rec, score = records[idx], float(matrix[qi][idx])
            print("  rank", rank, "id=" + rec["id"], f"score={score:.4f}", "doc=" + rec["doc"],
                  f"offsets=({rec['start']}-{rec['end']})")
            print("   text:", shorten(rec["text"], 140))
            logs.append({"strategy": strategy, "method": backend, "query": q["query"],
                         "split": q["split"], "rank": rank, "score": score, "chunk_id": rec["id"],
                         "doc": rec["doc"], "start": rec["start"], "end": rec["end"],
                         "text": rec["text"], "answerable": q["answerable"]})
            if q["answerable"] and is_evidence_hit(rec, q):
                matched = True

        if q["answerable"] and not matched:
            first = records[top[0]] if top else None
            failures.append({"strategy": strategy, "backend": backend, "query": q["query"],
                             "expected_doc": q["doc"], "rank": 1 if top else None,
                             "retrieved_id": first["id"] if first else None,
                             "retrieved_doc": first["doc"] if first else None,
                             "retrieved_text": first["text"] if first else None})
    print("")


def pick_best_dev(rows):
    """Best DEV row: higher Hit@3, then higher MRR@3, then prefer embedding (paraphrase support)."""
    dev_rows = [r for r in rows if r["split"] == "dev"]
    return max(dev_rows, key=lambda r: (r["hit_at_3"], r["mrr_at_3"], r["backend"] == "embedding"))


def demo_document_chatbot(engine):
    cases = [("exact-ish", "What is the late penalty and when is late work refused?", "1"),
             ("paraphrase", "How can I recover my login access?", "1"),
             ("empty", "", ""),
             ("unsupported", "What are tomorrow's flight prices?", "")]
    for label, question, choice in cases:
        print("DEMO:", label)
        print("Question:", show(question))
        hits = engine.search(question, k=TOP_K)
        if not hits:
            print("No candidate evidence; rephrase or contact a human.")
            print("")
            continue
        print("Candidate evidence; relevance must be checked:")
        for n, (rec, score) in enumerate(hits, 1):
            print(n, rec["id"], f"{score:.3f}", "characters", rec["start"], rec["end"])
            print("   ", shorten(rec["text"], PREVIEW_CHARS))
        if label == "unsupported" or choice == "":
            print("No answer selected. Rephrase or contact a human.")
        else:
            rec, _ = hits[int(choice) - 1]
            print("Retrieved passage:", rec["text"])
            print("Source:", rec["doc"], "characters", rec["start"], rec["end"])
        print("")


def run_task3(model):
    print("=" * 60)
    print("TASK 3: Retrieval evaluation (dev/test, 6 configs)")
    print("=" * 60)

    queries = DEV_QUERIES + TEST_QUERIES
    query_texts = [q["query"] for q in queries]
    rows, logs, failures = [], [], []

    print("MARKER_METRICS_START")
    print(f"{'Strategy':<12} {'Backend':<10} {'Split':<6} {'Hit@3':>7} {'MRR@3':>7} "
          f"{'Chunks':>7} {'Encode(s)':>9} {'Search(s)':>9}")
    print("-" * 77)

    for strategy in STRATEGIES:
        records = make_chunks(DOCS, strategy)
        texts = [r["text"] for r in records]

        print("Token audit for strategy", strategy)
        audit_token_lengths(model, texts + query_texts, strategy)

        for backend in BACKENDS:
            matrix, encode_s, search_s = score_matrix_for(backend, texts, query_texts, model)

            for split in ["dev", "test"]:
                hit, mrr = split_metrics(matrix, records, queries, split)
                rows.append({"strategy": strategy, "backend": backend, "split": split,
                             "hit_at_3": hit, "mrr_at_3": mrr, "n_chunks": len(records),
                             "encode_sec": encode_s, "search_sec": search_s})
                print(f"{strategy:<12} {backend:<10} {split:<6} {hit:>7.3f} {mrr:>7.3f} "
                      f"{len(records):>7} {encode_s:>9.3f} {search_s:>9.3f}")

            print_top3(strategy, backend, matrix, records, queries, logs, failures)
    print("MARKER_METRICS_END")

    print("")
    print("MARKER_SELECTION_START")
    best = pick_best_dev(rows)
    chosen_strategy, chosen_backend = best["strategy"], best["backend"]
    print("Selected from DEV only:", chosen_strategy, "+", chosen_backend,
          f"Hit@3={best['hit_at_3']:.3f}", f"MRR@3={best['mrr_at_3']:.3f}")
    for r in rows:
        if (r["split"], r["strategy"], r["backend"]) == ("test", chosen_strategy, chosen_backend):
            print(f"Held-out TEST for selected config: Hit@3={r['hit_at_3']:.3f}",
                  f"MRR@3={r['mrr_at_3']:.3f}")
    print("MARKER_SELECTION_END")
    print("")

    print("MARKER_FAILURES_START")
    print("Automatic failure list (answerable misses in top-3):")
    for n, f in enumerate(failures[:8], 1):
        print("FAIL", n)
        print("  config:", f["strategy"], f["backend"])
        print("  query:", f["query"])
        print("  expected_doc:", f["expected_doc"])
        print("  retrieved_rank:", f["rank"], "id=", f["retrieved_id"], "doc=", f["retrieved_doc"])
        print("  retrieved_text:", shorten(f["retrieved_text"] or "", 180))
    if not failures:
        print("No answerable misses found in this run.")
    print("MARKER_FAILURES_END")

    if logs:
        with open("retrieval_log.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(logs[0].keys()))
            writer.writeheader()
            writer.writerows(logs)
        print("Wrote retrieval_log.csv rows=", len(logs))

    with open("lab6_metrics.json", "w", encoding="utf-8") as fh:
        json.dump({"metrics": rows, "selected": best, "failures": failures[:8]}, fh, indent=2)

    print("")
    print("=" * 60)
    print("TASK 3: Document chatbot demos (selected config)")
    print("=" * 60)
    print("MARKER_DOC_CHAT_START")
    chunks = make_chunks(DOCS, chosen_strategy)
    engine = Retriever(chunks, backend=chosen_backend,
                       model=model if chosen_backend == "embedding" else None)
    demo_document_chatbot(engine)
    print("MARKER_DOC_CHAT_END")
    print("")
    return chosen_strategy, chosen_backend


def run_chatbot(model, strategy, backend):
    """Interactive document Q&A: shows candidates, the user picks one, nothing is asserted for them."""
    print("=" * 60)
    print("TASK 3 Chatbot: Document Q&A (" + strategy + " + " + backend + ")")
    print("=" * 60)
    print("Type a question about university policies.")
    print("Type 'exit' to stop.")
    print("Candidates are evidence to inspect, not verified answers.")
    print("")

    engine = Retriever(make_chunks(DOCS, strategy), backend=backend,
                       model=model if backend == "embedding" else None)
    while True:
        try:
            question = input("Question: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chatbot.")
            return
        if question.strip().lower() == "exit":
            print("Goodbye.")
            return

        hits = engine.search(question, k=TOP_K)
        if not hits:
            print("No candidate evidence; rephrase or contact a human.")
            continue

        print("Candidate evidence; relevance must be checked:")
        for n, (rec, score) in enumerate(hits, 1):
            print(n, rec["id"], f"{score:.3f}", "characters", rec["start"], rec["end"])
            print("   ", shorten(rec["text"], PREVIEW_CHARS))

        pick = input("Choose 1-3 to inspect; Enter if none apply: ").strip()
        if not (pick.isdigit() and 1 <= int(pick) <= len(hits)):
            print("No answer selected. Rephrase or contact a human.")
            continue
        rec, _ = hits[int(pick) - 1]
        print("Retrieved passage:", rec["text"])
        print("Source:", rec["doc"], "characters", rec["start"], rec["end"])
        print("")


# Main

def main():
    print("Lab 6 run")
    print("Python:", sys.version.replace("\n", " "))
    print("numpy:", np.__version__)
    print("scikit-learn:", sklearn.__version__)
    print("sentence-transformers:", sentence_transformers.__version__)
    print("MARKER_VERSIONS_START")
    print("python=" + sys.version.split()[0])
    print("numpy=" + np.__version__)
    print("scikit-learn=" + sklearn.__version__)
    print("sentence-transformers=" + sentence_transformers.__version__)
    print("MARKER_VERSIONS_END")
    print("")

    print("MARKER_TIMING_START")
    t0 = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    print("Model load seconds:", round(time.perf_counter() - t0, 3))
    print("Model max_seq_length:", model.max_seq_length)
    print("MARKER_TIMING_END")
    print("")

    run_task1(model)
    run_task2(model)
    strategy, backend = run_task3(model)

    if sys.stdin.isatty():
        print("Opening FAQ chatbot next (type exit when finished).")
        run_faq_chatbot_interactive()
        print("Opening document chatbot next (type exit when finished).")
        run_chatbot(model, strategy, backend)
    else:
        print("Skipping interactive chatbots (no TTY). Scripted demos are already in run_output.txt.")
    print("Done. Full transcript saved to run_output.txt")


if __name__ == "__main__":
    with RunLog("run_output.txt"):
        main()