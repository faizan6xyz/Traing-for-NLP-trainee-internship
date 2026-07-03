
import re
import string
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "you're",
    "you've", "you'll", "you'd", "your", "yours", "yourself", "yourselves", "he",
    "him", "his", "himself", "she", "she's", "her", "hers", "herself", "it", "it's",
    "its", "itself", "they", "them", "their", "theirs", "themselves", "what",
    "which", "who", "whom", "this", "that", "that'll", "these", "those", "am",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just", "don", "don't", "should", "should've",
    "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "aren't", "couldn",
    "couldn't", "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn",
    "hasn't", "haven", "haven't", "isn", "isn't", "ma", "mightn", "mightn't",
    "mustn", "mustn't", "needn", "needn't", "shan", "shan't", "shouldn",
    "shouldn't", "wasn", "wasn't", "weren", "weren't", "won", "won't", "wouldn",
    "wouldn't",
}
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_HTML_RE = re.compile(r"<.*?>")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")
# Very small negation-aware suffix stripper (a lightweight stand-in for a
# full lemmatizer, e.g. WordNet, without needing a corpus download).
_SUFFIXES = ["ing", "edly", "edness", "ed", "ly", "es", "s"]
def _simple_stem(word: str) -> str:
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(r"\1", text)          # keep the word, drop the '#'
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text
def tokenize(text: str) -> list:
    return text.split() if text else []
def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]
def stem_tokens(tokens: list) -> list:
    return [_simple_stem(t) for t in tokens]
def preprocess_text(text: str, do_stem: bool = True) -> list:
    tokens = tokenize(clean_text(text))
    tokens = remove_stopwords(tokens)
    if do_stem:
        tokens = stem_tokens(tokens)
    return tokens
def preprocess_to_string(text: str, do_stem: bool = True) -> str:
    return " ".join(preprocess_text(text, do_stem=do_stem))
if __name__ == "__main__":
    sample = "I LOVED this product!! Check it out at https://shop.com @store #amazing 😊"
    print("Original :", sample)
    print("Cleaned  :", clean_text(sample))
    print("Tokens   :", preprocess_text(sample))
    print("Rejoined :", preprocess_to_string(sample))
