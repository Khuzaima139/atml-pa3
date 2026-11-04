"""VQA answer normalization (decision D2): the steps of the official evaluator
GT-Vision-Lab/VQA PythonEvaluationTools/vqaEvaluation/vqaEval.py @ a013f0043c1e2cdc995922dfe257f7149aa9af06
(L21-66 tables, L88-94 whitespace, L130-155 processPunctuation / processDigitArticle).

normalize() is applied unconditionally to references and predictions everywhere (selection, NLL targets,
scoring), as manual L369 and L444-447 ask. The official code (L98-103) applies processPunctuation and
processDigitArticle only when the ten stripped references are not all identical; that condition is
deliberately not reproduced.

Quirks kept from the official code: a punctuation mark is deleted (not spaced) when it touches a space
anywhere in the input or the input has digit,digit; the period regex "(?!<=\\d)" is a lookahead typo, so
"2." -> "2" while "3.5" is kept; re.UNICODE (=32) is passed as the count of period deletions; the
capitalised contraction keys can never match after lowercasing.
"""
import re

CONTRACTIONS = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've",
    "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've",
    "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
    "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
    "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've",
    "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't",
    "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're",
    "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've",
    "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've",
    "youll": "you'll", "youre": "you're", "youve": "you've"}
MANUAL_MAP = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
              "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
ARTICLES = ["a", "an", "the"]
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
PUNCT = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]


def process_punctuation(in_text):
    """Official processPunctuation (L130-140)."""
    out = in_text
    for p in PUNCT:
        if (p + " " in in_text or " " + p in in_text) or (re.search(COMMA_STRIP, in_text) is not None):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    return PERIOD_STRIP.sub("", out, count=int(re.UNICODE))  # official passes re.UNICODE (=32) as the count


def process_digit_article(in_text):
    """Official processDigitArticle (L142-155): lowercase, number words, articles, contractions."""
    words = []
    for w in in_text.lower().split():
        w = MANUAL_MAP.get(w, w)
        if w not in ARTICLES:
            words.append(w)
    return " ".join(CONTRACTIONS.get(w, w) for w in words)


def normalize(s):
    """Whitespace (official L88-94), then processPunctuation, then processDigitArticle."""
    s = s.replace("\n", " ").replace("\t", " ").strip()
    return process_digit_article(process_punctuation(s))
