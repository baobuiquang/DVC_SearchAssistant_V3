# NLPT - Natural-Language-Processing Toolkits

from huggingface_hub import hf_hub_download as HF_Download
import re

# ====================================================================================================
# ====================================== Process_NLPT_Normalize ======================================
# ====================================================================================================

# ---------- Supported Languages: vi, en ----------

with open(HF_Download(repo_id="onelevelstudio/dataset", filename="nlp/diacritics_vi.txt"), mode="r", encoding="utf-8") as f:
    DATASET_diacritics_vi = f.read().splitlines()

def NLPT_Normalize(text, lower=False, remove_diacritics=False, replace_spacelikes_with_1space=False):
    vi_diacritics, vi_diacritics_normalized = DATASET_diacritics_vi
    translation_table = str.maketrans(vi_diacritics, vi_diacritics_normalized)
    # ----------
    if replace_spacelikes_with_1space:
        text = re.sub(r'\s+', ' ', text).strip()             # Replace consecutive spacelikes with single space
    if remove_diacritics:
        text = text.translate(translation_table)             # Remove Vietnamese diacritics
    if lower:
        text = text.lower()                                  # Lower
    # ----------
    return text.strip()

# ====================================================================================================
# ====================================== Process_NLPT_Tokenize =======================================
# ====================================================================================================

# ---------- Supported Languages: vi, en ----------

with open(HF_Download(repo_id="onelevelstudio/dataset", filename="nlp/punctuation.txt"), mode="r", encoding="utf-8") as f:
    DATASET_punctuation = f.read().splitlines()
with open(HF_Download(repo_id="onelevelstudio/dataset", filename="nlp/words_vi.txt"), mode="r", encoding="utf-8") as f:
    DATASET_vocab_vi = f.read().splitlines()
    DATASET_vocab_en = [] # No need, just split the text into words
with open(HF_Download(repo_id="onelevelstudio/dataset", filename="nlp/stopwords_vi.txt"), mode="r", encoding="utf-8") as f:
    DATASET_stopwords_vi = f.read().splitlines()
with open(HF_Download(repo_id="onelevelstudio/dataset", filename="nlp/stopwords_en.txt"), mode="r", encoding="utf-8") as f:
    DATASET_stopwords_en = f.read().splitlines()

DATASET_PUNCTUATION = set(DATASET_punctuation)
DATASET_VOCAB       = set(DATASET_vocab_vi + DATASET_vocab_en)
DATASET_STOPWORDS   = set(DATASET_stopwords_vi + DATASET_stopwords_en)

# Text -> ["token", "token"]
def NLPT_Tokenize(text):
    text = text.lower().strip()
    # tokens = re.findall(r'\w+|[^\w\s]', text)
    tokens = re.findall(r"\b\w+(?:'\w+)*(?:-\w+)*(?:\.\w+)*\b|[^\w\s]", text)
    res = []
    i = 0
    n = len(tokens)
    while i < n:
        max_match = 1  # Track the longest matching phrase
        matched_word = tokens[i]  # Default: single token
        for j in range(2, min(9, n - i) + 1):
            phrase = " ".join(tokens[i:i+j])
            if phrase in DATASET_VOCAB:                     # Include vocab
                max_match = j
                matched_word = phrase
        res.append(matched_word)                            # If not in vocab
        i += max_match  # Move the index by max_match
    res = [e for e in res if e not in DATASET_PUNCTUATION]  # Remove punctuation
    res = [e for e in res if e not in DATASET_STOPWORDS]    # Remove stopwords
    return res

# # ---------- Test ----------
# queries = [
# ".i'm playing video games! i her she state-of-the-art S.O.T.A.",
# "thuận vợ thuận chồng tát biển đông cũng cạn. đúng thế.",
# "ccc\tcc\ncc asd ,,,...",
# "Giấy tờ cần thiết để mình khởi nghiệp.",
# "Tôi muốn thành lập công ty tnhh 3 thành viên",
# "Để mua đất tôi cần giấy tờ nào?",
# "Xây nhà để ở cần làm thủ tục gì?",
# "Làm sao để cưới vợ?",
# "Mình muốn cưới chồng người nước ngoài",
# "Vợ tôi sắp sinh con tôi cần làm gì?",
# "Tôi muốn tố cáo hàng xóm trồng cần sa.",
# "Cháu muốn phúc khảo bài thi thpt của cháu",
# ]
# for q in queries:
#     print(Process_Text_Tokenize(q))
# # ['playing', 'video', 'games', 'state-of-the-art', 's.o.t.a']
# # ['thuận vợ thuận chồng', 'tát', 'biển', 'đông', 'cạn', 'đúng']
# # ['ccc', 'cc', 'cc', 'asd']
# # ['giấy tờ', 'cần thiết', 'khởi nghiệp']
# # ['thành lập', 'công ty', 'tnhh', '3', 'thành viên']
# # ['mua', 'đất', 'giấy tờ']
# # ['xây', 'nhà', 'ở', 'làm', 'thủ tục']
# # ['cưới', 'vợ']
# # ['cưới', 'chồng', 'người', 'nước ngoài']
# # ['vợ', 'sinh', 'con']
# # ['tố cáo', 'hàng xóm', 'trồng', 'cần sa']
# # ['cháu', 'phúc khảo', 'bài thi', 'thpt', 'cháu']

# ====================================================================================================
# ====================================================================================================
# ====================================================================================================