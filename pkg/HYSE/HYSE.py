# HYSE - HYbrid SEarch: Lexical-based search (BM25) + Semantic Search (SentenceTransformers)

# ====================================================================================================
# ====================================================================================================
# ====================================================================================================

_TEST_PASSAGES = ["Thủ tục thành lập công ty tư nhân", "Thủ tục đăng ký kết hôn", "Thủ tục chuyển nhượng quyền sử dụng đất", "Thủ tục đấu thầu đất xây dựng", "Thủ tục cấp lại lý lịch tư pháp", "Thủ tục chuyển trường cho học sinh trung học phổ thông", "Thủ tục chuyển trường cho học sinh trung học cơ sở", "Thủ tục chuyển trường cho học sinh tiểu học", "Thủ tục đăng ký lại kết hôn", "Thủ tục đăng ký kết hôn có yếu tố nước ngoài", "Thủ tục làm giấy khai sinh", "Thủ tục thành lập công ty trách nhiệm hữu hạn 1 thành viên", "Thủ tục thành lập công ty trách nhiệm hữu hạn 2 thành viên trở lên", "Thủ tục tố cáo tại cấp xã", "Thủ tục tố cáo tại cấp tỉnh"]
_TEST_QUERIES = [
    "Chuyển Trường", 
    "Chuyen Truong", 
    "Khai Sinh", 
    "Cháu muốn chuyển trường cấp 3 thì cần phải làm gì?", 
    "Tôi muốn mở công ty thì thủ tục gì?", 
    "khởi nghiệp", 
    "Sắp cưới vợ cần làm gì?", 
    "đăng ký kết hôn", 
    " \t\t\n\n dĂng  kÝ  kÊt \n\t\n  hoN\n        "
]

# ====================================================================================================
# ====================================================================================================
# ====================================================================================================

from huggingface_hub import hf_hub_download as HF_Download
from tokenizers import Tokenizer as STL_Tokenizer
from rank_bm25 import BM25Okapi as BM25_Retriever
import onnxruntime as ort
import numpy as np
import json
import os
from pkg.NLPT.NLPT import NLPT_Tokenize, NLPT_Normalize
os.makedirs("_hyse", exist_ok=True)

def dict2json(dict, jsonpath):
    try:
        with open(jsonpath, "w", encoding="utf-8") as f:
            json.dump(dict, f, ensure_ascii=False, indent=4)
    except Exception as er:
        print(f"⚠️ dict2json > Error: {er}")

def json2dict(jsonpath):
    dict = {}
    try:
        with open(jsonpath, "r", encoding="utf-8") as f:
            dict = json.load(f)
    except Exception as er:
        print(f"⚠️ json2dict > Error: {er}")
    return dict

def list2batches(ls, batch_size=5):
    try:
        return [ls[i:i + batch_size] for i in range(0, len(ls), batch_size)]
    except Exception as er:
        print(f"⚠️ list2batches > Error: {er}")
    
class SentenceTransformerLite:
    # Init: model_path -> model + tokenizer
    def __init__(self, model_path="onelevelstudio/ML-E5-0.3B"):
        try:
            # Model (ONNX)
            try: HF_Download(repo_id=model_path, filename="onnx/model.onnx_data")
            except: pass
            STL_model = ort.InferenceSession(HF_Download(repo_id=model_path, filename="onnx/model.onnx"))
            # Tokenizer
            STL_tokenizer = STL_Tokenizer.from_pretrained(model_path)
            STL_tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
            STL_tokenizer.enable_truncation(max_length=512)
        except Exception as er:
            raise ValueError(f"⚠️ > SentenceTransformerLite > init > Error: {er}")
        # Return
        self.STL_model = STL_model
        self.STL_tokenizer = STL_tokenizer
    # Encode: Text(s) -> Embedding(s)
    def encode(self, inputtexts):
        # Ensure inputtexts is a list of strings
        if isinstance(inputtexts, list) and all(isinstance(e, str) for e in inputtexts):
            if len(inputtexts) == 0:
                raise ValueError(f"⚠️ > SentenceTransformerLite > encode > inputtexts = empty list []")
        elif isinstance(inputtexts, str):
            inputtexts = [inputtexts]
        else:
            raise ValueError(f"⚠️ > SentenceTransformerLite > encode > inputtexts != string or list of strings")
        # Tokenize
        inputs = self.STL_tokenizer.encode_batch(inputtexts, is_pretokenized=False)
        inputs_ids = np.array([e.ids for e in inputs], dtype=np.int64)
        inputs_msk = np.array([e.attention_mask for e in inputs], dtype=np.int64)
        # Encoding
        embeddings = self.STL_model.run(None, {"input_ids": inputs_ids, "attention_mask": inputs_msk})[0]                                             # Encode
        embeddings = np.sum(embeddings * np.expand_dims(inputs_msk, axis=-1), axis=1) / np.maximum(np.sum(inputs_msk, axis=1, keepdims=True), 1e-9)   # Pooling
        embeddings = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-9)                                                 # Normalize
        # Return
        return embeddings

# ====================================================================================================
# ====================================================================================================
# ====================================================================================================

def queries_preprocessing(new_queries):
    return [NLPT_Normalize(e, replace_spacelikes_with_1space=True) for e in new_queries] # 🍌 Opinionated input queries pre-processing

class HYSE_EngineSemantic:
    # # ----- Example -----
    # engine_semantic = HYSE_EngineSemantic(name="a", modelpath="onelevelstudio/ML-E5-0.3B")
    # engine_semantic.update(_TEST_PASSAGES)
    # engine_semantic.search(_TEST_QUERIES)
    # engine_semantic = HYSE_EngineSemantic(name="b", modelpath="onelevelstudio/MPNET-0.3B")
    # engine_semantic.update(_TEST_PASSAGES)
    # engine_semantic.search(_TEST_QUERIES)
    # # -------------------
    def __init__(self, name="hyse001_sem1", modelpath="onelevelstudio/ML-E5-0.3B", stl_encoding_batch_size=100):
        self.name = name
        self.modelpath = modelpath
        self.stl_encoding_batch_size = stl_encoding_batch_size
        self.savepath_docs = f"_hyse/{name}_docs.json"
        self.savepath_embs = f"_hyse/{name}_embs.npy"
        self.model = SentenceTransformerLite(modelpath)
        # ----------
        self.docs = []
        self.embs = []
        if os.path.exists(self.savepath_docs) and os.path.exists(self.savepath_embs):
            self.docs = json2dict(self.savepath_docs)["docs"]  # 📤 Read file as docs
            self.embs = np.load(self.savepath_embs)            # 📤 Read file as embs
    def update(self, new_docs):
        if self.docs == new_docs:
            print(f"> HYSE_EngineSemantic '{self.name}' > No new docs")
            pass
        else:
            print(f"> HYSE_EngineSemantic '{self.name}' > Updating new docs...")
            self.docs = new_docs
            # ----------
            # ---------- ✨ STL Encoding: list of strings -> list of embs
            # ----------
            # self.embs = self.model.encode(self.docs)                                          # Original - will suffer when the self.docs is too large
            docs_batches = list2batches(self.docs, batch_size=self.stl_encoding_batch_size)
            embs_batches = []
            for iii, docsbatch in enumerate(docs_batches):
                print(f"> HYSE_EngineSemantic '{self.name}' > Encoding batch {iii+1}/{len(docs_batches)}...")
                embsbatch = self.model.encode(docsbatch)
                embs_batches.append(embsbatch)
            self.embs = np.concatenate(embs_batches, axis=0)
            # ----------
            # ----------
            # ----------
            dict2json({"docs": self.docs}, self.savepath_docs) # 📥 Save docs as file
            np.save(self.savepath_embs, self.embs)             # 📥 Save embs as file
    def search(self, new_queries, top=5):
        new_queries = queries_preprocessing(new_queries)
        # -----
        embs_queries = self.model.encode(new_queries)
        # -----
        similarities = embs_queries @ self.embs.T
        best_matching_idxs = [[idx for idx, _ in sorted(enumerate(sim), key=lambda x: x[1], reverse=True)][:min(top, len(self.docs))] for sim in similarities]
        best_matching_docs = [[self.docs[idx] for idx in e] for e in best_matching_idxs]
        best_matching_similarities = [[similarities[i][idx] for idx in idxs] for i, idxs in enumerate(best_matching_idxs)]
        # -----
        res = [[{"index": ee[0], "doc": ee[1], "score": round(float(ee[2]), 3)} for ee in zip(e[0], e[1], e[2])] for e in zip(best_matching_idxs, best_matching_docs, best_matching_similarities)]
        return [[e for e in q if e["score"] > 0] for q in res]

class HYSE_EngineLexical:
    # # ----- Example -----
    # engine_lexical = HYSE_EngineLexical(name="c")
    # engine_lexical.update(_TEST_PASSAGES)
    # engine_lexical.search(_TEST_QUERIES)
    # # -------------------
    def __init__(self, name="hyse001_lex1"):
        self.name = name
        self.savepath_docs = f"_hyse/{name}_docs.json"
        self.savepath_embs = f"_hyse/{name}_embs.json"
        # ----------
        self.docs = []
        self.embs = []
        self.model = None
        if os.path.exists(self.savepath_docs) and os.path.exists(self.savepath_embs):
            self.docs = json2dict(self.savepath_docs)["docs"]  # 📤 Read file as docs
            self.embs = json2dict(self.savepath_embs)["embs"]  # 📤 Read file as embs
            self.model = BM25_Retriever(self.embs)
    def update(self, new_docs):
        if self.docs == new_docs:
            print(f"> HYSE_EngineLexical '{self.name}' > No new docs")
            pass
        else:
            print(f"> HYSE_EngineLexical '{self.name}' > Updating new docs...")
            self.docs = new_docs
            self.embs = [NLPT_Tokenize(e) for e in self.docs]
            self.model = BM25_Retriever(self.embs)
            dict2json({"docs": self.docs}, self.savepath_docs) # 📥 Save docs as file
            dict2json({"embs": self.embs}, self.savepath_embs) # 📥 Save embs as file
    def search(self, new_queries, top=5):
        new_queries = queries_preprocessing(new_queries)
        # -----
        queries_embs = [NLPT_Tokenize(e) for e in new_queries]
        # -----
        similarities = [self.model.get_scores(query_emb) for query_emb in queries_embs]
        best_matching_idxs = [self.model.get_top_n(query_emb, range(len(self.docs)), n=top) for query_emb in queries_embs]
        best_matching_docs = [[self.docs[idx] for idx in e] for e in best_matching_idxs]
        best_matching_similarities = [[similarities[i][idx] for idx in idxs] for i, idxs in enumerate(best_matching_idxs)]
        # -----
        res = [[{"index": ee[0], "doc": ee[1], "score": round(float(ee[2]), 3)} for ee in zip(e[0], e[1], e[2])] for e in zip(best_matching_idxs, best_matching_docs, best_matching_similarities)]
        return [[e for e in q if e["score"] > 0] for q in res]

class HYSE_EngineExactMatch:
    # # ----- Example -----
    # engine_exactmatch = HYSE_EngineExactMatch(name="d")
    # engine_exactmatch.update(_TEST_PASSAGES)
    # engine_exactmatch.search(_TEST_QUERIES)
    # # -------------------
    def __init__(self, name="hyse001_exa1"):
        self.name = name
        self.savepath_docs = f"_hyse/{name}_docs.json"
        # ----------
        self.docs = []
        if os.path.exists(self.savepath_docs):
            self.docs = json2dict(self.savepath_docs)["docs"]  # 📤 Read file as docs
    def update(self, new_docs):
        if self.docs == new_docs:
            print(f"> HYSE_EngineExactMatch '{self.name}' > No new docs")
            pass
        else:
            print(f"> HYSE_EngineExactMatch '{self.name}' > Updating new docs...")
            self.docs = new_docs
            dict2json({"docs": self.docs}, self.savepath_docs) # 📥 Save docs as file
    def search(self, new_queries):
        new_queries = queries_preprocessing(new_queries)
        # -----
        best_matching_idxs = []
        for q in new_queries:
            # Exact match with diacritics
            tmp_idxs = [i for i, d in enumerate(self.docs) if NLPT_Normalize(q, lower=True) in NLPT_Normalize(d, lower=True)]
            if len(tmp_idxs) == 0:
                # Exact match without diacritics
                tmp_idxs = [i for i, d in enumerate(self.docs) if NLPT_Normalize(q, lower=True, remove_diacritics=True) in NLPT_Normalize(d, lower=True, remove_diacritics=True)]
            best_matching_idxs.append(tmp_idxs)
        best_matching_docs = [[self.docs[idx] for idx in e] for e in best_matching_idxs]
        best_matching_similarities = [[round(len(new_queries[qidx])/len(doc), 3) for doc in e] for qidx, e in enumerate(best_matching_docs)]
        # -----
        res = [[{"index": ee[0], "doc": ee[1], "score": round(float(ee[2]), 3)} for ee in zip(e[0], e[1], e[2])] for e in zip(best_matching_idxs, best_matching_docs, best_matching_similarities)]
        return [[e for e in q if e["score"] > 0] for q in res]

class HYSE_EngineHybrid:
    # # ----- Example -----
    # hyse_engine = HYSE_EngineHybrid(name="e")
    # hyse_engine.update(_TEST_PASSAGES)
    # hyse_engine.search(_TEST_QUERIES)
    # # -------------------
    def __init__(self, name="HYSE1"):
        self.search_engine_1 = HYSE_EngineExactMatch(name=f"{name}_EXA1")
        self.search_engine_2 = HYSE_EngineLexical(name=f"{name}_LEX1")
        self.search_engine_3 = HYSE_EngineSemantic(name=f"{name}_SEM1", modelpath="onelevelstudio/ML-E5-0.3B")
        self.search_engine_4 = HYSE_EngineSemantic(name=f"{name}_SEM2", modelpath="onelevelstudio/MPNET-0.3B")
    def update(self, new_docs):
        self.search_engine_1.update(new_docs)
        self.search_engine_2.update(new_docs)
        self.search_engine_3.update(new_docs)
        self.search_engine_4.update(new_docs)
    def search(self, new_queries):
        res_1 = self.search_engine_1.search(new_queries)
        res_2 = self.search_engine_2.search(new_queries)
        res_3 = self.search_engine_3.search(new_queries)
        res_4 = self.search_engine_4.search(new_queries)
        # -----
        res_search = []
        for i in range(len(res_1)):
            # ---------- Case 1: Exact match
            if len(res_1[i]) > 0:
                tmp_res = res_1[i]
            # ---------- Case 2: Not exact match
            else:
                res_234 = res_2[i]+res_3[i]+res_4[i]
                res_234 = [{"index": e["index"], "doc": e["doc"]} for e in res_234]
                doc_counts = {}
                for item in res_234: doc_counts[item['doc']] = doc_counts.get(item['doc'], 0) + 1
                tmp_res = [{'index': item['index'], 'doc': doc, 'score': doc_counts[doc]} for doc, item in {d['doc']: d for d in res_234}.items()]
            # ----------
            tmp_res = sorted(tmp_res, key=lambda x: (-x['score'], len(x['doc']))) # Sort by score (higher first), then sort by length (shorter first)
            res_search.append(tmp_res)
        return res_search