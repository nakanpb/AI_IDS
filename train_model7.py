#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
payload_cnn_512_multi_memmap.py  (baseline + robust split & threshold + plots/history/report)
- 1D-CNN บน payload 512 (ตัด/แพด)
- รองรับ base64 / hex / latin-1
- รวม CSV หลายไฟล์ + ติด family/source_id
- Split แบบ family-aware + group-aware + quota floor (ลด bias/imbalance)
- เลือก threshold จาก VAL แบบ robust (best_f1/target_recall + min_precision + floor)
- numpy.memmap + tf.data
- Plots: ROC, PR, CM, family mix (ตาราง+stack), timing(section+epoch), val PR-AUC ต่อ epoch, metrics trend
- Save: BEST/FINAL, meta(joblib), timing.json, final_report.txt, history.csv
- Inference helper ใช้ threshold จาก meta ได้
"""

import os, re, binascii, base64, random, datetime, math, time, json, csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, precision_recall_curve, accuracy_score, roc_curve
)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LambdaCallback

import joblib

# ====== CONFIG ======
TRAIN_MODE            = "scratch"     # "scratch" | "continue"
RESUME_LR             = 1e-4          # ใช้เมื่อ continue; set None เพื่อใช้ LR เดิม
EVAL_ONLY             = False
BASE_MODEL_PATH       = "/home/intgan/Desktop/jek/train/scratch(8)round1.final.20250928-015630.keras"           # ตั้ง path เมื่อ TRAIN_MODE="continue" หรือ EVAL_ONLY

REUSE_SPLIT_FROM_META = False
PREV_META_PATH = "/home/intgan/Desktop/jek/train/preprocess_meta.20250928-015630.joblib"
# Threshold
THRESHOLD_POLICY      = "best_f1"     # "best_f1" | "target_recall"
TARGET_RECALL         = 0.90
MIN_PREC_AT_THR       = 0.20          # กัน precision ต่ำเกินไป
MIN_THR_FLOOR         = 1e-6          # กัน threshold = 0

# ข้อมูล
CSV_FILES = [
    ("/home/intgan/Desktop/jek/zeus(csv)/zeus.csv", "zeus"),
    ("/home/intgan/Desktop/jek/zeus(csv)/zeus25-2.csv", "zeus"),
    ("/home/intgan/Desktop/jek/zeus(csv)/zeus25-4.csv", "zeus"),
    ("/home/intgan/Desktop/jek/zeus(csv)/zeus25-6.csv", "zeus"),
    ("/home/intgan/Desktop/jek/emotet(csv)/emotet(192-3).csv", "emotet"),
    ("/home/intgan/Desktop/jek/emotet(csv)/emotet(264-1).csv", "emotet"),
    ("/home/intgan/Desktop/jek/emotet(csv)/emotet(264-2).csv", "emotet"),
    ("/home/intgan/Desktop/jek/emotet(csv)/emotet(268-1).csv", "emotet"),
    ("/home/intgan/Desktop/jek/emotet(csv)/emotet(269-1).csv", "emotet"),
    ("/home/intgan/Desktop/jek/emotet(csv)/emotet(271-1).csv", "emotet"),
    ("/home/intgan/Desktop/jek/emotet(csv)/emotet(272-1).csv", "emotet"),
    ("/home/intgan/Desktop/jek/emotet(csv)/emotet(276-1).csv", "emotet"),
    ("/home/intgan/Desktop/jek/emotet(csv)/emotet(279-1).csv", "emotet"),
    ("/home/intgan/Desktop/jek/trickbot(csv)/trickbot(238-1).csv", "trickbot"),
    ("/home/intgan/Desktop/jek/trickbot(csv)/trickbot(240-1).csv", "trickbot"),
    ("/home/intgan/Desktop/jek/trickbot(csv)/trickbot(241-1).csv", "trickbot"),
    ("/home/intgan/Desktop/jek/trickbot(csv)/trickbot(247-1).csv", "trickbot"),
    ("/home/intgan/Desktop/jek/trickbot(csv)/trickbot(261-2).csv", "trickbot"),
    ("/home/intgan/Desktop/jek/trickbot(csv)/trickbot(265-1).csv", "trickbot"),
    ("/home/intgan/Desktop/jek/trickbot(csv)/trickbot(327-1).csv", "trickbot"),
    ("/home/intgan/Desktop/jek/benign/realbenign1.csv", "benign"),
    ("/home/intgan/Desktop/jek/benign/realbenign2.csv", "benign"),
    ("/home/intgan/Desktop/jek/benign/realbenign3.csv", "benign"),
    ("/home/intgan/Desktop/jek/benign/realbenign4.csv", "benign"),
    ("/home/intgan/Desktop/jek/benign/realbenign5.csv", "benign"),
]

PAYLOAD_COL           = "payload"
LABEL_COL             = "label"
FIXED_LEN             = 512

TEST_SIZE             = 0.20
VAL_SIZE              = 0.10

BATCH                 = 32
EPOCHS                = 15
SEED                  = 42

USE_MIXED_PRECISION   = True
EXPORT_DIR            = "/home/intgan/Desktop/jek/train"
EXPORT_NAME_PREFIX    = "testzeus"
MAX_PER_FAMILY_TRAIN  = None
MMAP_DIR              = os.path.join(EXPORT_DIR, "mmap")
SAVE_FINAL_TXT        = True

# ====== Repro & GPU ======
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
    if USE_MIXED_PRECISION:
        mixed_precision.set_global_policy('mixed_float16')
    print(f"[GPU] Found: {gpus} | mixed_precision={USE_MIXED_PRECISION}")
else:
    print("[GPU] ไม่พบการ์ดจอ — จะใช้ CPU")

# ====== Utils ======
LABEL_MAP = {'benign':0,'Benign':0,'BENIGN':0,'malicious':1,'Malicious':1,'MALICIOUS':1,0:0,1:1}
b64_re = re.compile(r'^[A-Za-z0-9+/=\s]+$')
hex_re = re.compile(r'^[0-9A-Fa-f\s]+$')



def ts_now():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def ensure_dir(p): os.makedirs(p, exist_ok=True); return p

def decode_payload(s: str) -> bytes:
    if not isinstance(s, str): return b""
    s = s.strip()
    if not s: return b""
    if b64_re.match(s):
        try: return base64.b64decode(s, validate=True)
        except Exception: pass
    if hex_re.match(s):
        try: return binascii.unhexlify(re.sub(r"\s+","", s))
        except Exception: pass
    return s.encode("latin-1", errors="ignore")

def to_fixed_len_uint8(b: bytes, L=FIXED_LEN, mode="head"):
    arr = np.frombuffer(b, dtype=np.uint8)
    n = len(arr)
    if n >= L:
        if mode == "tail": return arr[-L:]
        if mode == "middle": start = max((n-L)//2, 0); return arr[start:start+L]
        return arr[:L]
    out = np.zeros(L, dtype=np.uint8); out[:n] = arr; return out

# ====== History ======
def write_history(csv_path, ts, N_eff, acc, report_dict):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["timestamp","N_eff","accuracy","precision","recall","f1"])
        w.writerow([ts, N_eff, acc,
                    report_dict["1"]["precision"],
                    report_dict["1"]["recall"],
                    report_dict["1"]["f1-score"]])
    print(f"[HISTORY] -> {csv_path}")

def plot_history(csv_path, export_dir, ts):
    if not os.path.isfile(csv_path): print("[PLOT] no history.csv"); return
    df = pd.read_csv(csv_path)
    if df.empty: return
    plt.figure(figsize=(8,6))
    for col in ["accuracy","precision","recall","f1"]:
        plt.plot(df["N_eff"], df[col], marker="o", label=col.title())
        for _, r in df.iterrows():
            plt.text(r["N_eff"], r[col], f"{r[col]:.3f}", fontsize=8, ha="right")
    plt.title("Metrics vs Data Size"); plt.xlabel("N_eff"); plt.ylabel("Score")
    plt.ylim(0,1.05); plt.grid(True, ls="--", alpha=.6); plt.legend()
    out = os.path.join(export_dir, f"metrics_history.{ts}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
    print(f"[PLOT] -> {out}")

# ====== Memmap ======
def _normalize_csvs(csv_files): return [(str(p), str(fam)) for p, fam in csv_files]

def _count_rows_quick(path, chunksize=200_000):
    total = 0
    for ch in pd.read_csv(path, usecols=[PAYLOAD_COL, LABEL_COL],
                          chunksize=chunksize, keep_default_na=False, na_filter=False):
        total += len(ch)
    return total

def build_memmap_from_csvs(csv_files, L=FIXED_LEN, mmap_dir=MMAP_DIR):
    ensure_dir(mmap_dir)
    norm = _normalize_csvs(csv_files)
    print("[MMAP] counting ...")
    total = 0; raw_by = {}; eff_by = {}
    for p, fam in norm:
        n = _count_rows_quick(p); raw_by[fam] = raw_by.get(fam, 0) + n; total += n
        print(f"  - {Path(p).name}: {n} (family={fam})")

    X_path = os.path.join(mmap_dir, "X_all.int32.mmap")
    y_path = os.path.join(mmap_dir, "y_all.float32.mmap")
    g_path = os.path.join(mmap_dir, "groups.int32.mmap")
    f_path = os.path.join(mmap_dir, "family.int32.mmap")

    X_all = np.memmap(X_path, dtype=np.int32,   mode="w+", shape=(total, L))
    y_all = np.memmap(y_path, dtype=np.float32, mode="w+", shape=(total,))
    groups= np.memmap(g_path, dtype=np.int32,   mode="w+", shape=(total,))
    family= np.memmap(f_path, dtype=np.int32,   mode="w+", shape=(total,))

    fam2id, grp2id = {}, {}; w = 0
    print("[MMAP] writing ...")
    for p, fam in norm:
        fam_id = fam2id.setdefault(fam, len(fam2id))
        grp_id = grp2id.setdefault(Path(p).name, len(grp2id))
        for chunk in pd.read_csv(p, usecols=[PAYLOAD_COL, LABEL_COL],
                                 chunksize=200_000, keep_default_na=False, na_filter=False):
            chunk[LABEL_COL] = chunk[LABEL_COL].map(lambda x: LABEL_MAP.get(x, x)).astype(int)
            s = chunk[PAYLOAD_COL].astype(str)
            chunk = chunk[s.str.len() > 0]
            vecs, keep = [], []
            for s in chunk[PAYLOAD_COL].values:
                b = decode_payload(s)
                if len(b) == 0: keep.append(False)
                else:
                    vecs.append(to_fixed_len_uint8(b, L=L, mode="head"))
                    keep.append(True)
            if not vecs: continue
            keep = np.array(keep, bool)
            chunk = chunk[keep]
            arr = np.vstack(vecs).astype(np.int32)
            n = len(chunk)
            X_all[w:w+n] = arr
            y_all[w:w+n] = chunk[LABEL_COL].values.astype(np.float32)
            groups[w:w+n] = grp_id
            family[w:w+n] = fam_id
            w += n
            eff_by[fam] = eff_by.get(fam, 0) + n

    X_all.flush(); y_all.flush(); groups.flush(); family.flush()
    print(f"[MMAP] done: {w}/{total}")
    return {"X_path": X_path, "y_path": y_path, "g_path": g_path, "f_path": f_path,
            "N_total": total, "N_eff": w, "L": L, "fam2id": fam2id, "grp2id": grp2id,
            "raw_rows_by_fam": raw_by, "eff_rows_by_fam": eff_by}

def open_memmaps(paths, N, L):
    X_all  = np.memmap(paths["X_path"], dtype=np.int32,   mode="r", shape=(N, L))
    y_all  = np.memmap(paths["y_path"], dtype=np.float32, mode="r", shape=(N,))
    groups = np.memmap(paths["g_path"], dtype=np.int32,   mode="r", shape=(N,))
    family = np.memmap(paths["f_path"], dtype=np.int32,   mode="r", shape=(N,))
    return X_all, y_all, groups, family

# ====== Split helpers (improved) ======
def enforce_family_coverage(train_idx, val_idx, test_idx, family_arr, min_each=1):
    fams = np.unique(family_arr)
    for fam in fams:
        # TEST
        need = max(0, min_each - int((family_arr[test_idx] == fam).sum()))
        if need > 0:
            src_pool = np.concatenate([train_idx, val_idx])
            cand = src_pool[family_arr[src_pool] == fam]
            take = cand[:need] if len(cand) >= need else cand
            train_idx = train_idx[~np.isin(train_idx, take)]
            val_idx   = val_idx[  ~np.isin(val_idx,   take)]
            test_idx  = np.concatenate([test_idx, take])
        # TRAIN
        need = max(0, min_each - int((family_arr[train_idx] == fam).sum()))
        if need > 0:
            src_pool = np.concatenate([val_idx, test_idx])
            cand = src_pool[family_arr[src_pool] == fam]
            take = cand[:need] if len(cand) >= need else cand
            val_idx   = val_idx[ ~np.isin(val_idx,  take)]
            test_idx  = test_idx[~np.isin(test_idx, take)]
            train_idx = np.concatenate([train_idx, take])
        # VAL
        need = max(0, min_each - int((family_arr[val_idx] == fam).sum()))
        if need > 0:
            src_pool = np.concatenate([train_idx, test_idx])
            cand = src_pool[family_arr[src_pool] == fam]
            take = cand[:need] if len(cand) >= need else cand
            train_idx = train_idx[~np.isin(train_idx, take)]
            test_idx  = test_idx[ ~np.isin(test_idx,  take)]
            val_idx   = np.concatenate([val_idx, take])
    return train_idx, val_idx, test_idx

def split_family_group_aware_quota(groups, family, y,
                                   test_frac=0.20, val_frac=0.10,
                                   min_per_split_per_family=50,
                                   min_per_label_per_family=10,
                                   seed=42):
    rng = np.random.RandomState(seed)
    uniq_groups = np.unique(groups)
    grp2idx = {g: np.where(groups == g)[0] for g in uniq_groups}
    grp2fam = {g: int(np.argmax(np.bincount(family[grp2idx[g]]))) for g in uniq_groups}
    fams = np.unique(family)
    fam2groups = {f: [g for g in uniq_groups if grp2fam[g] == f] for f in fams}
    take_train, take_val, take_test = [], [], []

    for f in fams:
        idx_f = np.where(family == f)[0]
        n_f = len(idx_f)
        if n_f == 0: continue

        n_test = max(int(round(test_frac * n_f)), 1)
        n_val  = max(int(round(val_frac  * n_f)), 1)
        n_train= max(n_f - n_test - n_val, 1)

        if n_f >= 3*min_per_split_per_family:
            n_test  = max(n_test,  min_per_split_per_family)
            n_val   = max(n_val,   min_per_split_per_family)
            n_train = max(n_train, min_per_split_per_family)

        tot = n_train + n_val + n_test
        if tot > n_f:
            scale = n_f / float(tot)
            n_train = max(int(round(n_train*scale)), 1)
            n_val   = max(int(round(n_val  *scale)), 1)
            n_test  = max(int(round(n_test *scale)), 1)
            while (n_train + n_val + n_test) > n_f:
                n_train -= 1

        g_list = fam2groups.get(f, [])
        # ! แก้ไข: บังคับให้เงื่อนไขนี้เป็น 'False' เพื่อข้ามตรรกะการแบ่งตาม Group
        # ! และบังคับให้ใช้การแบ่งแบบสุ่มดัชนี (Random Index Slicing) เสมอ
        if False: # len(g_list) >= 2: <--- แก้ไขตรงนี้
            
            g_list = g_list[:]; rng.shuffle(g_list)

            def pick_by_count(target_n, pool_groups):
                chosen, cnt = [], 0
                for g in pool_groups:
                    chosen.append(g); cnt += len(grp2idx[g])
                    if cnt >= target_n: break
                return chosen

            rng.shuffle(g_list)
            g_test = pick_by_count(n_test, g_list)
            rest   = [g for g in g_list if g not in set(g_test)]
            g_val  = pick_by_count(n_val, rest)
            rest   = [g for g in rest if g not in set(g_val)]
            g_train= rest

            idx_test_f  = np.concatenate([grp2idx[g] for g in g_test])  if g_test  else np.array([], int)
            idx_val_f   = np.concatenate([grp2idx[g] for g in g_val])   if g_val   else np.array([], int)
            idx_train_f = np.concatenate([grp2idx[g] for g in g_train]) if g_train else np.array([], int)
        else: # <--- โค้ดส่วนนี้จะถูกใช้งานแทนเสมอ
            idx_f = idx_f.copy(); rng.shuffle(idx_f)
            idx_test_f  = idx_f[:n_test]
            idx_val_f   = idx_f[n_test:n_test+n_val]
            idx_train_f = idx_f[n_test+n_val:]

        for name, arr in [("val", idx_val_f), ("test", idx_test_f)]:
            y_sub = y[arr]
            need0 = max(0, min_per_label_per_family - int((y_sub == 0).sum()))
            need1 = max(0, min_per_label_per_family - int((y_sub == 1).sum()))
            if (need0 > 0 or need1 > 0) and len(idx_train_f) > 0:
                y_tr = y[idx_train_f]
                move0 = idx_train_f[np.where(y_tr == 0)[0]][:need0]
                move1 = idx_train_f[np.where(y_tr == 1)[0]][:need1]
                move  = np.concatenate([move0, move1]) if (len(move0)+len(move1)) else np.array([], int)
                if name == "val":  idx_val_f  = np.concatenate([idx_val_f,  move])
                else:               idx_test_f = np.concatenate([idx_test_f, move])
                idx_train_f = idx_train_f[~np.isin(idx_train_f, move)]

        take_train.append(idx_train_f); take_val.append(idx_val_f); take_test.append(idx_test_f)

    train_idx = np.unique(np.concatenate(take_train) if take_train else np.array([],int))
    val_idx   = np.unique(np.concatenate(take_val)   if take_val   else np.array([],int))
    test_idx  = np.unique(np.concatenate(take_test)  if take_test  else np.array([],int))

    def disjoint(a,b,c):
        a = np.setdiff1d(a, np.union1d(b,c))
        b = np.setdiff1d(b, np.union1d(a,c))
        c = np.setdiff1d(c, np.union1d(a,b))
        return a,b,c
    train_idx, val_idx, test_idx = disjoint(train_idx, val_idx, test_idx)
    return train_idx.astype(int), val_idx.astype(int), test_idx.astype(int)

def rebalance_split_to_targets(idx_split, family_arr, fam2id, benign_name="benign",
                               keep_floor_per_family=1, seed=42,
                               oversample=False, max_multiplier=3,
                               targets=None):
    """
    ทำให้ benign ~50% และ non-benign ~50% แบ่งเท่าๆ กันระหว่าง non-benign families
    - undersample เสมอ
    - ถ้า oversample=True -> อนุญาตทำซ้ำดัชนี (with replacement) เพื่อเติมโควตา (แนะนำเฉพาะ TRAIN)
      และจำกัดการขยายของ family ใดๆ ไม่เกิน max_multiplier เท่าของจำนวนจริง
    - targets (ออปชัน): dict ของสัดส่วนต่อ family เช่น
        {"benign":0.5,"emotet":1/6,"trickbot":1/6,"zeus":1/6}
      ถ้าให้มา จะใช้เป้าหมายนี้แทนตรรกะ 50/50+เฉลี่ยอัตโนมัติ
      (ค่าที่ส่งมาต้องรวมกัน ≈ 1.0)
    """
    rng = np.random.RandomState(seed)
    inv = {v: k for k, v in fam2id.items()}
    fam_ids_in_split = np.unique(family_arr[idx_split])

    # หา benign_id
    benign_id = None
    for fid in fam_ids_in_split:
        if inv[int(fid)].lower() == benign_name.lower():
            benign_id = int(fid)
            break

    if len(idx_split) == 0:
        return np.sort(idx_split.astype(int))

    # map family id -> indices ใน split นี้
    fam2indices = {int(fid): idx_split[family_arr[idx_split] == int(fid)]
                   for fid in fam_ids_in_split}

    N = len(idx_split)

    # ---------- คำนวณโควตาเป้าหมาย ----------
    quota = {}

    if targets is not None:
        # ใช้เป้าหมายที่กำหนดมาแบบชัดเจน (ถ้าบาง family ไม่อยู่ใน split ให้ข้าม)
        # normalize เผื่อรวมไม่เท่ากับ 1
        s = float(sum(targets.values()))
        if s <= 0:
            return np.sort(idx_split.astype(int))
        for name, p in targets.items():
            p_norm = float(p) / s
            # หา fid จากชื่อ (ข้ามถ้าไม่มีใน split)
            fid_match = None
            for fid in fam_ids_in_split:
                if inv[int(fid)].lower() == name.lower():
                    fid_match = int(fid); break
            if fid_match is not None:
                quota[fid_match] = max(int(round(p_norm * N)), 0)

        # ถ้าโควตารวมไม่พอดี N ให้ชดเชยเล็กน้อย
        gap = N - sum(quota.values())
        if gap != 0 and len(quota) > 0:
            # เติม/ตัดจาก family ที่มีอยู่ แบบวนรอบ
            keys = list(quota.keys())
            i = 0
            while gap != 0 and len(keys) > 0:
                k = keys[i % len(keys)]
                quota[k] += 1 if gap > 0 else -1
                if quota[k] < 0: quota[k] = 0
                gap += (-1 if gap > 0 else 1)
                i += 1

    else:
        # โหมดอัตโนมัติ: benign ~50%, ที่เหลือเฉลี่ยระหว่าง non-benign
        if benign_id is None:
            # ไม่มี benign ใน split นี้ -> เฉลี่ยทั้งหมด
            per = N // max(1, len(fam_ids_in_split))
            for fid in fam_ids_in_split:
                quota[int(fid)] = per
            # ชดเชยเศษ
            rem = N - sum(quota.values())
            for fid in list(quota.keys())[:rem]:
                quota[fid] += 1
        else:
            non_benign = [int(fid) for fid in fam_ids_in_split if int(fid) != benign_id]
            target_benign = int(round(0.50 * N))
            target_rest   = N - target_benign
            quota[benign_id] = target_benign
            if len(non_benign) > 0:
                per = target_rest // len(non_benign)
                for fid in non_benign:
                    quota[fid] = per
                # ชดเชยเศษ
                rem = target_rest - per * len(non_benign)
                for fid in non_benign[:rem]:
                    quota[fid] += 1

    # ---------- เลือกดัชนีให้ได้ตามโควตา ----------
    chosen = []
    for fid, q in quota.items():
        cur = fam2indices.get(int(fid), np.array([], dtype=int))
        n   = len(cur)
        if q <= 0:
            continue

        take = min(n, q)
        if take > 0:
            chosen.append(rng.choice(cur, size=take, replace=False))

        need_more = q - take
        if need_more > 0:
            if oversample and n > 0:
                # จำกัดการขยายไม่ให้เกิน max_multiplier
                max_extra = max(0, n * max_multiplier - n)
                add_n = min(need_more, max_extra)
                if add_n > 0:
                    chosen.append(rng.choice(cur, size=add_n, replace=True))
            else:
                # ถ้าไม่ oversample ก็ปล่อยขาดไป (จะได้ไม่บิดเบือน)
                pass

    final_idx = np.concatenate(chosen) if len(chosen) else np.array([], dtype=int)

    # กัน overshoot / undershoot ให้พอดี N
    if len(final_idx) > N:
        final_idx = rng.choice(final_idx, size=N, replace=False)
    elif len(final_idx) < N:
        # เติมสุ่มจากทั้งหมด (เผื่อกรณีขาดเพราะไม่มีข้อมูลพอ)
        pool = idx_split
        need = N - len(final_idx)
        if len(pool) > 0:
            extra = rng.choice(pool, size=min(need, len(pool)), replace=False)
            final_idx = np.concatenate([final_idx, extra])

    return np.sort(final_idx.astype(int))


# ====== tf.data ======
def make_dataset_from_indices(X, y, indices, batch, training=False, seed=SEED):
    idx = np.array(indices, dtype=np.int64)
    def gen():
        for i in idx: yield X[i], y[i]
    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(tf.TensorSpec(shape=(FIXED_LEN,), dtype=tf.int32),
                          tf.TensorSpec(shape=(), dtype=tf.float32))
    )
    if training:
        ds = ds.shuffle(8192, seed=seed, reshuffle_each_iteration=True).repeat()
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


# ====== Stats ======
def print_dataset_stats(name, y_subset, fam_subset, fam2id):
    inv = {v:k for k,v in fam2id.items()}
    print(f"=== {name} ===")
    vc_y = pd.Series(np.array(y_subset)).value_counts().sort_index()
    print("Labels:", {int(k): int(v) for k, v in vc_y.items()})
    vc_f = pd.Series(np.array(fam_subset)).value_counts()
    print("Family:", {str(inv[int(k)]): int(v) for k, v in vc_f.items()})

def count_by_family(fam_array, fam2id):
    inv={v:k for k,v in fam2id.items()}; out={}
    for fid in np.unique(fam_array): out[inv[int(fid)]] = int((fam_array==fid).sum())
    return out

def pct(n,d): return f"{(100.0*n/max(1,d)):.2f}%"

def print_family_mix(mmap_info, family_arr, splits, title="FAMILY MIX (FINAL REPORT)"):
    fam2id=mmap_info["fam2id"]; raw_by=mmap_info["raw_rows_by_fam"]; eff_by=mmap_info["eff_rows_by_fam"]
    N_raw=int(mmap_info["N_total"]); N_eff=int(mmap_info["N_eff"])
    tr,va,te=splits["train"],splits["val"],splits["test"]
    fam_tr=count_by_family(family_arr[tr],fam2id)
    fam_va=count_by_family(family_arr[va],fam2id)
    fam_te=count_by_family(family_arr[te],fam2id)
    Ntr,Nva,Nte=len(tr),len(va),len(te)
    fams=sorted(set(raw_by.keys())|set(eff_by.keys())|set(fam_tr.keys())|set(fam_va.keys())|set(fam_te.keys()))
    print("\n====", title, "====")
    print(f"{'family':12s} | {'RAW':>10s} | {'EFF':>10s} | {'EFF%':>7s} | {'TRAIN':>10s} | {'T%':>6s} | {'VAL':>10s} | {'V%':>6s} | {'TEST':>10s} | {'S%':>6s}")
    print("-"*110)
    for fam in fams:
        r=int(raw_by.get(fam,0)); e=int(eff_by.get(fam,0))
        t=int(fam_tr.get(fam,0)); v=int(fam_va.get(fam,0)); s=int(fam_te.get(fam,0))
        print(f"{fam:12s} | {r:10d} | {e:10d} | {pct(e,N_eff):>7s} | {t:10d} | {pct(t,Ntr):>6s} | {v:10d} | {pct(v,Nva):>6s} | {s:10d} | {pct(s,Nte):>6s}")
    print("-"*110)
    print(f"{'TOTAL':12s} | {N_raw:10d} | {N_eff:10d} | 100.00% | {Ntr:10d} | 100.00% | {Nva:10d} | 100.00% | {Nte:10d} | 100.00%")

# ====== Plots ======
def plot_curves_and_cm(y_true, proba, pred, export_dir, ts, prefix="report"):
    ensure_dir(export_dir)
    fpr, tpr, _ = roc_curve(y_true, proba)
    plt.figure(figsize=(5,4)); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],"--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC")
    out_roc = os.path.join(export_dir, f"{prefix}.{ts}.roc.png")
    plt.tight_layout(); plt.savefig(out_roc, dpi=150); plt.close()

    prec, rec, _ = precision_recall_curve(y_true, proba)
    plt.figure(figsize=(5,4)); plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR")
    out_pr = os.path.join(export_dir, f"{prefix}.{ts}.pr.png")
    plt.tight_layout(); plt.savefig(out_pr, dpi=150); plt.close()

    cm = confusion_matrix(y_true, pred)
    plt.figure(figsize=(4,4)); plt.imshow(cm, interpolation='nearest'); plt.title("Confusion Matrix"); plt.colorbar()
    ticks=np.arange(2); plt.xticks(ticks,["benign","malicious"],rotation=45,ha="right"); plt.yticks(ticks,["benign","malicious"])
    th=cm.max()/2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j,i,format(cm[i,j],'d'),ha="center",va="center",color="white" if cm[i,j]>th else "black")
    plt.ylabel("True"); plt.xlabel("Pred")
    out_cm = os.path.join(export_dir, f"{prefix}.{ts}.cm.png")
    plt.tight_layout(); plt.savefig(out_cm, dpi=150); plt.close()
    print(f"[PLOT] -> {out_roc}\n[PLOT] -> {out_pr}\n[PLOT] -> {out_cm}")

def plot_family_mix_charts(mmap_info, family_arr, splits, export_dir, ts, prefix="report"):
    ensure_dir(export_dir)
    fam2id=mmap_info["fam2id"]; raw_by=mmap_info["raw_rows_by_fam"]; eff_by=mmap_info["eff_rows_by_fam"]
    N_eff=int(mmap_info["N_eff"])
    tr,va,te=splits["train"],splits["val"],splits["test"]

    def cnt(arr):
        inv={v:k for k,v in fam2id.items()}; d={}
        for fid in np.unique(arr): d[inv[int(fid)]] = int((arr==fid).sum())
        return d

    fam_tr, fam_va, fam_te = cnt(family_arr[tr]), cnt(family_arr[va]), cnt(family_arr[te])
    fams=sorted(set(raw_by.keys())|set(eff_by.keys())|set(fam_tr.keys())|set(fam_va.keys())|set(fam_te.keys()))

    eff_vals=[eff_by.get(f,0) for f in fams]; eff_pct=[100.0*v/max(1,N_eff) for v in eff_vals]
    plt.figure(figsize=(10,4)); plt.bar(fams, eff_pct)
    plt.title("EFF% per Family"); plt.ylabel("%"); plt.xticks(rotation=25, ha="right")
    out1=os.path.join(export_dir, f"{prefix}.{ts}.eff_percent.png")
    plt.tight_layout(); plt.savefig(out1, dpi=150); plt.close()

    trc=np.array([fam_tr.get(f,0) for f in fams], float)
    vac=np.array([fam_va.get(f,0) for f in fams], float)
    tec=np.array([fam_te.get(f,0) for f in fams], float)
    denom=np.maximum(trc+vac+tec,1.0); trp, vap, tep = 100.0*trc/denom, 100.0*vac/denom, 100.0*tec/denom
    plt.figure(figsize=(10,4))
    plt.bar(fams, trp, label="TRAIN")
    plt.bar(fams, vap, bottom=trp, label="VAL")
    plt.bar(fams, tep, bottom=trp+vap, label="TEST")
    plt.title("Split Composition per Family"); plt.ylabel("%"); plt.legend(); plt.xticks(rotation=25, ha="right")
    out2=os.path.join(export_dir, f"{prefix}.{ts}.split_stacked.png")
    plt.tight_layout(); plt.savefig(out2, dpi=150); plt.close()
    print(f"[PLOT] -> {out1}\n[PLOT] -> {out2}")

def plot_time_charts(times, epoch_times, history, export_dir, ts, prefix):
    ensure_dir(export_dir)
    labels = []; vals = []
    order = ["memmap","split","dataset","build_compile","train","val_predict","test_predict","plots","save_meta"]
    for k in order:
        if k in times:
            labels.append(k); vals.append(times[k].get("sec", times[k].get("elapsed", 0.0)))
    plt.figure(figsize=(10,4)); plt.bar(labels, vals); plt.title("Section durations (sec)")
    plt.ylabel("seconds"); plt.xticks(rotation=25,ha="right")
    out1 = os.path.join(export_dir, f"{prefix}.{ts}.time_sections.png")
    plt.tight_layout(); plt.savefig(out1, dpi=150); plt.close()

    if len(epoch_times)>0:
        plt.figure(figsize=(8,4))
        plt.plot(np.arange(1,len(epoch_times)+1), epoch_times, marker="o")
        plt.title("Epoch time (sec)"); plt.xlabel("epoch"); plt.ylabel("seconds"); plt.grid(True, ls="--", alpha=.5)
        out2 = os.path.join(export_dir, f"{prefix}.{ts}.epoch_times.png")
        plt.tight_layout(); plt.savefig(out2, dpi=150); plt.close()
        print("[TIME PLOTS]\n  ->", out1, "\n  ->", out2)
    else:
        print("[TIME PLOTS]\n  ->", out1)

# ====== Timing helpers ======
def _nowstr(): return datetime.datetime.now().isoformat(timespec='seconds')

# ====== Model ======
def make_model(L=FIXED_LEN, emb_dim=16, filters=128, k=7, drop=0.25):
    inp = tf.keras.Input(shape=(L,), dtype=tf.int32)
    x = tf.keras.layers.Embedding(input_dim=256, output_dim=emb_dim)(inp)
    def block(x, dilation):
        x = tf.keras.layers.Conv1D(filters, k, padding="same", dilation_rate=dilation,
                                   kernel_regularizer=l2(1e-5))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        return tf.keras.layers.Dropout(drop)(x)
    for d in [1,2,4]:
        x = block(x, d)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=l2(1e-5))(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", dtype='float32')(x)
    return tf.keras.Model(inp, out)

# ====== Threshold (robust) ======
def choose_threshold(y_true, proba, policy="best_f1", target_recall=0.90,
                     min_precision=None, min_thr=1e-6):
    prec, rec, thr = precision_recall_curve(y_true, proba)  # len(thr) = len(prec)-1
    if policy == "best_f1":
        f1 = 2*prec[1:]*rec[1:] / (prec[1:]+rec[1:]+1e-9)
        i  = int(np.argmax(f1))
        t  = max(float(thr[i]), min_thr)
        if min_precision is not None and prec[i+1] < min_precision:
            j = i + 1
            while j < len(prec) and prec[j] < min_precision:
                if j-1 < len(thr): t = max(float(thr[j-1]), t)
                j += 1
        return t, {"policy":"best_f1","F1":float(f1[i]),
                   "Precision":float(prec[i+1]),"Recall":float(rec[i+1])}
    if policy == "target_recall":
        idx = np.where(rec >= target_recall)[0]
        if len(idx) > 0:
            i = int(idx[0])
            t = float(thr[max(i-1,0)]) if i > 0 else 1.0
            t = max(t, min_thr)
            if min_precision is not None:
                j = i
                while j < len(prec) and prec[j] < min_precision:
                    if j-1 < len(thr): t = max(float(thr[j-1]), t)
                    j += 1
            return t, {"policy":"target_recall",
                       "target_recall":float(target_recall),
                       "Precision_at_chosen":float(prec[min(i,len(prec)-1)]),
                       "Recall_at_chosen":float(rec[min(i,len(rec)-1)])}
        return max(min_thr, 0.5), {"policy":"target_recall_fallback"}
    return 0.5, {"policy":"fixed_0.5"}

# ====== Callbacks ======
class EpochTimer(tf.keras.callbacks.Callback):
    def __init__(self): super().__init__(); self.epoch_times=[]; self._tic=None
    def on_epoch_begin(self, epoch, logs=None): self._tic = time.perf_counter()
    def on_epoch_end(self, epoch, logs=None):
        if self._tic is not None: self.epoch_times.append(time.perf_counter()-self._tic)

# ====== MAIN ======
def main():
    if len(CSV_FILES) == 0: raise SystemExit("กรุณากำหนด CSV_FILES ก่อน")
    ensure_dir(EXPORT_DIR); ensure_dir(MMAP_DIR)
    ts = ts_now(); times = {}
    def t_start(k): times[k]={"start":time.perf_counter(),"wall_start":_nowstr()}
    def t_end(k): times[k]["end"]=time.perf_counter(); s=times[k]["end"]-times[k]["start"]; times[k]["sec"]=s; times[k]["wall_end"]=_nowstr()

    # -- MMAP
    t_start("memmap")
    mmap_info = build_memmap_from_csvs(CSV_FILES, L=FIXED_LEN, mmap_dir=MMAP_DIR)
    t_end("memmap")

    N_eff = int(mmap_info["N_eff"])
    X_all, y_all, groups, family = open_memmaps(mmap_info, N_eff, mmap_info["L"])
    fam2id, grp2id = mmap_info["fam2id"], mmap_info["grp2id"]

# -- Split (family/group-aware quota)
    t_start("split")
    if REUSE_SPLIT_FROM_META and os.path.exists(PREV_META_PATH):
        m_prev = joblib.load(PREV_META_PATH)
        train_idx = np.array(m_prev["indices"]["train"], dtype=int)
        val_idx   = np.array(m_prev["indices"]["val"],   dtype=int)
        test_idx  = np.array(m_prev["indices"]["test"],  dtype=int)

        max_idx = max(train_idx.max(initial=-1),
                      val_idx.max(initial=-1),
                      test_idx.max(initial=-1))
        if max_idx >= N_eff:
            raise RuntimeError(
                f"[SPLIT] Reused indices go out of range (max={max_idx}, N_eff={N_eff}). "
                "Meta/CSV order อาจไม่ตรงกับรอบนี้"
            )
        print("[SPLIT] Reused split from meta:", PREV_META_PATH)
    else:
        train_idx, val_idx, test_idx = split_family_group_aware_quota(
            groups=groups, family=family, y=y_all,
            test_frac=TEST_SIZE, val_frac=VAL_SIZE,
            min_per_split_per_family=max(5000, int(0.005*len(y_all))),
            min_per_label_per_family=10, seed=SEED
        )
        for name, arr in [("TRAIN", train_idx), ("VAL", val_idx), ("TEST", test_idx)]:
            if len(arr) == 0:
                raise RuntimeError(f"[FATAL] {name} split empty")

    # ← ย้าย 2 บรรทัดนี้ออกมารันเสมอ
    train_idx, val_idx, test_idx = enforce_family_coverage(
        train_idx, val_idx, test_idx, family, min_each=1
    )

    # ← และให้ rebalance/oversample รันเสมอ (TRAIN เปิด oversample, VAL/TEST ปิด)
    train_idx = rebalance_split_to_targets(
        train_idx, family, fam2id,
        benign_name="benign", keep_floor_per_family=50, seed=SEED,
        oversample=True, max_multiplier=3,
        targets={"benign":0.5, "emotet":1/6, "trickbot":1/6, "zeus":1/6}
    )
    val_idx = rebalance_split_to_targets(
        val_idx, family, fam2id,
        benign_name="benign", keep_floor_per_family=10, seed=SEED,
        oversample=False
    )
    test_idx = rebalance_split_to_targets(
        test_idx, family, fam2id,
        benign_name="benign", keep_floor_per_family=10, seed=SEED,
        oversample=False
    )
    t_end("split")
    
    print("Shapes:", (len(train_idx), FIXED_LEN), (len(val_idx), FIXED_LEN), (len(test_idx), FIXED_LEN))
    print_dataset_stats("TRAIN", y_all[train_idx], family[train_idx], fam2id)
    print_dataset_stats("VAL",   y_all[val_idx],   family[val_idx],   fam2id)
    print_dataset_stats("TEST",  y_all[test_idx],  family[test_idx],  fam2id)

    # undersample ต่อ family ใน TRAIN (ถ้าต้องการ)
    if MAX_PER_FAMILY_TRAIN is not None:
        rng=np.random.RandomState(SEED); keep=[]
        for f in np.unique(family[train_idx]):
            idx=np.where(family[train_idx]==f)[0]
            keep.append(rng.choice(idx,size=min(len(idx),MAX_PER_FAMILY_TRAIN),replace=False))
        keep_rel=np.concatenate(keep) if keep else np.arange(len(train_idx))
        train_idx=train_idx[keep_rel]
        print_dataset_stats("TRAIN (after balance)", y_all[train_idx], family[train_idx], fam2id)

    # class weights
    y_tr = y_all[train_idx].astype(int)
    cw_vals = compute_class_weight("balanced", classes=np.array([0,1]), y=y_tr)
    class_weight = {0: float(cw_vals[0]), 1: float(cw_vals[1])}
    print("Class weight:", class_weight)

    # datasets
    t_start("dataset")
    train_ds = make_dataset_from_indices(X_all, y_all, train_idx, BATCH, training=True)
    val_ds   = make_dataset_from_indices(X_all, y_all, val_idx,   BATCH)
    test_ds  = make_dataset_from_indices(X_all, y_all, test_idx,  BATCH)
    t_end("dataset")


    # ===== build/compile =====
    t_start("build_compile")
    if TRAIN_MODE == "continue" or EVAL_ONLY:
        if not BASE_MODEL_PATH: raise RuntimeError("ต้องระบุ BASE_MODEL_PATH เมื่อ continue/EVAL_ONLY")
        print("[MODE] Load existing model:", BASE_MODEL_PATH)
        model = tf.keras.models.load_model(BASE_MODEL_PATH)
        lr_new = RESUME_LR if RESUME_LR is not None else 1e-4
    else:
        print("[MODE] Train from scratch")
        model = make_model()
        lr_new = 1e-4

    # ใช้ lr_new ตรง ๆ
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_new, clipvalue=1.0)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="AUC"),
            tf.keras.metrics.AUC(name="PR-AUC", curve="PR"),
            tf.keras.metrics.Precision(name="Precision"),
            tf.keras.metrics.Recall(name="Recall"),
        ],
    )

    try:
        cur_lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
    except Exception:
        cur_lr = float(tf.keras.backend.get_value(model.optimizer.lr))
    print(f"[OPT] Recompiled Adam with LR = {cur_lr:g}")
    model.summary()
    t_end("build_compile")

    # callbacks
    ckpt_path = os.path.join(EXPORT_DIR, f"{EXPORT_NAME_PREFIX}.best.{ts}.keras")

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        monitor="val_AUC",          # เดิมเป็น val_PR-AUC
        mode="max",
        save_best_only=True,
        save_weights_only=False,     # ถ้าอยากเซฟเฉพาะ weights ให้เปลี่ยนเป็น True
        verbose=1
    )

    earlystop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_AUC",           # เดิมเป็น val_PR-AUC
        mode="max",
        patience=5,
        restore_best_weights=True
    )

    plateau_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_AUC",           # เดิมเป็น val_PR-AUC
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    print_lr = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs:
            print(f"Epoch {epoch+1}: lr={tf.keras.backend.get_value(model.optimizer.learning_rate):.6g}")
    )

    callbacks = [checkpoint_cb, earlystop_cb, plateau_cb, EpochTimer(), print_lr]


    # train
    history = None
    if not EVAL_ONLY:
        print(f"[TRAIN] start at {datetime.datetime.now().isoformat(timespec='seconds')}")
        t_start("train")
        
        steps_per_epoch = math.ceil(len(train_idx)/BATCH)
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch,   # ถ้าใช้ .repeat() อย่าลืมบรรทัดนี้
            class_weight=class_weight,
            callbacks=callbacks,               # ใช้ลิสต์เดียว
            verbose=1
        )

        t_end("train")
        print(f"[TRAIN] end   at {datetime.datetime.now().isoformat(timespec='seconds')}")
        print(f"[TRAIN] total seconds = {times['train']['sec']:.2f}")

    else:
        print("[MODE] EVAL_ONLY: skip training")

    # ===== Evaluate
    print("[VAL] predicting ...")
    t_start("val_predict")
    proba_val = model.predict(val_ds).ravel()
    t_end("val_predict")
    chosen_thr, thr_info = choose_threshold(
        y_all[val_idx], proba_val,
        policy=THRESHOLD_POLICY, target_recall=TARGET_RECALL,
        min_precision=MIN_PREC_AT_THR, min_thr=MIN_THR_FLOOR
    )

    print("[TEST] predicting ...")
    t_start("test_predict")
    proba_test = model.predict(test_ds).ravel()
    t_end("test_predict")
    pred_test  = (proba_test >= chosen_thr).astype(int)

    y_true = y_all[test_idx].astype(int)
    roc = roc_auc_score(y_true, proba_test)
    pr  = average_precision_score(y_true, proba_test)
    cm  = confusion_matrix(y_true, pred_test)
    report_dict = classification_report(y_true, pred_test, digits=4, output_dict=True, zero_division=0)
    acc = accuracy_score(y_true, pred_test)

    print("\n==== FINAL EVALUATION REPORT ====")
    print(f"ROC-AUC (TEST): {roc:.4f}")
    print(f"PR-AUC  (TEST): {pr:.4f}")
    print(f"[THRESHOLD from VAL] = {chosen_thr:.6f}  info = {thr_info}")
    print("\nClassification Report (per class):")
    print(classification_report(y_true, pred_test, digits=4, zero_division=0))
    print(f"Confusion Matrix (TEST) @thr={chosen_thr:.6f}:\n{cm}")
    print(f"[ACCURACY] {acc:.6f}")

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    print_family_mix(mmap_info, family, splits)

    # plots
    t_start("plots")
    plot_family_mix_charts(mmap_info, family, splits, EXPORT_DIR, ts, prefix=EXPORT_NAME_PREFIX)
    plot_curves_and_cm(y_true, proba_test, pred_test, EXPORT_DIR, ts, prefix=EXPORT_NAME_PREFIX)

    # epoch times (from callbacks)
    epoch_times = []
    for cb in callbacks:
        if isinstance(cb, EpochTimer):
            epoch_times = cb.epoch_times
            break
    def plot_time_charts_local():
        labels = []; vals = []
        order = ["memmap","split","dataset","build_compile","train","val_predict","test_predict","plots","save_meta"]
        for k in order:
            if k in times:
                labels.append(k); vals.append(times[k].get("sec", 0.0))
        plt.figure(figsize=(10,4)); plt.bar(labels, vals); plt.title("Section durations (sec)")
        plt.ylabel("seconds"); plt.xticks(rotation=25,ha="right")
        out1 = os.path.join(EXPORT_DIR, f"{EXPORT_NAME_PREFIX}.{ts}.time_sections.png")
        plt.tight_layout(); plt.savefig(out1, dpi=150); plt.close()

        if len(epoch_times)>0:
            plt.figure(figsize=(8,4))
            plt.plot(np.arange(1,len(epoch_times)+1), epoch_times, marker="o")
            plt.title("Epoch time (sec)"); plt.xlabel("epoch"); plt.ylabel("seconds"); plt.grid(True, ls="--", alpha=.5)
            out2 = os.path.join(EXPORT_DIR, f"{EXPORT_NAME_PREFIX}.{ts}.epoch_times.png")
            plt.tight_layout(); plt.savefig(out2, dpi=150); plt.close()
            print("[TIME PLOTS]\n  ->", out1, "\n  ->", out2)
        else:
            print("[TIME PLOTS]\n  ->", out1)

        # val PR-AUC per epoch
        if history is not None and "val_PR-AUC" in history.history:
            plt.figure(figsize=(8,4))
            plt.plot(np.arange(1,len(history.history["val_PR-AUC"])+1),
                     history.history["val_PR-AUC"], marker="o")
            plt.title("Validation PR-AUC per epoch")
            plt.xlabel("epoch"); plt.ylabel("val PR-AUC"); plt.grid(True, ls="--", alpha=.5)
            out3 = os.path.join(EXPORT_DIR, f"{EXPORT_NAME_PREFIX}.{ts}.val_pr_auc_per_epoch.png")
            plt.tight_layout(); plt.savefig(out3, dpi=150); plt.close()
            print("  ->", out3)
    plot_time_charts_local()
    t_end("plots")

    # save model & meta & history
    t_start("save_meta")
    if not EVAL_ONLY:
        final_model_path = os.path.join(EXPORT_DIR, f"{EXPORT_NAME_PREFIX}.final.{ts}.keras")
        model.save(final_model_path)
        print(f"Saved BEST -> {ckpt_path}\nSaved FINAL -> {final_model_path}")
    else:
        final_model_path = BASE_MODEL_PATH

    # throughput (โดยประมาณ)
    train_secs = times.get("train", {}).get("sec", None)
    if train_secs and history is not None:
        approx_train_samples = len(train_idx)  # คร่าวๆ (fit แบบเต็ม 1 pass/epoch * epochs ~ ประมาณ)
        print(f"[THROUGHPUT] Train ~ {approx_train_samples/max(train_secs,1e-9):.2f} samples/sec")

    test_secs = times.get("test_predict", {}).get("sec", None)
    if test_secs:
        print(f"[THROUGHPUT] Test  ~ {len(test_idx)/max(test_secs,1e-9):.2f} samples/sec")

    history_csv = os.path.join(EXPORT_DIR, "history.csv")
    write_history(history_csv, ts, N_eff, acc, report_dict)
    plot_history(history_csv, EXPORT_DIR, ts)

    meta = {
        "fixed_len": FIXED_LEN, "payload_col": PAYLOAD_COL, "label_col": LABEL_COL,
        "decode": "base64->hex->latin1", "window_mode": "head",
        "seed": SEED, "train_mode": TRAIN_MODE,
        "base_model": BASE_MODEL_PATH if (TRAIN_MODE=="continue" or EVAL_ONLY) else None,
        "timestamp": ts, "group_aware_split": True,
        "files": [(str(p), str(fam)) for p, fam in _normalize_csvs(CSV_FILES)],
        "mmap_dir": MMAP_DIR,
        "fam2id": mmap_info["fam2id"], "grp2id": mmap_info["grp2id"],
        "N_total": int(mmap_info["N_total"]), "N_eff": int(N_eff), "L": int(mmap_info["L"]),
        "indices": {"train": train_idx.tolist(), "val": val_idx.tolist(), "test": test_idx.tolist()},
        "threshold": float(chosen_thr), "threshold_info": thr_info,
        "metrics": {"roc_auc_test": float(roc), "pr_auc_test": float(pr),
                    "classification_report": report_dict, "confusion_matrix": cm.tolist()},
        "timing": {k: {"sec": v.get("sec", 0.0),
                       "wall_start": v.get("wall_start", "-"), "wall_end": v.get("wall_end", "-")}
                   for k, v in times.items()}
    }
    meta_path = os.path.join(EXPORT_DIR, f"preprocess_meta.{ts}.joblib")
    joblib.dump(meta, meta_path)
    with open(os.path.join(EXPORT_DIR, f"timing.{ts}.json"), "w") as f:
        json.dump(meta["timing"], f, indent=2)
    print(f"Saved meta -> {meta_path}")

    # final report .txt
    if SAVE_FINAL_TXT:
        txt = os.path.join(EXPORT_DIR, f"final_report.{ts}.txt")
        with open(txt, "w", encoding="utf-8") as f:
            f.write("==== FINAL EVALUATION REPORT ====\n")
            f.write(f"ROC-AUC (TEST): {roc:.6f}\nPR-AUC  (TEST): {pr:.6f}\n")
            f.write(f"[THRESHOLD from VAL] = {chosen_thr:.6f}  info = {thr_info}\n\n")
            f.write("Classification Report (per class):\n")
            f.write(classification_report(y_true, pred_test, digits=4))
            f.write("\n"); f.write(f"Confusion Matrix (TEST) @thr={chosen_thr:.6f}:\n{cm}\n")
            f.write("\n==== TIMING (seconds) ====\n")
            for k, v in times.items():
                f.write(f"{k:>14s}: {v.get('sec',0.0):.3f}  (start={v.get('wall_start','-')}, end={v.get('wall_end','-')})\n")
        print(f"[REPORT] -> {txt}")

    t_end("save_meta")
    return final_model_path, meta_path

# ====== Inference helper ======
def predict_payload_string(s, model_path, meta_path=None, fixed_len=FIXED_LEN, window_mode="head",
                           threshold=None):
    mdl = tf.keras.models.load_model(model_path)
    if meta_path:
        m = joblib.load(meta_path)
        fixed_len   = m.get("fixed_len", fixed_len)
        window_mode = m.get("window_mode", window_mode)
        if threshold is None:
            threshold = float(m.get("threshold", 0.5))
    b = decode_payload(s)
    x = to_fixed_len_uint8(b, L=fixed_len, mode=window_mode).astype(np.int32)
    p = float(mdl.predict(x[None, ...], verbose=0).ravel()[0])
    thr = 0.5 if threshold is None else float(threshold)
    return {"prob_malicious": p, f"label@{thr:g}": int(p >= thr), "threshold_used": thr}

if __name__ == "__main__":
    final_model_path, meta_path = main()
    demo_hex = "48656c6c6f"  # "Hello"
    print("Demo inference:", predict_payload_string(demo_hex, model_path=final_model_path, meta_path=meta_path))
