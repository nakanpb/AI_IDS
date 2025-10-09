#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-IDS Realtime (single CSV) + Popup ที่ทำงานได้แม้รันด้วย sudo
- Terminal: แสดงเฉพาะ alert (pred=1) โดยเอา traffic ขึ้นก่อน แล้วตามด้วย p/thr/label/latency
- CSV: เก็บทุกแพ็กเก็ต + event (info/email/check)
- Popup: ถ้าโปรเซสเป็น root จะสลับไปเรียก notify-send ในนาม user desktop อัตโนมัติ
หมายเหตุ: โค้ดนี้สมมติว่ามีโมเดล/เมตา+ฟังก์ชัน preprocess พร้อมใช้งานแล้ว (ตามที่คุณมีอยู่)
"""

import os, sys, time, csv, datetime, threading, queue, subprocess
from typing import List, Tuple, Optional, Dict, Set
import numpy as np

from scapy.all import AsyncSniffer, IP, TCP, UDP, Raw
import tensorflow as tf
import joblib
from email.message import EmailMessage
import smtplib

# ===================== CONFIG =====================
ENABLE_LIVE_SNIFF = True
IFACE     = "eno2"
BPF_FILTER = "(tcp or udp) and not (dst net 224.0.0.0/4 or dst 255.255.255.255 or broadcast or port 5353 or port 67 or port 68)"
SRC_IP    = None
DST_IP    = None

MODEL_PATH = "/home/intgan/Desktop/jek/train/model7.best.20251005-230508.keras"
META_PATH  = "/home/intgan/Desktop/jek/train/preprocess_meta.20251005-230508.joblib"
TERM_LOG_CSV_PATH = "/home/intgan/Desktop/jek/log/model_outputlog(mix)2.csv"

BATCH_SIZE  = 32
THRESHOLD   = "auto"
PRINT_ONLY_ALERTS = True  # terminal แสดงเฉพาะ pred=1

EMAIL_NOTIFY = True
os.environ.setdefault("AIIDS_SMTP_USER", "intgan.sender@gmail.com")
os.environ.setdefault("AIIDS_SMTP_PASS", "urgtstcqjqdmgisi")
os.environ.setdefault("AIIDS_EMAIL_FROM", "intgan.sender@gmail.com")
os.environ.setdefault("AIIDS_EMAIL_TO",   "intgan.receive@gmail.com")
SMTP_HOST="smtp.gmail.com"; SMTP_PORT=587; SMTP_USE_TLS=True
SMTP_USER=os.getenv("AIIDS_SMTP_USER",""); SMTP_PASS=os.getenv("AIIDS_SMTP_PASS","")
EMAIL_FROM=os.getenv("AIIDS_EMAIL_FROM", SMTP_USER or "aiids@localhost")
EMAIL_TO=os.getenv("AIIDS_EMAIL_TO","")
EMAIL_MIN_COOLDOWN_SEC = 30.0

USE_TRUTH   = True
TRUTH_CSV   = "/home/intgan/Desktop/jek/sampletest/truth_mix1.csv"
MALICIOUS_SRCS = ""   # path รายชื่อ IP (optional)
BENIGN_SRCS    = ""
DEFAULT_TRUE_LABEL = None

ENABLE_AUTO_CHECK = False
CHECK_EVERY_SEC   = 30
# ===================== CONFIG popup (สำหรับรันด้วย sudo) =====================
DESKTOP_USER = os.environ.get("SUDO_USER") or "intgan"
DESKTOP_UID  = int(os.environ.get("SUDO_UID") or "1000")
DESKTOP_DISPLAY = os.environ.get("DISPLAY") or ":1"

# ===================== GPU setup =====================
_gpus = tf.config.list_physical_devices('GPU')
if _gpus:
    try:
        for g in _gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print("[GPU] Using:", _gpus)
    except Exception as e:
        print("[GPU] setup error:", e)
else:
    print("[GPU] Not found, using CPU.")

# ===================== Import preprocess =====================
try:
    from train import to_fixed_len_uint8 as ext_to_fixed_len_uint8, FIXED_LEN as EXT_FIXED_LEN
except Exception as e:
    raise ImportError("ต้องมีไฟล์ train.py ที่มี to_fixed_len_uint8 และ FIXED_LEN") from e

# ===================== Popup helper =====================
def send_popup(title: str, body: str, urgency: str = "normal", timeout_ms: int = 8000) -> bool:
    """
    ส่ง popup ให้ user desktop:
      - ถ้าไม่ได้รันด้วย sudo → เรียก notify-send ตรง ๆ
      - ถ้ารันด้วย sudo/root → สลับไปเรียกในนาม DESKTOP_USER ด้วย env DISPLAY/DBUS ของ user นั้น
    """
    bus = f"unix:path=/run/user/{DESKTOP_UID}/bus"
    if os.geteuid() != 0:
        env = dict(os.environ)
        env.setdefault("DBUS_SESSION_BUS_ADDRESS", bus)
        env.setdefault("DISPLAY", DESKTOP_DISPLAY)
        try:
            subprocess.run(
                ["notify-send","-u",urgency,"-t",str(int(timeout_ms)), title, body],
                check=False, env=env
            )
            return True
        except Exception:
            return False
    else:
        cmd = [
            "sudo","-u",DESKTOP_USER,"-H","env",
            f"DISPLAY={DESKTOP_DISPLAY}",
            f"DBUS_SESSION_BUS_ADDRESS={bus}",
            "notify-send","-u",urgency,"-t",str(int(timeout_ms)), title, body
        ]
        try:
            subprocess.run(cmd, check=False)
            return True
        except Exception:
            return False

# ===================== Email / Logging =====================
START_TS = time.time()
_last_email_ts = 0.0

class TerminalCsvLogger:
    HEADER = ["type","timestamp","src","sport","dst","dport","proto",
              "payload_len","prob","pred","true","latency","detail"]
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fh = open(self.path, "a", newline="", encoding="utf-8")
        self._w  = csv.writer(self._fh)
        if os.stat(self.path).st_size == 0:
            self._w.writerow(self.HEADER); self._fh.flush()
        print(f"[TERM] logging to {self.path}")
    def write_packet(self, meta: dict, prob: float, pred: int, true_val: Optional[str], latency_s: float):
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        self._w.writerow(["packet", ts,
                          meta.get("src",""), meta.get("sport",""),
                          meta.get("dst",""), meta.get("dport",""),
                          meta.get("proto",""), meta.get("plen",""),
                          f"{prob:.9g}", pred, (true_val or ""), f"{latency_s:.3f}", ""])
        self._fh.flush()
    def write_event(self, etype: str, detail: str):
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        self._w.writerow([etype, ts, "", "", "", "", "", "", "", "", "", "", detail])
        self._fh.flush()
    def close(self):
        try: self._fh.flush(); self._fh.close()
        except Exception: pass

def tee(msg: str, etype: Optional[str]=None, detail: Optional[str]=None):
    print(msg)
    if _termlog and etype:
        _termlog.write_event(etype, detail if detail is not None else msg)

def send_email(subject: str, body: str):
    global _last_email_ts
    if not EMAIL_NOTIFY: return
    now = time.time()
    if (now - _last_email_ts) < EMAIL_MIN_COOLDOWN_SEC:
        tee(f"[EMAIL] throttled ({int(EMAIL_MIN_COOLDOWN_SEC - (now - _last_email_ts))}s left)", "email", "throttled")
        return
    if not (SMTP_HOST and SMTP_PORT and SMTP_USER and SMTP_PASS and EMAIL_TO):
        tee("[EMAIL] skipped: SMTP/recipient not configured", "email", "skipped")
        return
    try:
        msg = EmailMessage()
        msg["From"] = EMAIL_FROM; msg["To"] = EMAIL_TO
        msg["Subject"] = f"[AIIDS] {subject}"
        msg.set_content(body)
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as s:
            s.ehlo(); 
            if SMTP_USE_TLS: s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        _last_email_ts = now
        tee("[EMAIL] sent", "email", "sent")
    except Exception as e:
        tee(f"[EMAIL] failed: {e}", "email", f"failed: {e}")

# ===================== Truth helpers =====================
def _norm_proto(x) -> str:
    if x is None: return "IP"
    if isinstance(x, str):
        u = x.strip().upper()
        if u in ("TCP","UDP","IP"): return u
        try:
            v = int(u); return "TCP" if v==6 else ("UDP" if v==17 else "IP")
        except Exception: return "IP"
    try:
        v = int(x); return "TCP" if v==6 else ("UDP" if v==17 else "IP")
    except Exception:
        return "IP"

def _load_truth_csv(path: str) -> Dict[Tuple[str,int,str,int,str], int]:
    mp: Dict[Tuple[str,int,str,int,str], int] = {}
    if not (path and os.path.exists(path)): return mp
    import csv as _csv
    def _get(row, keys, default=""):
        for k in keys:
            if k in row and row[k] not in (None, ""): return row[k]
        return default
    def _to_int(x, default=0):
        try: return int(x)
        except Exception: return default
    def _to_label(v) -> Optional[int]:
        if v is None: return None
        s = str(v).strip().lower()
        if s in ("1","malicious","mal","bad"): return 1
        if s in ("0","benign","good"): return 0
        return None
    with open(path, newline="", encoding="utf-8") as fp:
        r = _csv.DictReader(fp)
        for row in r:
            src   = _get(row, ["src","src_ip"]).strip()
            dst   = _get(row, ["dst","dst_ip"]).strip()
            sport = _to_int(_get(row, ["sport","src_port"]))
            dport = _to_int(_get(row, ["dport","dst_port"]))
            proto = _norm_proto(_get(row, ["proto","protocol","l4"]))
            lab   = _to_label(_get(row, ["label","true","ground_truth"]))
            if src and dst and lab is not None:
                mp[(src, sport, dst, dport, proto)] = lab
    print(f"[TRUTH] loaded exact entries: {len(mp)}")
    return mp

def _load_ip_set(path: str) -> Set[str]:
    s: Set[str] = set()
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fp:
            for line in fp:
                ip=line.strip()
                if ip: s.add(ip)
    return s

def _lookup_true(meta: dict,
                 mp_exact: Dict[Tuple[str, int, str, int, str], int],
                 mal_srcs: Set[str],
                 ben_srcs: Set[str],
                 default_true: Optional[int]) -> Optional[int]:
    k1=(meta["src"], int(meta["sport"]), meta["dst"], int(meta["dport"]), _norm_proto(meta["proto"]))
    k2=(meta["dst"], int(meta["dport"]), meta["src"], int(meta["sport"]), _norm_proto(meta["proto"]))
    if k1 in mp_exact: return mp_exact[k1]
    if k2 in mp_exact: return mp_exact[k2]
    if meta["src"] in mal_srcs: return 1
    if meta["src"] in ben_srcs: return 0
    if default_true is not None: return int(default_true)
    return None

# ===================== Main =====================
_termlog: Optional[TerminalCsvLogger] = None

def run_main():
    global _termlog
    _termlog = TerminalCsvLogger(TERM_LOG_CSV_PATH)

    print(f"[INFO] Loading model: {MODEL_PATH}")
    mdl = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"[INFO] Loading meta : {META_PATH}")
    meta = joblib.load(META_PATH)
    fixed_len  = int(meta.get("fixed_len", EXT_FIXED_LEN))
    window_mode= meta.get("window_mode", "head")
    thr_auto   = float(meta.get("threshold", 0.5))
    threshold = max(thr_auto, 1e-4) if THRESHOLD == "auto" else float(THRESHOLD)

    tp = tn = fp = fn = 0
    skipped_truth = 0

    if USE_TRUTH:
        mp_exact = _load_truth_csv(TRUTH_CSV)
        mal_srcs = _load_ip_set(MALICIOUS_SRCS)
        ben_srcs = _load_ip_set(BENIGN_SRCS)
        default_true = None if DEFAULT_TRUE_LABEL is None else (1 if str(DEFAULT_TRUE_LABEL).lower().startswith("mal") else 0)
    else:
        mp_exact = {}; mal_srcs=set(); ben_srcs=set(); default_true=None

    work_q: "queue.Queue[Tuple[dict, bytes, float]]" = queue.Queue()
    stop_flag = threading.Event()
    stat_total = 0
    stat_alert = 0

    tee(f"[INFO] Using threshold={threshold:.9g} (auto from meta={thr_auto:.9g})",
        "info", f"threshold={threshold:.9g}")

    def make_x(payload: bytes):
        return ext_to_fixed_len_uint8(payload, L=fixed_len, mode=window_mode).astype(np.int32, copy=False)

    def predict_batch(mdl, X: np.ndarray) -> np.ndarray:
        y = mdl.predict(X, verbose=0)  
        y = np.asarray(y).ravel()
        return y.astype(float)

    def infer_and_log(buf_meta: List[dict], buf_x: List[np.ndarray], buf_ts: List[float]):
        nonlocal stat_total, stat_alert, tp, tn, fp, fn, skipped_truth
        if not buf_x: return
        X = np.stack(buf_x, axis=0).astype(np.int32)
        probs = predict_batch(mdl, X)
        for meta_i, p, pkt_ts in zip(buf_meta, probs, buf_ts):
            lbl = int(p >= threshold)
            stat_total += 1
            if lbl == 1: stat_alert += 1
            latency = time.time() - float(pkt_ts)

            # truth
            t_num = None; t_str = ""
            if USE_TRUTH:
                t_num = _lookup_true(meta_i, mp_exact, mal_srcs, ben_srcs, default_true)
                if t_num is not None:
                    t_num = int(t_num)
                    t_str = "malicious" if t_num == 1 else "benign"

            if t_num is not None:
                if   lbl == 1 and t_num == 1: tp += 1
                elif lbl == 0 and t_num == 0: tn += 1
                elif lbl == 1 and t_num == 0: fp += 1
                elif lbl == 0 and t_num == 1: fn += 1
            else:
                skipped_truth += 1

            # log ทุกแพ็กเก็ต
            _termlog.write_packet(meta_i, p, lbl, t_str, latency)

            # terminal (เฉพาะ alert)
            if (not PRINT_ONLY_ALERTS) or lbl == 1:
                print(
                    f"[DETECT] {meta_i['src']}:{meta_i['sport']} -> {meta_i['dst']}:{meta_i['dport']} "
                    f"proto={meta_i['proto']} len={meta_i['plen']} | "
                    f"p={p:.9g} thr={threshold:.9g} label={lbl} | latency={latency:.3f}s"
                )

            # popup + email เมื่อ alert
            if lbl == 1:
                send_popup("🚨 AI-IDS Alert",
                           f"{meta_i['src']}:{meta_i['sport']} -> {meta_i['dst']}:{meta_i['dport']}\nprob={p:.4g} lat={latency:.3f}s",
                           urgency="critical", timeout_ms=8000)
                body=(f"Time: {datetime.datetime.now().isoformat(timespec='seconds')}\n"
                      f"src: {meta_i['src']}:{meta_i['sport']}\n"
                      f"dst: {meta_i['dst']}:{meta_i['dport']}\n"
                      f"proto: {meta_i['proto']}\n"
                      f"payload_len: {meta_i['plen']}\n"
                      f"prob: {p:.6f}\n"
                      f"pred: {lbl}\n"
                      f"latency: {latency:.3f}s\n")
                send_email("🚨 Malware suspected", body)

    def infer_worker():
        buf_meta, buf_x, buf_ts = [], [], []
        last = time.time()
        while not stop_flag.is_set():
            try:
                m, payload_bytes, pkt_ts = work_q.get(timeout=0.1)
                x = make_x(payload_bytes)
                buf_meta.append(m); buf_x.append(x); buf_ts.append(pkt_ts)
            except queue.Empty:
                pass
            if buf_x and (len(buf_x) >= BATCH_SIZE or time.time()-last > 0.25):
                infer_and_log(buf_meta, buf_x, buf_ts)
                buf_meta.clear(); buf_x.clear(); buf_ts.clear(); last = time.time()
        if buf_x:
            infer_and_log(buf_meta, buf_x, buf_ts)

    threading.Thread(target=infer_worker, daemon=True).start()

    def summary_worker():
        nonlocal stat_total, stat_alert, tp, tn, fp, fn, skipped_truth
        while not stop_flag.is_set():
            time.sleep(60)
            prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec  = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            acc  = ((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 0.0
            f1   = (2*prec*rec/(prec+rec)) if (prec+rec) > 0 else 0.0
            msg = (f"Uptime: {int((time.time()-START_TS)//60)} min | total={stat_total}, alerts={stat_alert} | "
                   f"TP={tp}, TN={tn}, FP={fp}, FN={fn}, skipped={skipped_truth} | "
                   f"Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
            tee(f"[INFO] {msg}", "info", msg)
            send_email("AI-IDS 1-min Summary",
                       f"Total: {stat_total}\nAlerts: {stat_alert}\nUptime(min): {int((time.time()-START_TS)//60)}")
    threading.Thread(target=summary_worker, daemon=True).start()

    def on_packet(pkt):
        try:
            if IP not in pkt: return
            if SRC_IP and pkt[IP].src != SRC_IP: return
            if DST_IP and pkt[IP].dst != DST_IP: return
            if TCP in pkt:
                proto="TCP"; payload=bytes(pkt[TCP].payload); sport=pkt[TCP].sport; dport=pkt[TCP].dport
            elif UDP in pkt:
                proto="UDP"; payload=bytes(pkt[UDP].payload); sport=pkt[UDP].sport; dport=pkt[UDP].dport
            else:
                proto="IP"; payload=bytes(pkt[Raw].load) if Raw in pkt else b""; sport=0; dport=0
            meta_i={"src":pkt[IP].src,"dst":pkt[IP].dst,"sport":sport,"dport":dport,"proto":proto,"plen":len(payload)}
            pkt_ts=float(getattr(pkt,"time",time.time()))
            work_q.put((meta_i,payload,pkt_ts))
        except Exception as e:
            tee(f"[WARN] on_packet error: {e}", "info", f"on_packet error: {e}")

    sniffer=None
    if ENABLE_LIVE_SNIFF:
        print(f"[INFO] Sniffing iface={IFACE} bpf='{BPF_FILTER}' ...")
        sniffer=AsyncSniffer(iface=IFACE, filter=BPF_FILTER, prn=on_packet, store=False, promisc=True)
        sniffer.start()
        print("[DBG] sniffer started")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        tee("[INFO] KeyboardInterrupt, stopping...", "info", "KeyboardInterrupt")
    finally:
        try:
            if sniffer: sniffer.stop()
        finally:
            _termlog.close()
            total = time.time()-START_TS
            tee(f"[INFO] Stopped. Logged to {TERM_LOG_CSV_PATH}", "info", f"stopped; csv={TERM_LOG_CSV_PATH}")
            tee(f"[INFO] Total runtime: {total/60:.2f} minutes ({total:.1f} sec)", "info", f"runtime={total:.1f}s")

if __name__ == "__main__":
    run_main()
