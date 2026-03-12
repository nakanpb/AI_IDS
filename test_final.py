#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-IDS Realtime: Selectable Threshold + Smart Alert Tag + Low Latency
"""

import os, sys, time, csv, datetime, threading, queue, subprocess
from typing import List, Tuple, Optional, Dict, Set
import numpy as np

from scapy.all import AsyncSniffer, IP, TCP, UDP, Raw
import tensorflow as tf
import joblib
from email.message import EmailMessage
import smtplib

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    print("[ERROR] Please install scikit-learn: pip install scikit-learn")
    sys.exit(1)

# ===================== CONFIG =====================
ENABLE_LIVE_SNIFF = True
IFACE     = "eno2"
BPF_FILTER = "(tcp or udp) and not (dst net 224.0.0.0/4 or dst 255.255.255.255 or broadcast or port 5353 or port 67 or port 68)"
SRC_IP    = None
DST_IP    = None

# ---------------------------------------------------------
# *** ตั้งค่า Threshold ตรงนี้ ***
# ---------------------------------------------------------
# True = ใช้ค่าที่เราพิมพ์เอง (Manual), False = ใช้ค่าจากไฟล์ Train (Auto)
USE_MANUAL_THRESHOLD = True 

# ค่าที่จะใช้เมื่อ USE_MANUAL_THRESHOLD = True
MANUAL_THRESHOLD_VAL = 0.15
# ---------------------------------------------------------

MODEL_PATH = "/home/intgan/Desktop/jek/train/final5_dataset.final.20260226-144417.keras" 
META_PATH  = "/home/intgan/Desktop/jek/train/preprocess_meta.20260226-144417.joblib"
TERM_LOG_CSV_PATH = "/home/intgan/Desktop/jek/log/final5_outputlog2.csv"

# Config Flow Timeout
FLOW_TIMEOUT_SEC = 120
FLOW_CLEANUP_INTERVAL = 5.0 # ให้เช็ค Cleanup ทุกๆ 5 วินาที
FLOW_TIMEOUT_LOG_PATH = "/home/intgan/Desktop/jek/log/final/final2_flow_timeout_log.csv"
MAX_FLOW_PAYLOAD = None

PRINT_ONLY_ALERTS = True
REPORT_INTERVAL_SEC = 60

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

USE_TRUTH    = True
TRUTH_CSV    = "/home/intgan/Desktop/jek/sampletest/truth_mix_final.csv"

# ===================== Preprocess Logic =====================
FIXED_LEN = 1460 
def to_fixed_len_uint8(b: bytes, L=FIXED_LEN, mode="head"):
    arr = np.frombuffer(b, dtype=np.uint8)
    n = len(arr)
    arr_shifted = arr.astype(np.int32) + 1
    if n >= L:
        if mode == "tail": return arr_shifted[-L:]
        return arr_shifted[:L]
    out = np.zeros(L, dtype=np.int32)
    out[:n] = arr_shifted
    return out

# ===================== Helpers =====================
DESKTOP_USER = os.environ.get("SUDO_USER") or "intgan"
DESKTOP_UID  = int(os.environ.get("SUDO_UID") or "1000")
DESKTOP_DISPLAY = os.environ.get("DISPLAY") or ":1"

def send_popup(title: str, body: str, urgency: str = "normal", timeout_ms: int = 8000) -> bool:
    bus = f"unix:path=/run/user/{DESKTOP_UID}/bus"
    if os.geteuid() != 0:
        env = dict(os.environ)
        env.setdefault("DBUS_SESSION_BUS_ADDRESS", bus)
        env.setdefault("DISPLAY", DESKTOP_DISPLAY)
        try:
            subprocess.run(["notify-send","-u",urgency,"-t",str(int(timeout_ms)), title, body], check=False, env=env)
            return True
        except Exception: return False
    else:
        cmd = ["sudo","-u",DESKTOP_USER,"-H","env", f"DISPLAY={DESKTOP_DISPLAY}", f"DBUS_SESSION_BUS_ADDRESS={bus}",
               "notify-send","-u",urgency,"-t",str(int(timeout_ms)), title, body]
        try:
            subprocess.run(cmd, check=False)
            return True
        except Exception: return False

START_TS = time.time(); _last_email_ts = 0.0

def tee(msg: str, etype: Optional[str]=None, detail: Optional[str]=None):
    print(msg)
    if _termlog and etype: _termlog.write_event(etype, detail if detail is not None else msg)

def send_email(subject: str, body: str):
    def _send_task():
        global _last_email_ts
        if not EMAIL_NOTIFY: return
        if not (SMTP_HOST and SMTP_PORT and SMTP_USER and SMTP_PASS and EMAIL_TO): return
        try:
            msg = EmailMessage()
            msg["From"] = EMAIL_FROM
            msg["To"] = EMAIL_TO
            msg["Subject"] = f"[AIIDS] {subject}"
            msg.set_content(body)
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as s:
                s.ehlo()
                if SMTP_USE_TLS: s.starttls()
                s.login(SMTP_USER, SMTP_PASS)
                s.send_message(msg)
            tee("[EMAIL] sent", "email", "sent")
        except Exception as e:
            tee(f"[EMAIL] failed: {e}", "email", f"failed: {e}")

    global _last_email_ts
    now = time.time()
    if (now - _last_email_ts) < EMAIL_MIN_COOLDOWN_SEC:
        return
    _last_email_ts = now 
    threading.Thread(target=_send_task, daemon=True).start()

# ===================== Stats Class =====================
class RealtimeStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.c_tp = 0; self.c_tn = 0; self.c_fp = 0; self.c_fn = 0
        self.c_skipped = 0; self.c_total = 0; self.c_alerts = 0
        self.y_true = [] 
        self.y_prob = []

    def update(self, pred_label, true_label, prob):
        with self.lock:
            self.c_total += 1
            if pred_label == 1: self.c_alerts += 1
            if true_label is None:
                self.c_skipped += 1
                return
            self.y_true.append(true_label)
            self.y_prob.append(prob)
            if true_label == 1 and pred_label == 1: self.c_tp += 1
            elif true_label == 0 and pred_label == 0: self.c_tn += 1
            elif true_label == 0 and pred_label == 1: self.c_fp += 1
            elif true_label == 1 and pred_label == 0: self.c_fn += 1

    def generate_report(self):
        with self.lock:
            uptime_min = int((time.time() - self.start_time) / 60)
            acc = 0.0; prec = 0.0; rec = 0.0; f1 = 0.0; auc = 0.0
            
            denom_acc = (self.c_tp + self.c_tn + self.c_fp + self.c_fn)
            total_correct = (self.c_tp + self.c_tn)
            
            if denom_acc > 0: acc = total_correct / denom_acc
            if (self.c_tp + self.c_fp) > 0: prec = self.c_tp / (self.c_tp + self.c_fp)
            if (self.c_tp + self.c_fn) > 0: rec = self.c_tp / (self.c_tp + self.c_fn)
            if (prec + rec) > 0: f1 = 2 * (prec * rec) / (prec + rec)

            if len(self.y_true) > 0:
                try:
                    if len(set(self.y_true)) > 1: auc = roc_auc_score(self.y_true, self.y_prob)
                    else: auc = 0.0
                except: auc = 0.0
            
            correct_ratio = f"{total_correct}/{denom_acc}"
            correct_pct = (total_correct/denom_acc)*100 if denom_acc > 0 else 0.0

            msg = (f"Uptime:{uptime_min}m | Alerts={self.c_alerts} | "
                   f"TP={self.c_tp} TN={self.c_tn} FP={self.c_fp} FN={self.c_fn} | "
                   f"Correct={correct_ratio} ({correct_pct:.2f}%) | "
                   f"Acc={acc:.4f} AUC={auc:.4f}")
            return msg

# ===================== Loggers =====================
class TerminalCsvLogger:
    HEADER = ["type","timestamp","flow_id","flow_packets","src","sport","dst","dport","proto","payload_len","single_prob","single_pred","flow_prob","flow_pred","true","latency","detail"]
    def __init__(self, path: str):
        self.path = path; os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fh = open(self.path, "a", newline="", encoding="utf-8"); self._w = csv.writer(self._fh)
        if os.stat(self.path).st_size == 0: self._w.writerow(self.HEADER); self._fh.flush()
    def write_packet(self, meta, sp, sl, fp, fl, t_val, lat, fpk):
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        fid = f"{meta.get('src','')}-{meta.get('sport','')}-{meta.get('dst','')}-{meta.get('dport','')}-{meta.get('proto','')}"
        self._w.writerow(["packet", ts, fid, fpk, meta.get("src",""), meta.get("sport",""), meta.get("dst",""), meta.get("dport",""), meta.get("proto",""), meta.get("plen",""), f"{sp:.9g}", sl, f"{fp:.9g}", fl, (t_val or ""), f"{lat:.3f}", ""])
        self._fh.flush()
    def write_event(self, etype, detail):
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        self._w.writerow([etype, ts, "", "", "", "", "", "", "", "", "", "", "", "", "", "", detail]); self._fh.flush()
    def close(self): self._fh.close()

class FlowTimeoutLogger:
    HEADER = ["timestamp","flow_id","packets","payload_len","flow_prob","flow_pred","last_seen_iso","src","sport","dst","dport","proto"]
    def __init__(self, path: str):
        self.path = path; os.makedirs(os.path.dirname(path) or ".", exist_ok=True); self._lock = threading.Lock()
        self._fh = open(self.path, "a", newline="", encoding="utf-8"); self._w = csv.writer(self._fh)
        if os.stat(self.path).st_size == 0: self._w.writerow(self.HEADER); self._fh.flush()
    def write_flow_evicted(self, flow_k, entry):
        src, sport, dst, dport, proto = flow_k
        fid = f"{src}-{sport}-{dst}-{dport}-{proto}"
        ts_iso = datetime.datetime.now().isoformat(timespec="seconds")
        row = [ts_iso, fid, entry.get("packets",0), len(entry.get("payload", b"")), f"{entry.get('flow_prob',0.0):.9g}", entry.get("flow_pred",0), datetime.datetime.fromtimestamp(entry.get("last_ts",0.0)).isoformat(timespec="seconds"), src, sport, dst, dport, proto]
        with self._lock: self._w.writerow(row); self._fh.flush()
    def close(self): self._fh.close()

def _norm_proto(x):
    if x is None: return "IP"
    s = str(x).upper().strip()
    return "TCP" if s=="6" or s=="TCP" else ("UDP" if s=="17" or s=="UDP" else "IP")

def _load_truth_csv(path):
    mp = {}
    if not (path and os.path.exists(path)): return mp
    with open(path, newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            try:
                src=row.get("src") or row.get("src_ip"); dst=row.get("dst") or row.get("dst_ip")
                sport=int(row.get("sport") or row.get("src_port")); dport=int(row.get("dport") or row.get("dst_port"))
                proto=_norm_proto(row.get("proto") or row.get("protocol"))
                lbl_s=str(row.get("label") or row.get("true")).lower()
                lab = 1 if lbl_s in ("1","malicious","mal") else (0 if lbl_s in ("0","benign") else None)
                if lab is not None: mp[(src,sport,dst,dport,proto)] = lab
            except: pass
    return mp

# ===================== Main =====================
_termlog = None; _flowlogger = None
_flow_table = {}; _flow_lock = threading.Lock()
_stats = RealtimeStats() 

def _cleanup_flows_logic(timeout):
    now = time.time(); removed = []
    with _flow_lock:
        keys = list(_flow_table.keys())
    with _flow_lock:
        for k in keys:
            v = _flow_table.get(k)
            if v and (now - v.get("last_ts", 0.0) > timeout):
                removed.append((k, v))
                _flow_table.pop(k, None)

    if _flowlogger:
        for k, v in removed: _flowlogger.write_flow_evicted(k, v)

def run_main():
    global _termlog, _flowlogger
    _termlog = TerminalCsvLogger(TERM_LOG_CSV_PATH)
    _flowlogger = FlowTimeoutLogger(FLOW_TIMEOUT_LOG_PATH)

    print(f"[INFO] Loading model: {MODEL_PATH}")
    mdl = tf.keras.models.load_model(MODEL_PATH, compile=False)
    meta = joblib.load(META_PATH)
    
    global FIXED_LEN
    FIXED_LEN = int(meta.get("fixed_len", FIXED_LEN))
    window_mode = meta.get("window_mode", "head")
    max_flow_payload = MAX_FLOW_PAYLOAD or FIXED_LEN
    
    # ---------------------------------------------------------
    # Logic เลือก Threshold
    # ---------------------------------------------------------
    thr_train = float(meta.get("threshold", 0.5))
    
    if USE_MANUAL_THRESHOLD:
        active_threshold = float(MANUAL_THRESHOLD_VAL)
        thresh_mode = "MANUAL (Custom)"
    else:
        active_threshold = thr_train
        thresh_mode = "TRAIN (Auto .joblib)"

    print(f"-"*60)
    print(f"[CONFIG] Threshold Mode : {thresh_mode}")
    print(f"[CONFIG] Active Value   : {active_threshold:.20f}")
    if USE_MANUAL_THRESHOLD:
        print(f"         (Original Train: {thr_train:.20f})")
    print(f"-"*60)

    @tf.function
    def _model_call(x): return mdl(x, training=False)
    
    try:
        warm = np.zeros((1, FIXED_LEN), dtype=np.int32)
        _ = _model_call(tf.constant(warm))
    except: pass

    mp_exact = _load_truth_csv(TRUTH_CSV) if USE_TRUTH else {}
    work_q = queue.Queue(); stop_flag = threading.Event()
    
    def make_x(payload_bytes):
        return to_fixed_len_uint8(payload_bytes, L=FIXED_LEN, mode=window_mode)

    def _predict_from_arr(arr):
        if arr.ndim == 1: arr = arr[None, ...]
        y = _model_call(tf.constant(arr)).numpy().ravel()
        return float(y[0]) if y.size == 1 else float(y[-1])

    # --- Thread 1: Inference Worker ---
    def infer_worker():
        while not stop_flag.is_set():
            try:
                m, p_bytes, pkt_ts = work_q.get(timeout=0.2)
                
                # Single Packet Pred
                x_single = make_x(p_bytes)
                single_prob = _predict_from_arr(x_single)
                single_pred = 1 if single_prob >= active_threshold else 0

                # Flow Pred
                flow_k = (m["src"], int(m["sport"]), m["dst"], int(m["dport"]), _norm_proto(m["proto"]))
                with _flow_lock:
                    entry = _flow_table.get(flow_k)
                    if not entry:
                        entry = {"payload": b"", "last_ts": pkt_ts, "packets": 0, "flow_prob": 0.0}
                        _flow_table[flow_k] = entry
                    
                    if window_mode == "tail": new_pl = (entry["payload"] + p_bytes)[-max_flow_payload:]
                    else: new_pl = (entry["payload"] + p_bytes)[:max_flow_payload]
                    entry["payload"] = new_pl; entry["last_ts"] = pkt_ts; entry["packets"] += 1
                    flow_packets = entry["packets"]

                x_flow = make_x(new_pl)
                flow_prob = _predict_from_arr(x_flow)
                flow_pred = 1 if flow_prob >= active_threshold else 0
                
                with _flow_lock:
                    if flow_k in _flow_table:
                        _flow_table[flow_k]["flow_prob"] = flow_prob
                        _flow_table[flow_k]["flow_pred"] = flow_pred 

                # Truth Checking
                t_lbl_str = None; t_val_int = None
                chk_key = flow_k
                if chk_key not in mp_exact:
                     rev_key = (m["dst"], int(m["dport"]), m["src"], int(m["sport"]), flow_k[4])
                     if rev_key in mp_exact: chk_key = rev_key
                if chk_key in mp_exact:
                    t_val_int = mp_exact[chk_key]
                    t_lbl_str = "malicious" if t_val_int==1 else "benign"

                # Update Stats (ใช้ flow_pred เป็นหลักในการวัดผล)
                _stats.update(flow_pred, t_val_int, flow_prob)

                lat = time.time() - pkt_ts 
                _termlog.write_packet(m, single_prob, single_pred, flow_prob, flow_pred, t_lbl_str, lat, flow_packets)

                # แจ้งเตือนในหน้าจอ (ตรงนี้ที่เปลี่ยนเป็น ALERT)
                if (not PRINT_ONLY_ALERTS) or flow_pred==1 or single_pred==1:
                    prefix = "[ALERT]" if (flow_pred==1 or single_pred==1) else "[PKT]"
                    print(f"{prefix} {m['src']}->{m['dst']} | Prob={flow_prob:.2e} | Pred={flow_pred}")

                # ส่ง Popup/Email
                if flow_pred == 1 or single_pred == 1:
                    send_popup("🚨 AI-IDS Alert", f"{m['src']} -> {m['dst']}\nProb: {max(single_prob, flow_prob):.2e}", urgency="critical")
                    send_email("Malware Detected", f"Src: {m['src']}\nDst: {m['dst']}\nProb: {max(single_prob, flow_prob):.2e}")

            except queue.Empty: pass

    threading.Thread(target=infer_worker, daemon=True).start()

    # --- Thread 2: Cleanup Worker ---
    def cleanup_loop():
        while not stop_flag.is_set():
            time.sleep(FLOW_CLEANUP_INTERVAL)
            try:
                _cleanup_flows_logic(FLOW_TIMEOUT_SEC)
            except Exception as e:
                print(f"[WARN] Cleanup failed: {e}")

    threading.Thread(target=cleanup_loop, daemon=True).start()

    # --- Thread 3: Reporter ---
    def reporter_loop():
        while not stop_flag.is_set():
            time.sleep(REPORT_INTERVAL_SEC)
            msg = _stats.generate_report()
            
            print("\n" + "="*80)
            print(f"📊 STATISTICS REPORT ({thresh_mode})")
            print(f"   [INFO] {msg}")
            print("="*80 + "\n")
            
            _termlog.write_event("info", msg)

    threading.Thread(target=reporter_loop, daemon=True).start()

    def on_packet(pkt):
        try:
            if IP not in pkt: return
            if SRC_IP and pkt[IP].src != SRC_IP: return
            if DST_IP and pkt[IP].dst != DST_IP: return
            if TCP in pkt: proto="TCP"; payload=bytes(pkt[TCP].payload); sp=pkt[TCP].sport; dp=pkt[TCP].dport
            elif UDP in pkt: proto="UDP"; payload=bytes(pkt[UDP].payload); sp=pkt[UDP].sport; dp=pkt[UDP].dport
            else: proto="IP"; payload=bytes(pkt[Raw].load) if Raw in pkt else b""; sp=0; dp=0
            meta_i={"src":pkt[IP].src,"dst":pkt[IP].dst,"sport":sp,"dport":dp,"proto":proto,"plen":len(payload)}
            work_q.put((meta_i, payload, time.time()))
        except: pass

    print(f"[INFO] Sniffing {IFACE} ...")
    sniffer = AsyncSniffer(iface=IFACE, filter=BPF_FILTER, prn=on_packet, store=False, promisc=True)
    sniffer.start()
    
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        sniffer.stop(); stop_flag.set()
        _termlog.close(); _flowlogger.close()

if __name__ == "__main__":
    run_main()