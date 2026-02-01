#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-IDS Realtime + FlowTimeoutLogger
(Updated to match the specific training code provided)
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

# *** ตรวจสอบ path ให้ถูกต้อง ***
MODEL_PATH = "/home/intgan/Desktop/jek/train/final24.final.20260120-192327.keras" 
META_PATH  = "/home/intgan/Desktop/jek/train/preprocess_meta.20260120-192327.joblib"
TERM_LOG_CSV_PATH = "/home/intgan/Desktop/jek/log/final24_outputlog2.csv"

# Flow timeout log (evicted flows)
FLOW_TIMEOUT_SEC = 120
FLOW_TIMEOUT_LOG_PATH = "/home/intgan/Desktop/jek/log/final/24_flow_timeout_log.csv"
MAX_FLOW_PAYLOAD = None

# [1] ไม่ต้อง Import Custom Layer แล้ว เพราะโค้ดเทรนไม่ได้ใช้
# sys.path.append("/home/intgan/Desktop/jek/train") 

BATCH_SIZE  = 1
THRESHOLD   = "auto"
PRINT_ONLY_ALERTS = True

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
MALICIOUS_SRCS = ""
BENIGN_SRCS    = ""
DEFAULT_TRUE_LABEL = None

# ===================== Preprocess Logic =====================
# [2] ต้องก๊อปปี้ฟังก์ชันนี้มาจากโค้ดเทรนโดยตรง เพื่อให้ Logic ตรงกัน (มีการ +1)
FIXED_LEN = 1460 

def to_fixed_len_uint8(b: bytes, L=FIXED_LEN, mode="head"):
    arr = np.frombuffer(b, dtype=np.uint8)
    n = len(arr)
    
    # ขยับค่า Byte จาก 0-255 เป็น 1-256 (ตามโค้ดเทรน)
    arr_shifted = arr.astype(np.int32) + 1
    
    if n >= L:
        if mode == "tail": return arr_shifted[-L:]
        return arr_shifted[:L]
    
    # สร้าง array ค่า 0 (Padding)
    out = np.zeros(L, dtype=np.int32)
    out[:n] = arr_shifted
    return out

# ===================== Popup helper =====================
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

# ===================== Email / Logging / Truth =====================
# (ส่วนนี้เหมือนเดิม ย่อไว้เพื่อประหยัดพื้นที่)
START_TS = time.time(); _last_email_ts = 0.0
def tee(msg: str, etype: Optional[str]=None, detail: Optional[str]=None):
    print(msg)
    if _termlog and etype: _termlog.write_event(etype, detail if detail is not None else msg)

# แก้ไขฟังก์ชัน send_email เดิม
def send_email(subject: str, body: str):
    # ฟังก์ชันตัวจริงที่จะถูกรันใน Thread แยก
    def _send_task():
        global _last_email_ts
        if not EMAIL_NOTIFY: return
        now = time.time()
        # เช็ค Cooldown ใน Thread แยก (หรือเช็คก่อนเรียกก็ได้)
        # แต่เพื่อความชัวร์ เช็คข้างนอกก่อนเรียกดีกว่า เพื่อไม่ให้สร้าง Thread พร่ำเพรื่อ
        
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
            # _last_email_ts = now # ย้ายไปอัพเดทข้างนอกดีกว่าเพื่อกัน race condition
            tee("[EMAIL] sent", "email", "sent")
        except Exception as e:
            tee(f"[EMAIL] failed: {e}", "email", f"failed: {e}")

    # เช็ค Cooldown ตรงนี้ก่อนเลย จะได้ไม่เสียเวลาสร้าง Thread ถ้ายังไม่ถึงเวลา
    global _last_email_ts
    now = time.time()
    if (now - _last_email_ts) < EMAIL_MIN_COOLDOWN_SEC:
        return
    
    _last_email_ts = now # อัพเดทเวลาทันทีที่ตัดสินใจส่ง
    
    # *** จุดสำคัญ: สั่งรัน _send_task ใน Thread ใหม่ทันที โดยไม่รอ ***
    threading.Thread(target=_send_task, daemon=True).start()
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
                # Adjust column names as needed
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

def _cleanup_flows(timeout):
    now = time.time(); removed = []
    with _flow_lock:
        for k, v in list(_flow_table.items()):
            if now - v.get("last_ts", 0.0) > timeout: removed.append((k, v)); _flow_table.pop(k, None)
    if _flowlogger:
        for k, v in removed: _flowlogger.write_flow_evicted(k, v)

def run_main():
    global _termlog, _flowlogger
    _termlog = TerminalCsvLogger(TERM_LOG_CSV_PATH)
    _flowlogger = FlowTimeoutLogger(FLOW_TIMEOUT_LOG_PATH)

    # [3] โหลดโมเดลแบบปกติ (ลบ custom_objects ออก เพราะโค้ดเทรนนี้ใช้ layer มาตรฐาน)
    print(f"[INFO] Loading model: {MODEL_PATH}")
    mdl = tf.keras.models.load_model(MODEL_PATH, compile=False)

    print(f"[INFO] Loading meta : {META_PATH}")
    meta = joblib.load(META_PATH)
    
    # ดึงค่า Config จาก Meta (หรือใช้ค่า Default ถ้าไม่มี)
    global FIXED_LEN
    FIXED_LEN = int(meta.get("fixed_len", FIXED_LEN))
    window_mode = meta.get("window_mode", "head")
    thr_auto = float(meta.get("threshold", 0.5))
    threshold = thr_auto if THRESHOLD == "auto" else float(THRESHOLD)
    max_flow_payload = MAX_FLOW_PAYLOAD or FIXED_LEN

    print(f"[CONFIG] Fixed Len={FIXED_LEN}, Window={window_mode}, Threshold={threshold:.4f}")

    @tf.function
    def _model_call(x): return mdl(x, training=False)

    # warm-up
    try:
        warm = np.zeros((1, FIXED_LEN), dtype=np.int32)
        _ = _model_call(tf.constant(warm))
        print("[INFO] Model warm-up done.")
    except Exception as e: print("[WARN] warm-up failed:", e)

    mp_exact = _load_truth_csv(TRUTH_CSV) if USE_TRUTH else {}
    
    work_q = queue.Queue(); stop_flag = threading.Event()
    
    def make_x(payload_bytes):
        # เรียกใช้ฟังก์ชันที่ประกาศไว้ข้างบน (ซึ่งมี Logic +1 ตรงกับโค้ดเทรน)
        return to_fixed_len_uint8(payload_bytes, L=FIXED_LEN, mode=window_mode)

    def _predict_from_arr(arr):
        if arr.ndim == 1: arr = arr[None, ...]
        y = _model_call(tf.constant(arr)).numpy().ravel()
        p = float(y[0]) if y.size == 1 else float(y[-1])
        return p

    def infer_worker():
        while not stop_flag.is_set():
            try:
                m, p_bytes, pkt_ts = work_q.get(timeout=0.2)
                
                # Single Packet Inference
                x_single = make_x(p_bytes)
                single_prob = _predict_from_arr(x_single)
                single_pred = int(single_prob >= threshold)

                # Flow Inference
                flow_k = (m["src"], int(m["sport"]), m["dst"], int(m["dport"]), _norm_proto(m["proto"]))
                with _flow_lock:
                    entry = _flow_table.get(flow_k)
                    if not entry:
                        entry = {"payload": b"", "last_ts": pkt_ts, "packets": 0, "flow_prob": 0.0}
                        _flow_table[flow_k] = entry
                    
                    if window_mode == "tail":
                        new_pl = (entry["payload"] + p_bytes)[-max_flow_payload:]
                    else:
                        new_pl = (entry["payload"] + p_bytes)[:max_flow_payload]
                        
                    entry["payload"] = new_pl; entry["last_ts"] = pkt_ts; entry["packets"] += 1
                    flow_packets = entry["packets"]

                x_flow = make_x(new_pl)
                flow_prob = _predict_from_arr(x_flow)
                flow_pred = int(flow_prob >= threshold)
                
                with _flow_lock:
                    if flow_k in _flow_table:
                        _flow_table[flow_k]["flow_prob"] = flow_prob
                        _flow_table[flow_k]["flow_pred"] = flow_pred

                # Logging & Display
                # (Simple Truth checking logic here - simplified for brevity)
                t_lbl = None
                if flow_k in mp_exact: t_lbl = "malicious" if mp_exact[flow_k]==1 else "benign"
                elif (m["dst"], int(m["dport"]), m["src"], int(m["sport"]), flow_k[4]) in mp_exact:
                     t_lbl = "malicious" if mp_exact[(m["dst"], int(m["dport"]), m["src"], int(m["sport"]), flow_k[4])]==1 else "benign"

                lat = time.time() - pkt_ts # approx latency
                _termlog.write_packet(m, single_prob, single_pred, flow_prob, flow_pred, t_lbl, lat, flow_packets)

                if (not PRINT_ONLY_ALERTS) or single_pred==1 or flow_pred==1:
                    print(f"[DETECT] {m['src']}:{m['sport']}->{m['dst']}:{m['dport']} | S={single_prob:.4f}({single_pred}) F={flow_prob:.4f}({flow_pred})")

                if single_pred == 1 or flow_pred == 1:
                    send_popup("🚨 AI-IDS Alert", f"{m['src']} -> {m['dst']}\nProb: {max(single_prob,flow_prob):.4f}", urgency="critical")
                    send_email("Malware Detected", f"Src: {m['src']}\nDst: {m['dst']}\nProb: {max(single_prob,flow_prob)}")

            except queue.Empty: pass
            _cleanup_flows(FLOW_TIMEOUT_SEC)

    threading.Thread(target=infer_worker, daemon=True).start()

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
