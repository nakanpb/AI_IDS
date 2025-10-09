#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scapy.all import PcapReader, PcapWriter, IP, TCP, UDP, Raw
import pandas as pd
import csv, base64, sys

# ======== กำหนด PATH และค่าต่างๆ =========
PCAP_FILE       = "/home/intgan/Desktop/jek/benign/chunks/be1.pcap"
CSV_LABELS_FILE = "/home/intgan/Desktop/jek/benign/Monday.csv"
OUT_PCAP_FILE   = "/home/intgan/Desktop/jek/benign/realbenign1.pcap"
OUT_CSV_FILE    = "/home/intgan/Desktop/jek/benign/realbenign1.csv"
PAYLOAD_MAX     = 512        # ความยาว payload สูงสุด
ONLY_PAYLOAD    = False      # True = เก็บเฉพาะ packet ที่มี payload
TARGET_GB       = 2.5        # None = ไม่จำกัดขนาดไฟล์, ตัวเลข = GB ที่ต้องการ
# =========================================

def build_flow_id(src, sport, dst, dport, proto):
    return f"{src}-{dst}-{sport}-{dport}-{proto}".upper()

def load_benign_flow_ids(csv_path: str) -> set:
    benign = set()
    for chunk in pd.read_csv(csv_path, chunksize=200_000):
        cols_norm = {c.strip(): c for c in chunk.columns}
        if 'Flow ID' not in cols_norm or 'Label' not in cols_norm:
            raise ValueError("CSV ไม่มีคอลัมน์ 'Flow ID' หรือ 'Label'")
        fid_col = cols_norm['Flow ID']; lbl_col = cols_norm['Label']

        fids = chunk.loc[
            chunk[lbl_col].astype(str).str.upper() == 'BENIGN', fid_col
        ].astype(str)

        for fid in fids:
            parts = [p.strip() for p in fid.split('-')]
            if len(parts) != 5:
                continue
            s_ip, d_ip, s_p, d_p, proto_raw = parts
            # ทำ proto เป็นตัวเลข 6/17
            if proto_raw.upper() in ('TCP', '6'):
                pcode = 6
            elif proto_raw.upper() in ('UDP', '17'):
                pcode = 17
            else:
                continue
            try:
                s_p = int(s_p); d_p = int(d_p)
            except:
                continue
            # เก็บสองทิศทางด้วย proto เป็นตัวเลข
            benign.add((s_ip, s_p, d_ip, d_p, pcode))
            benign.add((d_ip, d_p, s_ip, s_p, pcode))
    return benign

def main():
    target_bytes = None
    if TARGET_GB is not None:
        target_bytes = int(TARGET_GB * (1024**3))
        print(f"[i] จะหยุดเมื่อเขียนถึง ~{TARGET_GB} GB ≈ {target_bytes:,} bytes")

    print(f"[i] โหลด Flow IDs (BENIGN) จาก: {CSV_LABELS_FILE}")
    benign_flows = load_benign_flow_ids(CSV_LABELS_FILE)
    print(f"[i] พบ BENIGN flow IDs = {len(benign_flows):,}")

    csv_fields = [
        "timestamp","src_ip","src_port","dst_ip","dst_port",
    "protocol","packet_length","payload","label"
    ]
    fout = open(OUT_CSV_FILE, "w", newline="")
    writer = csv.DictWriter(fout, fieldnames=csv_fields)
    writer.writeheader()

    kept_pkts = 0
    total_pkts = 0
    written_bytes = 0

    with PcapReader(PCAP_FILE) as pr, PcapWriter(OUT_PCAP_FILE, append=False, sync=True) as pw:
        for pkt in pr:
            total_pkts += 1
            
            try:
                if IP not in pkt:
                    continue
                ip = pkt[IP]

                if TCP in pkt:
                    pcode = 6
                    sport, dport = pkt[TCP].sport, pkt[TCP].dport
                elif UDP in pkt:
                    pcode = 17
                    sport, dport = pkt[UDP].sport, pkt[UDP].dport
                else:
                    continue
                key = (ip.src, sport, ip.dst, dport, pcode)
                if key not in benign_flows:
                    # ลองกลับทิศด้วยตัวเลขเหมือนกัน (ถ้าคุณไม่ได้บันทึกสองทิศตั้งแต่ตอนโหลด)
                    key_rev = (ip.dst, dport, ip.src, sport, pcode)
                    if key_rev not in benign_flows:
                        continue
                if total_pkts < 3:
                    print("[DBG] key=", key, " in_set=", key in benign_flows)


                payload_b64 = ""
                if Raw in pkt and pkt[Raw].load:
                    raw_bytes = bytes(pkt[Raw].load)
                    trimmed = raw_bytes[:PAYLOAD_MAX].ljust(PAYLOAD_MAX, b"\x00")
                    payload_b64 = base64.b64encode(trimmed).decode("utf-8")
                elif ONLY_PAYLOAD:
                    continue

                pw.write(pkt)
                kept_pkts += 1
                pkt_size = len(bytes(pkt))
                written_bytes += pkt_size
                proto_str = "TCP" if pcode == 6 else "UDP"


                writer.writerow({
                    "timestamp": float(pkt.time),
                    "src_ip": ip.src,
                    "dst_ip": ip.dst,
                    "src_port": sport,
                    "dst_port": dport,
                    "protocol": proto_str,
                    "packet_length": pkt_size,
                    "payload": payload_b64,
                    "label": "benign"
                })

                if kept_pkts % 100000 == 0:
                    if target_bytes:
                        print(f"[i] kept={kept_pkts:,} | out≈{written_bytes:,} bytes / target {target_bytes:,}")
                    else:
                        print(f"[i] kept={kept_pkts:,} | out≈{written_bytes:,} bytes")

                if target_bytes and written_bytes >= target_bytes:
                    print(f"[✓] ถึงขนาดเป้าหมายแล้ว: {written_bytes:,} bytes")
                    break

            except Exception:
                continue

    fout.close()
    print(f"[done] อ่านทั้งหมด {total_pkts:,} pkts | เขียน {kept_pkts:,} pkts")
    if target_bytes:
        print(f"[size] ขนาดเอาต์พุตโดยประมาณ: {written_bytes:,} bytes")
    print(f"[out] PCAP: {OUT_PCAP_FILE}")
    print(f"[out] CSV : {OUT_CSV_FILE}")

if __name__ == "__main__":
    main()
