#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scapy.all import PcapReader, IP
from datetime import datetime, timezone, timedelta

PCAP_FILE = "/home/intgan/Desktop/jek/trickbot(csv)/trickbot_239-1.pcap"

# ── เวลาใน README (ปรับโซนเวลาให้ตรง: Mar 8, 2017 ยังเป็น CET = UTC+1)
CET = timezone(timedelta(hours=1))
START_TIME_STR    = "2017-03-08 15:24:16"  # started win10
INFECT_TIME_STR   = "2017-03-08 15:29:46"  # infected

start_abs = datetime.strptime(START_TIME_STR,  "%Y-%m-%d %H:%M:%S").replace(tzinfo=CET)
infect_abs = datetime.strptime(INFECT_TIME_STR, "%Y-%m-%d %H:%M:%S").replace(tzinfo=CET)

# 1) หาเวลาแพ็กเก็ตแรกจาก PCAP (จะเจอปี 1970 ในชุดนี้ ซึ่งไม่ต้องใช้เป็นฐาน)
with PcapReader(PCAP_FILE) as r:
    first_ip_dt = None
    for p in r:
        if IP in p:
            first_ip_dt = datetime.fromtimestamp(float(p.time), tz=CET)
            break
if first_ip_dt is None:
    raise SystemExit("ไม่พบแพ็กเก็ต IPv4 ในไฟล์")

print("🕒 Raw start of capture (PCAP):", first_ip_dt.strftime("%Y-%m-%d %H:%M:%S %Z"))

# 2) คำนวณค่า relative ตรงจาก README: infected - started
infection_rel_sec = (infect_abs - start_abs).total_seconds()

print("🕒 Corrected start (README):   ", start_abs.strftime("%Y-%m-%d %H:%M:%S %Z"))
print("🦠 Infection (README):        ", infect_abs.strftime("%Y-%m-%d %H:%M:%S %Z"))
print(f"⏱️ Infection relative:         {infection_rel_sec:.6f} seconds")

print(f"\n👉 INFECTION_EPOCH_SECONDS = {infection_rel_sec:.6f}")
