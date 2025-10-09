
from scapy.all import PcapReader, IP, TCP, UDP

import csv
import base64

PCAP_FILE = "zeus25-6.pcap"
CSV_FILE = "zeus25-6.csv"
MALICIOUS_IP = "10.0.2.103"
MAX_PAYLOAD_LEN = 512  # จำกัดขนาด payload


def process_pcap_to_csv(pcap_file, output_csv):
    with open(output_csv, mode='w', newline='') as csvfile:
        fieldnames = ["timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "packet_length", "payload", "label"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        count = 0

        with PcapReader(pcap_file) as pcap_reader:
            for pkt in pcap_reader:
                try:
                    if IP in pkt:
                        src_ip = pkt[IP].src
                        dst_ip = pkt[IP].dst
                        proto = pkt[IP].proto
                        pkt_len = len(pkt)
                        time = pkt.time  # ใช้เวลาที่ Scapy ดึงมาได้

                        sport = dport = None
                        if TCP in pkt:
                            sport = pkt[TCP].sport
                            dport = pkt[TCP].dport
                        elif UDP in pkt:
                            sport = pkt[UDP].sport
                            dport = pkt[UDP].dport

                        label = "malicious" if (src_ip == MALICIOUS_IP or dst_ip == MALICIOUS_IP) else "benign"

                        # payload extraction
                        raw_bytes = bytes(pkt[TCP].payload) if TCP in pkt else bytes(pkt[UDP].payload) if UDP in pkt else b""

                        if raw_bytes:
                            padded_bytes = raw_bytes[:MAX_PAYLOAD_LEN].ljust(MAX_PAYLOAD_LEN, b'\x00')  # เติม 0x00 ถ้ายังไม่ครบ
                            payload_str = base64.b64encode(padded_bytes).decode('utf-8')
                        else:
                            payload_str = ""


                        writer.writerow({
                            "timestamp": time,
                            "src_ip": src_ip,
                            "dst_ip": dst_ip,
                            "src_port": sport,
                            "dst_port": dport,
                            "protocol": proto,
                            "packet_length": pkt_len,
                            "payload": payload_str,
                            "label": label
                        })

                        count += 1
                        if count % 10000 == 0:
                            print(f"เขียนไปแล้ว {count} แพ็กเก็ต")

                except Exception as e:
                    print(f"ข้ามแพ็กเก็ต: {e}")

    print("✅ เสร็จสิ้น! ส่งออกข้อมูลไปยัง", output_csv)
process_pcap_to_csv(PCAP_FILE, CSV_FILE)