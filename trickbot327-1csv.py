from scapy.all import PcapReader, IP, TCP, UDP
import csv
import base64

PCAP_FILE = "/home/intgan/Desktop/jek/trickbot(csv)/trickbot_327-1.pcap"
CSV_FILE = "/home/intgan/Desktop/jek/trickbot(csv)/trickbot(327-1).csv"
INFECTED_IP = "192.168.1.121"
MAX_PAYLOAD_LEN = 512
INFECTION_EPOCH_SECONDS = 177.167926 # infection เริ่มหลังจาก 6937 วินาที

def process_pcap_ipv4_only(pcap_file, output_csv):
    with open(output_csv, mode='w', newline='') as csvfile:
        fieldnames = ["timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "packet_length", "payload", "label"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        count = 0

        with PcapReader(pcap_file) as pcap_reader:
            for pkt in pcap_reader:
                try:
                    if IP not in pkt:
                        continue  # ข้ามถ้าไม่ใช่ IPv4

                    ip_layer = pkt[IP]
                    src_ip = ip_layer.src
                    dst_ip = ip_layer.dst
                    pkt_len = len(pkt)
                    time = pkt.time

                    sport = dport = None
                    if TCP in pkt:
                        sport = pkt[TCP].sport
                        dport = pkt[TCP].dport
                        raw_bytes = bytes(pkt[TCP].payload)
                        protocol = "TCP"
                    elif UDP in pkt:
                        sport = pkt[UDP].sport
                        dport = pkt[UDP].dport
                        raw_bytes = bytes(pkt[UDP].payload)
                        protocol = "UDP"
                    else:
                        raw_bytes = b""
                        protocol = "Other"

                    # Payload
                    if raw_bytes:
                        padded_bytes = raw_bytes[:MAX_PAYLOAD_LEN].ljust(MAX_PAYLOAD_LEN, b'\x00')
                        payload_str = base64.b64encode(padded_bytes).decode('utf-8')
                    else:
                        payload_str = ""

                    # Label
                    if (src_ip == INFECTED_IP or dst_ip == INFECTED_IP) and time >= INFECTION_EPOCH_SECONDS:
                        label = "malicious"
                    else:
                        label = "benign"

                    writer.writerow({
                        "timestamp": time,
                        "src_ip": src_ip,
                        "dst_ip": dst_ip,
                        "src_port": sport,
                        "dst_port": dport,
                        "protocol": protocol,
                        "packet_length": pkt_len,
                        "payload": payload_str,
                        "label": label
                    })

                    count += 1
                    if count % 10000 == 0:
                        print(f"เขียนไปแล้ว {count} แพ็กเก็ต")

                except Exception as e:
                    print(f"ข้ามแพ็กเก็ต: {e}")

    print("เสร็จแล้ว! ส่งออก IPv4 packets ไปยัง:", output_csv)

process_pcap_ipv4_only(PCAP_FILE, CSV_FILE)
