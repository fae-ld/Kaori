
import os

# Prints current work directory
print(os.getcwd())

from satoru import process_journal_entry

# ENTRY 1: Fokus pada Temporal Relative & Multimodality (Gojo sebagai Dosen)
ENTRY_1 = """
Tanggal: 2026-03-08
Kemarin aku ketemu Satoru di kampus. Dia lagi mode serius banget sebagai Dosen tamu 
buat kelas Informatika. Padahal tiga hari yang lalu, dia cuma kelihatan kayak 
pengangguran yang main catur di taman. Dia ngasih tahu soal teknik 'Six Eyes' 
yang katanya adalah warisan turun-temurun Klan Gojo.
"""

# ENTRY 2: Fokus pada Entity Resolution & Temporal Chaos (Gojo sebagai Kepala Klan)
ENTRY_2 = """
Tanggal: 2026-03-15
Seminggu setelah pertemuan di kampus, si Blindfolded Sorcerer itu muncul lagi di 
depan rumah. Kali ini dia dateng atas nama Kepala Klan Gojo untuk ngebahas 
masa depan Megumi. Padahal di tahun 2023, dia bilang dia udah nggak mau 
urusan lagi sama politik klan. Aneh banget ngeliat dia ganti-ganti peran gini.
"""

def run_test():
    print("🚀 Starting Six-Eyes Stress Test...")
    
    entries = [ENTRY_1, ENTRY_2]
    for i, entry in enumerate(entries):
        print(f"\nProcessing Entry {i+1}...")
        process_journal_entry(entry)
        
    print("\n✅ Test Complete. Check Neo4j and look for:")
    print("1. Only ONE node for 'Gojo Satoru' (Entity Resolution)")
    print("2. Roles property containing both 'Dosen' and 'Kepala Klan' (Multimodality)")
    print("3. Dates like '2026-03-07' and '2026-03-05' in history (Temporal Normalization)")

if __name__ == "__main__":
    run_test()