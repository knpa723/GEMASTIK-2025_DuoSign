import os
import torch
from symspellpy import SymSpell, Verbosity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from functools import lru_cache
# from TTS.api import TTS  # Uncomment jika ingin pakai TTS

# ========== Step 1: Setup Rephrasing Model (IndoT5 / IndoBART) ==========
model_name = "cahya/t5-base-indonesian-summarization-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ========== Step 2: Lazy Spell Correction Setup ==========
sym_spell = None
dictionary_path = "Stage2/dictionary/frequency_dictionary_hundred.txt"

def lazy_load_symspell():
    global sym_spell
    if sym_spell is None:
        print("[📥] Memuat kamus ejaan (SymSpell)...")
        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# ========== Step 3: Koreksi Ejaan ==========
@lru_cache(maxsize=128)
def correct_spelling(text):
    lazy_load_symspell()
    words = text.strip().split()
    corrected = []
    for word in words:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        corrected_word = suggestions[0].term if suggestions else word
        corrected.append(corrected_word)
    return " ".join(corrected)

# ========== Step 4: Koreksi Grammar & Susun Ulang ==========
@lru_cache(maxsize=128)
def rephrase_sentence(text):
    prompt = f"perbaiki kalimat: {text}"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=128,
            num_beams=3,  # Kurangi beam untuk kecepatan
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.5
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ========== Step 5: Fungsi Utama ==========
def main():
    while True:
        user_input = input("\n📝 Masukkan kata-kata acak atau typo (contoh: saya jual mangga pasr maniss): ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("❌ Keluar dari program.")
            break

        print("\n[1️⃣] Koreksi Ejaan...")
        fixed_spelling = correct_spelling(user_input)
        print(f"Hasil koreksi ejaan: {fixed_spelling}")

        print("\n[2️⃣] Susun Ulang Kalimat...")
        rephrased = rephrase_sentence(fixed_spelling)
        print(f"Hasil kalimat akhir: {rephrased}")

# ========== Jalankan Program ==========
if __name__ == "__main__":
    main()
