import os
import torch
from symspellpy import SymSpell, Verbosity
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from indobenchmark import IndoNLGTokenizer
from functools import lru_cache

# Setup Environment
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ======= Step 1: Setup IndoBART Model =========
model_name = "indobenchmark/indobart"
tokenizer = IndoNLGTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ======= Step 2: Lazy Spell Correction (SymSpell) =========
sym_spell = None
dictionary_path = "Stage2/dictionary/frequency_dictionary_hundred.txt"

def lazy_load_symspell():
    global sym_spell
    if sym_spell is None:
        print("[📥] Memuat kamus ejaan (SymSpell)...")
        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# ======= Step 3: Koreksi Ejaan =========
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

# ======= Step 4: Grammar Correction dengan IndoBART =========
@lru_cache(maxsize=128)
def correct_grammar(text: str) -> str:
    text_with_lang_token = f"<ind> {text}"

    inputs = tokenizer(
        text_with_lang_token,
        return_tensors="pt",
        truncation=True,
        # padding="max_length",
        max_length=128
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=128,
        num_beams=5,
        early_stopping=True
    )

    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected

# ======= Step 5: Fungsi Utama =========
def main():
    current_text = ""
    print("=== Spell & Grammar Corrector Bahasa Indonesia ===")
    print("Ketik tambahan kata, atau 'reset' untuk mengosongkan, 'exit' untuk keluar.\n")

    while True:
        tambahan = input("➕ Tambahan teks: ").strip()

        if tambahan.lower() in ['exit', 'quit']:
            print("❌ Keluar dari program.")
            break
        elif tambahan.lower() == 'reset':
            current_text = ""
            print("🔄 Teks telah direset.\n")
            continue

        current_text += " " + tambahan
        current_text = current_text.strip()

        print("\n[1️⃣] Koreksi Ejaan...")
        fixed_spelling = correct_spelling(current_text)
        print(f"{fixed_spelling}")

        print("\n[2️⃣] Koreksi Grammar & Susun Kalimat...")
        fixed_grammar = correct_grammar(fixed_spelling)
        print(f"{fixed_grammar}\n")

# ======= Jalankan Program =========
if __name__ == "__main__":
    main()
