from transformers import GPT2LMHeadModel, GPT2Tokenizer, set_seed
import torch
import time

# ------------------------------
# Text Generator (GPT2 Indo)
# ------------------------------
tokenizer = GPT2Tokenizer.from_pretrained("cahya/gpt2-small-indonesian-522M")
model = GPT2LMHeadModel.from_pretrained("cahya/gpt2-small-indonesian-522M")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_sentence(input_keywords, min_length=1, max_new_tokens=10, seed=42):
    
    set_seed(seed)
    
    input_text = " ".join(input_keywords)
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        min_length=min_length,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        no_repeat_ngram_size=2,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # Decode hasil dan ambil bagian baru setelah input
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    continuation = full_text[len(input_text):].strip()

    # Ambil hanya kalimat pertama
    first_sentence = continuation.split(".")[0].strip()

    return f"{input_text} {first_sentence}.".strip()

# ------------------------------
# Main Pipeline
# ------------------------------
if __name__ == "__main__":

    print("==== DuoSign : Club BEJO ====\n")
    print("ketik 1 untuk mulai, ketik 0 untuk keluar.\n")

    while True:
        cmd = input("[🕹️] Mulai generate? (1 = ya, 0 = keluar): ").strip()
        
        if cmd == "0":
            print("👋 Program dihentikan.")
            break
        elif cmd == "1":
            input_kata = input("Masukkan beberapa kata (misal: saya jual mangga): ").strip()
            keyword_list = input_kata.split()

            start_time = time.time()
            kalimat = generate_sentence(keyword_list)
            end_time = time.time()

            print(f"\n[📝] Kalimat hasil GPT2:\n{kalimat}")
            print(f"[⏱️] Waktu eksekusi: {end_time - start_time:.4f} detik\n")
        else:
            print("⚠️ Masukkan hanya 1 atau 0.\n")