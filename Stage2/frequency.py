import re
from collections import Counter

with open("/home/emery/Downloads/wiki.txt", "r", encoding="utf-8") as file:
    text = file.read().lower()
    
tokens = re.findall(r'\b[a-zA-ZéÉèÈêÊäÄöÖüÜ]+\b', text)

word_counts = Counter(tokens)

with open("/Stage2/dictionary/frequency.txt", "w", encoding="utf-8") as output_file:
    for word, count in word_counts.most_common():
        output_file.write(f"{word} {count}\n")