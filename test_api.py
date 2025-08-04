from flask import Flask, request, jsonify
from Stage2.symspell_gemini import correct_spelling, rephrase_sentence

app = Flask(__name__)

@app.route('/process-text', methods=['POST'])
def process_text():
    data = request.get_json(force=True)
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    spelling_corrected = correct_spelling(text)
    rephrased = rephrase_sentence(spelling_corrected)

    return jsonify({
        'original': text,
        'rephrased': rephrased
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
