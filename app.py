import os
from flask import Flask, render_template, request, send_from_directory
from transformers import GPT2LMHeadModel, GPT2Tokenizer, MarianMTModel, MarianTokenizer
from gtts import gTTS
from huggingface_hub import login

app = Flask(__name__)
login(token="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")

# Translation models
translation_models = {
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "de": "Helsinki-NLP/opus-mt-en-de",
    "ja": "Helsinki-NLP/opus-mt-en-jap",
    "hi": "Helsinki-NLP/opus-mt-en-hi",
    "es": "Helsinki-N LP/opus-mt-en-es",
    "ru": "Helsinki-NLP/opus-mt-en-ru",
    "zh": "Helsinki-NLP/opus-mt-en-zh",
    "it": "Helsinki-NLP/opus-mt-en-it",
    "pt": "Helsinki-NLP/opus-mt-tc-big-en-pt",
    "ko": "Helsinki-NLP/opus-mt-tc-big-en-ko",
    "tr": "Helsinki-NLP/opus-mt-en-trk",
    "id": "Helsinki-NLP/opus-mt-en-id",
    "el": "Helsinki-NLP/opus-mt-en-el",
    "nl": "Helsinki-NLP/opus-mt-en-nl",
    "sv": "Helsinki-NLP/opus-mt-en-sv",
    "pl": "allegro/BiDi-eng-pol",
    "th": "facebook/m2m100_418M"
}

countries = {
    "France": "fr",
    "Germany": "de",
    "Japan": "ja",
    "India": "hi",
    "Spain": "es",
    "Russia": "ru",
    "China": "zh",
    "Italy": "it",
    "Brazil": "pt",
    "South Korea": "ko",
    "Turkey": "tr",
    "Indonesia": "id",
    "Greece": "el",
    "Netherlands": "nl",
    "Sweden": "sv",
    "Poland": "pl",
    "Thailand": "th"
}


def get_translation_model(target_lang_code):
    try:
        model_name = translation_models.get(target_lang_code)
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        print(f"Error loading translation model for {target_lang_code}: {e}")
        return None, None


def translate_text(text, target_lang_code):
    tokenizer, model = get_translation_model(target_lang_code)
    if tokenizer and model:
        try:
            tokens = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")
            translation = model.generate(**tokens)
            return tokenizer.decode(translation[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error during translation: {e}")
            return "Translation error."
    return "Model not available."


def generate_story(country, language):
    try:
        prompt = f"""
        You just arrived in {country} and step out at a famous landmark.
        As you're looking around, a friendly local greets you in {language} and begins to share two things:

        1. A unique greeting tradition in their culture — including a short dialogue in {language} (with English translation).
        2. A traditional practice — such as how they eat, what they wear, or a daily ritual — again with a short {language} dialogue and its English explanation.

        Keep the tone fun, friendly, and natural.
        """
        
        model_name = "gpt2-medium"
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs, max_length=500, pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2, num_return_sequences=1
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error generating story: {e}")
        return "Error generating story."


def narrate_story(text, lang_code="en"):
    try:
        tts = gTTS(text, lang=lang_code)
        audio_path = os.path.join("static", "story.mp3")
        tts.save(audio_path)
        return "story.mp3"
    except Exception as e:
        print(f"Error generating narration: {e}")
        return "Error generating audio."


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        selected_country = request.form["country"]
        lang_code = countries.get(selected_country)
        if not lang_code:
            return "Country not found", 400
        
        try:
            story = generate_story(selected_country, selected_country)
            translated = translate_text(story, lang_code)
            audio_file = narrate_story(translated, lang_code)
            return render_template("index.html", countries=countries, story=story, translated=translated, audio_file=audio_file, selected_country=selected_country)
        except Exception as e:
            print(f"Error processing request: {e}")
            return "Internal Server Error", 500

    return render_template("index.html", countries=countries)


@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("static", filename)


if __name__ == "__main__":
    app.run(debug=True)
