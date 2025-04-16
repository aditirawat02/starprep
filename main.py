import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import speech_recognition as sr
from huggingface_hub import login
import pyaudio
pa = pyaudio.PyAudio()


login("<ENTER_HUGGINGFACE_TOKEN_HERE>")

# Load a local LLM (e.g., Llama 2 or Gemma)
# MODEL_NAME = "google/gemma-3-1b-pt"
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # Replace with your local model path or Hugging Face model name
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16)

# Define job domain and experience level
JOB_DOMAIN = "Software Engineering"
EXPERIENCE_LEVEL = "Mid-Level"

# STAR format explanation
STAR_EXPLANATION = """
The STAR format is a structured way to answer behavioral interview questions:
- Situation: Describe the context or background.
- Task: Explain your responsibility or goal.
- Action: Detail the steps you took.
- Result: Share the outcome and impact of your actions.
"""

# List of sample behavioral questions
BEHAVIORAL_QUESTIONS = [
    "Tell me about a time when you faced a challenging problem at work. How did you handle it?",
    "Describe a situation where you had to work with a difficult team member. What was the outcome?",
    "Give an example of a time when you had to meet a tight deadline. How did you manage your time?",
]


# Function to generate a question using the local LLM
def generate_question():
    prompt = f"Generate a behavioral interview question for a {EXPERIENCE_LEVEL} {JOB_DOMAIN} role."
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_length=50, num_return_sequences=1)
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question


def evaluate_answer(question, answer):
    prompt = f"""
    Evaluate the following answer to the behavioral interview question based on relevance, STAR format, and clarity.
    Provide constructive feedback and suggest improvements.

    Question: {question}
    Answer: {answer}

    Evaluation Criteria:
    1. Relevance: Does the answer directly address the question?
    2. STAR Format: Does the answer follow the Situation, Task, Action, Result structure?
    3. Clarity: Is the answer clear and concise?
    4. Impact: Does the answer demonstrate the candidate's skills and achievements?

    Feedback:
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_length=512, num_return_sequences=1, temperature=0.7)
    feedback = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return feedback


# Function to capture audio input and convert it to text
def get_audio_input():

    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"Device {i}: {info['name']}")

    # Manually set the correct index (replace 0 with your mic index)
    mic_index = 3

    recognizer = sr.Recognizer()
    with sr.Microphone(device_index=mic_index) as source:
        print("Listening...")
        audio = recognizer.listen(source)
        print("Done listening")
        print(audio.sample_width)
        filename = "microphone-results.wav"
        with open(filename, "wb") as f:
            f.write(audio.get_wav_data())
        print("saved")
        # text = recognizer.recognize(audio, language="english")  # Use Google Web Speech API

        # open the file
        with sr.AudioFile(filename) as source:
            # listen for the data (load audio to memory)
            audio_data = recognizer.record(source)
            # recognize (convert from speech to text)
            text = recognizer.recognize_google(audio_data)
            print(text)
        print(f"You said: {text}")

        # write audio to a RAW file

        # try:
        #     text = recognizer.recognize_google_cloud(audio)  # Use Google Web Speech API
        #     print(f"You said: {text}")
            # return text
        # except sr.UnknownValueError:
        #     print("Sorry, I could not understand the audio.")
        #     return None
        # except sr.RequestError:
        #     print("Sorry, there was an issue with the speech recognition service.")
        #     return None


# Main interview loop
def conduct_interview():
    print("Welcome to the Behavioral Interview Simulator!")
    print(f"Job Domain: {JOB_DOMAIN}")
    print(f"Experience Level: {EXPERIENCE_LEVEL}")
    print("\nLet's begin the interview. Please answer the following questions.\n")

    for i, question in enumerate(BEHAVIORAL_QUESTIONS):
        print(f"Question {i + 1}: {question}")
        print("Please speak your answer after the beep.")
        input("Press Enter to start recording...")
        user_answer = get_audio_input()

        if user_answer:
            print("\nEvaluating your answer...\n")
            feedback = evaluate_answer(question, user_answer)
            print(f"Feedback:\n{feedback}\n")
            print(STAR_EXPLANATION)
            print("-" * 50)
        else:
            print("Skipping to the next question due to audio input issues.\n")

    print("Interview completed. Thank you for participating!")


# Run the interview
if __name__ == "__main__":
    conduct_interview()

