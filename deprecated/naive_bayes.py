"""
original.py

This script implements a bedtime-story agent using OpenAI’s API and a Naive Bayes classifier for topic routing.
It:
 - Loads your API key from a .env file
 - Trains a simple Naive Bayes classifier on example prompts
 - Classifies the user’s prompt into a story category
 - Selects an appropriate story arc
 - Generates a story via OpenAI
 - Judges the story for age-appropriateness and coherence
 - Allows the user to provide feedback and revises the story accordingly

To run successfully, you must install required packages locally:
    pip install openai python-dotenv scikit-learn

"""

from dotenv import load_dotenv
import os
import sys

# Attempt to import OpenAI and sklearn dependencies
try:
    import openai
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
except ModuleNotFoundError as e:
    sys.exit(f"Error: {e.name} not found. Please install with `pip install openai python-dotenv scikit-learn`.")

# --- Load API Key ---
def load_api_key():
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found in environment. Add it to your .env or export it.")
    openai.api_key = key

# --- Naive Bayes Classifier Setup ---
CATEGORIES = ['fairy_tale', 'moral_story', 'counting', 'bedtime_rhyme']
TRAINING_DATA = [
    ("Tell me a fairy tale about a princess", 'fairy_tale'),
    ("I want a story with a moral lesson", 'moral_story'),
    ("Count the stars before sleep", 'counting'),
    ("Write a bedtime rhyme for kids", 'bedtime_rhyme'),
    ("A dragon and knight adventure", 'fairy_tale'),
    ("Teach me about honesty and kindness", 'moral_story'),
    ("One, two, three, count with me", 'counting'),
    ("Make a poem that rhymes", 'bedtime_rhyme'),
]

vectorizer = CountVectorizer()
classifier = MultinomialNB()

def train_classifier():
    texts, labels = zip(*TRAINING_DATA)
    X = vectorizer.fit_transform(texts)
    classifier.fit(X, labels)

# --- Category Classification ---
def classify_topic(user_prompt: str) -> str:
    """
    Use a trained Naive Bayes classifier to map prompt to one of the CATEGORIES.
    """
    X = vectorizer.transform([user_prompt])
    pred = classifier.predict(X)[0]
    return pred

# --- Story Arc Selector ---
def select_arc(category: str) -> str:
    arcs = {
        'fairy_tale': "ThreeAct",
        'moral_story': "ProblemResolution",
        'counting': "Cumulative",
        'bedtime_rhyme': "RhymeScheme"
    }
    return arcs.get(category, 'ThreeAct')

# --- Core LLM calls ---
def generate_story(topic: str, arc: str, category: str) -> str:
    prompt = (
        f"You are a storyteller for children aged 5-10. "
        f"Use simple vocabulary, gentle themes, and a '{arc}' story structure. "
        f"Category: {category}. Topic: {topic}."
    )
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt},
                  {"role": "user", "content": topic}],
        max_tokens=400,
        temperature=0.7,
    )
    return resp.choices[0].message["content"].strip()

# --- LLM Judge ---
def judge_story(story_text: str) -> bool:
    judge_prompt = (
        "Please evaluate this story for: "
        "(1) Age-appropriateness (5-10), "
        "(2) Coherence & engagement, "
        "(3) Simple vocabulary. "
        "Respond with 'PASS' or 'FAIL'.\n\nStory:\n" + story_text
    )
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"system","content":"You are a helpful judge."},
                  {"role":"user","content":judge_prompt}],
        max_tokens=50,
        temperature=0.1,
    )
    verdict = resp.choices[0].message["content"].strip().upper()
    return verdict.startswith("PASS")

# --- Feedback Handler ---
def apply_feedback(story: str, feedback: str) -> str:
    prompt = (
        "Original story:\n" + story + "\n\n" 
        "User feedback: '" + feedback + "'. Please revise accordingly."
    )
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"system","content":"You are a story editor."},
                  {"role":"user","content":prompt}],
        max_tokens=400,
        temperature=0.7,
    )
    return resp.choices[0].message["content"].strip()

# --- Tests for classifier and selector ---
def _run_tests():
    train_classifier()
    assert classify_topic("Tell a fairy tale with unicorns") == 'fairy_tale'
    assert classify_topic("Please write a rhyme poem") == 'bedtime_rhyme'
    assert classify_topic("Count to ten with animals") == 'counting'
    assert classify_topic("A story teaching honesty") == 'moral_story'
    assert select_arc('bedtime_rhyme') == 'RhymeScheme'
    print("All tests passed.")

# --- Main Flow ---
def main():
    # Option to run tests
    if os.getenv("RUN_TESTS") == "1":
        _run_tests()
        return

    load_api_key()
    train_classifier()
    topic = input("What kind of story do you want to hear? ")
    category = classify_topic(topic)
    arc = select_arc(category)

    # Generate and judge
    story = generate_story(topic, arc, category)
    if not judge_story(story):
        print("Initial story did not pass the judge—regenerating...")
        story = generate_story(topic, arc, category)

    print("\n---- Here is your story ----\n")
    print(story)
    print("-----------------------------\n")

    feedback = input("Any feedback or changes? (Enter to finish) ")
    if feedback:
        revised = apply_feedback(story, feedback)
        print("\n---- Revised Story ----\n")
        print(revised)
        print("------------------------\n")

if __name__ == "__main__":
    main()
