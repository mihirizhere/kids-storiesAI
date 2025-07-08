"""
original.py

This script implements a bedtime-story agent using OpenAI’s API. It:
 - Loads your API key from a .env file
 - Classifies the user’s prompt into a story category
 - Selects an appropriate story arc
 - Generates a story via OpenAI
 - Judges the story for age-appropriateness and coherence
 - Allows the user to provide feedback and revises the story accordingly

To run successfully, you must install required packages locally:
    pip install openai python-dotenv

"""

from dotenv import load_dotenv
import os
import sys

try:
    import openai
except ModuleNotFoundError:
    sys.exit("Error: openai package not found. Please install it with `pip install openai` in your local environment.")
from story_classifier import classify_topic_ml

def load_api_key():
    """Load the OpenAI API key from .env and configure the client."""
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found in environment. Add it to your .env or export it.")
    openai.api_key = key

THEMES = {
    'fantasy_adventure': 'Fantasy Adventure',
    'moral_quest':       'Moral Quest',
    'number_journey':    'Number Journey',
    'lullaby_rhyme':     'Lullaby Rhyme'
}


# --- Category Classifier ---
def classify_topic(user_prompt: str) -> str:
    """
    ML-based classifier returning one of:
    'fantasy_adventure', 'moral_quest',
    'number_journey', or 'lullaby_rhyme'
    """
    # text = user_prompt.lower()
    # if any(word in text for word in ['count', 'numbers', 'one', 'two']):
    #     return 'counting'
    # if 'rhyme' in text or 'poem' in text:
    #     return 'bedtime_rhyme'
    # if 'lesson' in text or 'morals' in text:
    #     return 'moral_story'
    # return 'fairy_tale'
    return classify_topic_ml(user_prompt)


# --- Story Arc Selector ---
def select_arc(category: str) -> str:
    arcs = {
        'fantasy_adventure': 'ThreeAct',
        'moral_quest':       'ProblemResolution',
        'number_journey':    'Cumulative',
        'lullaby_rhyme':     'RhymeScheme',
    }
    return arcs.get(category, 'ThreeAct')
    # category = classify_topic(topic)
    # arc = select_arc(category)


# --- Core LLM calls ---
def generate_story(topic: str, arc: str, category: str) -> str:
    prompt = (
        f"You are a storyteller for children aged 5-10. "
        f"Please write a story of **about 225 words** and no more than 250 words total."
        f"Use simple vocabulary, gentle themes, and a '{arc}' story structure. "
        f"Category: {THEMES.get(category, category)}. Topic: {topic}."
    )
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt},
                  {"role": "user", "content": topic}],
        max_tokens=450,
        temperature=0.7,
    )
    return resp.choices[0].message["content"].strip()


# --- LLM Judge ---
def judge_story(topic, story_text: str) -> bool:
    judge_prompt = (
        "Please evaluate this story for: "
        "(1) Age-appropriateness (5-10), "
        "(2) Coherence & engagement, "
        "(3) Simple vocabulary. "
        "(4) Adherence to original prompt: " + topic +
        "\nRespond with 'PASS' or 'FAIL'.\n\nStory:\n" + story_text
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
        max_tokens=450,
        temperature=0.7,
    )
    return resp.choices[0].message["content"].strip()




def main():
    # Optionally run tests if env var set
    # if os.getenv("RUN_TESTS") == "1":
    #     _run_tests()
    #     return

    load_api_key()
    topic = input("What kind of story do you want to hear? ")
    category = classify_topic(topic)
    arc = select_arc(category)

    # Generate and judge
    story = generate_story(topic, arc, category)
    if not judge_story(topic, story):
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
