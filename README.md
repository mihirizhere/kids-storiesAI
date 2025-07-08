# Bedtime Story Generator

A simple Python application that uses OpenAI’s API to generate and judge bedtime stories for children aged 5–10. It classifies a user prompt into one of four themes, selects an appropriate story arc, generates a story, checks quality, and applies user feedback.

## Features

- **TF-IDF + Naive Bayes classifier** for prompt categorization
- **Four story themes**:
  - Fantasy Adventure
  - Moral Quest
  - Number Journey
  - Lullaby Rhyme
- **Story arcs** mapped to each theme
- **Generation** via `gpt-3.5-turbo`
- **Automated quality check** (age-appropriateness, coherence, vocabulary)
- **Feedback loop** for revisions

## Prerequisites

- Python 3.8 or higher  
- An OpenAI API key  
- A recent Conda or virtualenv environment

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bedtime-story-generator.git
   cd bedtime-story-generator
# Setup

## 1. Create and activate an environment:

```bash
conda create -n stories python=3.9
conda activate stories
# or
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Install dependencies:

```bash
pip install -r requirements.txt
```

## 3. Create a `.env` file in the project root:

```ini
OPENAI_API_KEY=sk-...
```

# Training the Classifier

train the model once:

```bash
python train_classifier.py
```

This uses 80 example prompts (20 per theme) to build a TF-IDF + MultinomialNB classifier and saves `classifier.joblib`.

# Running the App

```bash
python main.py
```

1. Enter your story request when prompted.
2. The app categorizes your request, selects a narrative arc, and generates a story.
3. If the story does not pass the built-in judge, it will regenerate.
4. Provide optional feedback to revise the story.

# Customization

* **Themes**: Edit the `THEMES` dict in `main.py` to add or rename themes.
* **Arcs**: Update the `select_arc` function to change story structures.
* **Token limits**: Adjust `max_tokens` in `generate_story` and `apply_feedback` for longer or shorter outputs.
* **Classifier**: Add more examples in `train_classifier.py` and re-run training to improve accuracy.


