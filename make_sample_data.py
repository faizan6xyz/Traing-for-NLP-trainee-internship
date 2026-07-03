import random
import pandas as pd
random.seed(42)
POSITIVE_TEMPLATES = [
    "I absolutely loved this {noun}, it exceeded every expectation!",
    "This {noun} is fantastic, works perfectly and looks amazing.",
    "Best {noun} I've bought in years, highly recommend to everyone.",
    "Great {noun}! Fast delivery and excellent quality overall.",
    "Really happy with this {noun}, five stars all the way.",
    "Such an amazing {noun}, I can't stop telling my friends about it.",
    "The {noun} was wonderful, exactly what I was hoping for.",
    "Superb {noun}, worth every penny and more.",
    "I'm impressed by this {noun}, the quality is outstanding.",
    "This {noun} made my day, pure joy using it every time.",
]
NEGATIVE_TEMPLATES = [
    "I hated this {noun}, complete waste of money.",
    "Terrible {noun}, broke after just two days of use.",
    "Worst {noun} I have ever purchased, very disappointed.",
    "This {noun} is awful, nothing like what was advertised.",
    "Really bad experience with this {noun}, would not recommend.",
    "The {noun} stopped working almost immediately, so frustrating.",
    "Poor quality {noun}, feels cheap and flimsy.",
    "I'm furious about this {noun}, customer service was no help either.",
    "Such a disappointing {noun}, regret buying it.",
    "This {noun} is horrible, save your money and avoid it.",
]
NEUTRAL_TEMPLATES = [
    "The {noun} arrived on time, nothing special to report.",
    "This {noun} is okay, does what it says but nothing more.",
    "An average {noun}, neither good nor bad in my experience.",
    "The {noun} works as expected, no strong feelings either way.",
    "It's a standard {noun}, similar to others in this price range.",
    "The {noun} came in plain packaging, functions normally.",
    "Just a regular {noun}, I use it occasionally.",
    "The {noun} is fine for basic use, nothing more to add.",
    "Received the {noun} yesterday, still testing it out.",
    "This {noun} matches the description, no complaints or praise.",
]
NOUNS = [
    "product", "phone case", "laptop", "movie", "book", "headphones",
    "blender", "restaurant", "app", "service", "camera", "shoes",
    "coffee maker", "backpack", "TV show", "game", "hotel stay", "watch",
]

def make_row(templates, noun, label):
    text = random.choice(templates).format(noun=noun)
    return {"text": text, "label": label}
def build_dataset(n_per_class=250):
    rows = []
    for _ in range(n_per_class):
        rows.append(make_row(POSITIVE_TEMPLATES, random.choice(NOUNS), "positive"))
        rows.append(make_row(NEGATIVE_TEMPLATES, random.choice(NOUNS), "negative"))
        rows.append(make_row(NEUTRAL_TEMPLATES, random.choice(NOUNS), "neutral"))
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df
if __name__ == "__main__":
    df = build_dataset(n_per_class=250)  # 750 rows total
    df.to_csv("twitter_training.csv", index=False)
    print(f"Wrote sample_data.csv with {len(df)} rows")
    print(df['label'].value_counts())
