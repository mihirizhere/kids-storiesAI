from story_classifier import train_and_save_classifier

# 20 example prompts for each theme:
texts = [
    # — fantasy_adventure —
    "Take me on a dragon’s quest in a sky castle",
    "I want a tale of brave knights and hidden treasure",
    "Tell me about a magical forest and talking trees",
    "Adventure in a pirate ship across the seven seas",
    "A unicorn journey through rainbow valleys",
    "Sneak into a wizard’s tower and discover spells",
    "A lost city under the ocean with mermaid friends",
    "Journey across a desert on a camel caravan",
    "A space explorer discovers an alien planet",
    "A secret map leads to a mountain of gold",
    "A young hero’s quest to rescue a princess",
    "Battle trolls under a bridge to save a village",
    "Race against time in an enchanted clock tower",
    "A fairy guide leads you through moonlit caves",
    "Find a magic sword in an abandoned castle",
    "Ride a flying carpet over ancient ruins",
    "Explore hidden temples in a jungle",
    "A cloud kingdom where birds are your allies",
    "Discover a portal to a land of giants",
    "A knight and a dragon become unlikely friends",

    # — moral_quest —
    "Tell me a story about why honesty matters",
    "A tale teaching the value of sharing toys",
    "Why being kind to others is important",
    "A story about patience and waiting your turn",
    "How hard work leads to great rewards",
    "A lesson on saying sorry when you’re wrong",
    "Why we should help friends in need",
    "The importance of listening to your parents",
    "A fable about courage in the face of fear",
    "Why we shouldn’t judge others by looks",
    "A story showing respect for elders",
    "Teaching forgiveness after a fight",
    "Why it’s good to say ‘thank you’",
    "A tale of perseverance in learning to ride a bike",
    "Why telling the truth keeps friendships strong",
    "The power of teamwork to solve problems",
    "How responsibility makes you trustworthy",
    "Why it’s okay to ask for help",
    "A lesson on caring for pets responsibly",
    "Why showing gratitude makes people happy",

    # — number_journey —
    "Count the farm animals one by one",
    "Let’s count stars in the night sky",
    "A story that teaches numbers up to ten",
    "Counting apples as they drop from a tree",
    "One frog, two frogs, three jumping frogs",
    "Count the blocks while building a tower",
    "A train with five colorful carriages",
    "Counting shells on the sandy beach",
    "Seven balloons floating into the air",
    "Count the raindrops on your window",
    "Three little kittens and their mittens",
    "Count the cars in a busy parking lot",
    "One, two, buckle my shoe rhyme story",
    "Counting candies in a jar",
    "A parade with four marching bands",
    "Count the petals on a flower",
    "Five happy ducks swimming in a pond",
    "Count the leaves on a tree branch",
    "A birthday cake with six candles",
    "Count the fish in the aquarium",

    # — lullaby_rhyme —
    "Sing me a gentle rhyme about moonlight",
    "A bedtime poem to calm my mind",
    "Soft bedtime lullaby with stars and sleep",
    "Rhyme about a sleepy teddy bear",
    "A poem about drifting on a cloud",
    "Rhyme to help little ones fall asleep",
    "Lullaby about the sun saying goodnight",
    "Soft rhyme of kittens curling up",
    "Bedtime poem with gentle ocean waves",
    "Rhyme about a cradle rocking slowly",
    "A soothing rhyme about dreams",
    "Lullaby with fireflies lighting the night",
    "Gentle poem of snowflakes softly falling",
    "Rhyme about a sleepy little owl",
    "Bedtime verse of a quiet forest",
    "Rhyme about a puppy snuggling down",
    "Lullaby of a starlit meadow",
    "Soft poem of raindrops tapping window",
    "Rhyme about a sleepy dragon’s yawn",
    "Gentle lullaby of a lull in the breeze",
]


labels = (
    ['fantasy_adventure'] * 20 +
    ['moral_quest'] * 20 +
    ['number_journey'] * 20 +
    ['lullaby_rhyme'] * 20
)

train_and_save_classifier(texts, labels)
