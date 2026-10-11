import torch
from collections import defaultdict
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
)
from safetensors.torch import load_file


# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "answerdotai/ModernBERT-base"
WEIGHTS_PATH = "model.safetensors"

ID_TO_LABEL = {
    0: "Algebra",
    1: "Intermediate Algebra",
    2: "Prealgebra",
    3: "Geometry",
    4: "Number Theory",
    5: "Counting & Probability",
    6: "Precalculus",
}


# ============================================================
# LOAD MODEL
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

config = AutoConfig.from_pretrained(
    BASE_MODEL,
    num_labels=7,
)

model = AutoModelForSequenceClassification.from_config(config)

weights = load_file(WEIGHTS_PATH)
model.load_state_dict(weights)

model.eval()


# ============================================================
# INFERENCE
# ============================================================

def classify(problem, top_k=3):
    inputs = tokenizer(
        problem,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    top_probs, top_ids = torch.topk(probabilities, k=top_k)

    results = []

    for probability, label_id in zip(top_probs, top_ids):
        results.append(
            (
                ID_TO_LABEL[label_id.item()],
                probability.item()
            )
        )

    return results


# ============================================================
# 100 TEST PROBLEMS
#
# Format:
# (
#     intended_category,
#     problem
# )
# ============================================================

tests = [

    # --------------------------------------------------------
    # PREALGEBRA
    # --------------------------------------------------------

    ("Prealgebra",
     "A number is increased by 20% and then decreased by 20%. "
     "The resulting number is 96. What was the original number?"),

    ("Prealgebra",
     "A shirt originally costs $80 and is discounted by 25%. "
     "What is its new price?"),

    ("Prealgebra",
     "The ratio of boys to girls in a class is 3:5. "
     "If there are 15 boys, how many girls are there?"),

    ("Prealgebra",
     "What is 3/4 of 120?"),

    ("Prealgebra",
     "A car travels 180 kilometres in 3 hours. "
     "What is its average speed?"),

    ("Prealgebra",
     "The average of five numbers is 12. "
     "What is their total sum?"),

    ("Prealgebra",
     "Solve for x: 3x + 7 = 22."),

    ("Prealgebra",
     "A recipe requires 3 cups of flour for 12 biscuits. "
     "How many cups are needed for 20 biscuits?"),

    ("Prealgebra",
     "What percentage of 250 is 50?"),

    ("Prealgebra",
     "A tank is 3/5 full. If its capacity is 200 litres, "
     "how many litres of water are in it?"),

    ("Prealgebra",
     "The temperature rises from 20 degrees to 25 degrees. "
     "What is the percentage increase?"),

    ("Prealgebra",
     "John has twice as many marbles as Sam. Together they have 36. "
     "How many marbles does John have?"),

    ("Prealgebra",
     "Simplify the fraction 84/126."),

    ("Prealgebra",
     "A train travels 240 km at 80 km/h. "
     "How long does the journey take?"),

    ("Prealgebra",
     "If 7 notebooks cost $21, how much do 12 notebooks cost?"),


    # --------------------------------------------------------
    # ALGEBRA
    # --------------------------------------------------------

    ("Algebra",
     "If x + 1/x = 3, find x^2 + 1/x^2."),

    ("Algebra",
     "Factor x^2 - 9."),

    ("Algebra",
     "Solve x^2 - 5x + 6 = 0."),

    ("Algebra",
     "If x + y = 10 and xy = 21, find x^2 + y^2."),

    ("Algebra",
     "Expand (x + 3)^2."),

    ("Algebra",
     "Solve the system x + y = 7 and x - y = 3."),

    ("Algebra",
     "If a + b = 8 and ab = 12, find a^2 + b^2."),

    ("Algebra",
     "Simplify (x^2 - 4)/(x - 2), assuming x is not 2."),

    ("Algebra",
     "Find all real x satisfying x^2 = 49."),

    ("Algebra",
     "If f(x) = 2x + 5, find f(7)."),

    ("Algebra",
     "Solve 2x^2 + 3x - 2 = 0."),

    ("Algebra",
     "If x - y = 4 and xy = 5, find x^2 + y^2."),

    ("Algebra",
     "Factor x^2 + 7x + 12."),

    ("Algebra",
     "Solve |x - 3| = 7."),

    ("Algebra",
     "If x^2 + y^2 = 25 and xy = 6, find (x + y)^2."),


    # --------------------------------------------------------
    # INTERMEDIATE ALGEBRA
    # --------------------------------------------------------

    ("Intermediate Algebra",
     "The roots of x^2 - 7x + k = 0 differ by 3. "
     "Find k."),

    ("Intermediate Algebra",
     "Find the sum of the roots of 3x^2 - 8x + 2 = 0."),

    ("Intermediate Algebra",
     "Solve the inequality x^2 - 5x + 6 < 0."),

    ("Intermediate Algebra",
     "Find the inverse of f(x) = (2x + 3)/(x - 1)."),

    ("Intermediate Algebra",
     "If the roots of x^2 + px + 12 = 0 are 3 and 4, find p."),

    ("Intermediate Algebra",
     "Solve 2^x = 32."),

    ("Intermediate Algebra",
     "Solve log base 2 of x = 5."),

    ("Intermediate Algebra",
     "Find the vertex of the parabola y = x^2 - 6x + 5."),

    ("Intermediate Algebra",
     "For what values of k does x^2 + kx + 9 = 0 "
     "have exactly one real solution?"),

    ("Intermediate Algebra",
     "Simplify (x^(1/2))^6 for positive x."),

    ("Intermediate Algebra",
     "Solve sqrt(x + 5) = x - 1."),

    ("Intermediate Algebra",
     "If f(x) = x^2 and g(x) = 2x + 1, find f(g(3))."),

    ("Intermediate Algebra",
     "Find the remainder when x^3 + 2x^2 - x + 5 "
     "is divided by x - 2."),

    ("Intermediate Algebra",
     "Solve x^4 - 5x^2 + 4 = 0."),


    # --------------------------------------------------------
    # GEOMETRY
    # --------------------------------------------------------

    ("Geometry",
     "A square has diagonal length 10. Find its area."),

    ("Geometry",
     "A circle has radius 7. Find its area."),

    ("Geometry",
     "A right triangle has legs 6 and 8. Find its hypotenuse."),

    ("Geometry",
     "A rectangle has length 12 and width 5. "
     "Find the length of its diagonal."),

    ("Geometry",
     "The angles of a triangle are in the ratio 2:3:4. "
     "Find the largest angle."),

    ("Geometry",
     "A circle has circumference 10π. Find its radius."),

    ("Geometry",
     "A regular hexagon has side length 4. Find its perimeter."),

    ("Geometry",
     "A right triangle has legs 6 and 8. "
     "A circle is inscribed inside it. Find the radius of the circle."),

    ("Geometry",
     "Two similar triangles have corresponding sides 3 and 9. "
     "If the area of the smaller triangle is 10, "
     "find the area of the larger triangle."),

    ("Geometry",
     "A cube has side length 5. Find its surface area."),

    ("Geometry",
     "A cylinder has radius 3 and height 10. Find its volume."),

    ("Geometry",
     "An equilateral triangle has side length 8. Find its altitude."),

    ("Geometry",
     "The interior angle of a regular polygon is 150 degrees. "
     "How many sides does the polygon have?"),

    ("Geometry",
     "Two parallel lines are cut by a transversal. "
     "One acute angle is 37 degrees. Find the corresponding acute angle."),

    ("Geometry",
     "A chord passes through the centre of a circle and has length 18. "
     "Find the radius."),


    # --------------------------------------------------------
    # NUMBER THEORY
    # --------------------------------------------------------

    ("Number Theory",
     "Find the remainder when 2^100 is divided by 7."),

    ("Number Theory",
     "How many positive divisors does 360 have?"),

    ("Number Theory",
     "Find the greatest common divisor of 252 and 198."),

    ("Number Theory",
     "Find the least common multiple of 24 and 36."),

    ("Number Theory",
     "What is the smallest positive integer that leaves remainder 2 "
     "when divided by 5 and remainder 3 when divided by 7?"),

    ("Number Theory",
     "What is the last digit of 7^2026?"),

    ("Number Theory",
     "How many integers from 1 to 100 are divisible by 6?"),

    ("Number Theory",
     "Find the prime factorisation of 756."),

    ("Number Theory",
     "Find the remainder when 123456 is divided by 9."),

    ("Number Theory",
     "What is the smallest prime number greater than 100?"),

    ("Number Theory",
     "How many trailing zeros are in 100 factorial?"),

    ("Number Theory",
     "Find the greatest integer n such that 2^n divides 100!."),

    ("Number Theory",
     "Find all positive integers less than 20 that are relatively prime to 20."),

    ("Number Theory",
     "Find the remainder when 3^50 + 2^50 is divided by 5."),


    # --------------------------------------------------------
    # COUNTING & PROBABILITY
    # --------------------------------------------------------

    ("Counting & Probability",
     "How many ways can 3 students be selected from a group of 10?"),

    ("Counting & Probability",
     "Five students sit in a row. How many arrangements are possible "
     "if Alice and Bob must sit next to each other?"),

    ("Counting & Probability",
     "Two fair six-sided dice are rolled. "
     "What is the probability that their sum is greater than 9?"),

    ("Counting & Probability",
     "How many different arrangements of the letters in MATH are possible?"),

    ("Counting & Probability",
     "A fair coin is tossed 5 times. "
     "What is the probability of obtaining exactly 3 heads?"),

    ("Counting & Probability",
     "How many 4-digit numbers can be formed using the digits "
     "1, 2, 3, 4, 5 without repetition?"),

    ("Counting & Probability",
     "A bag contains 3 red and 5 blue balls. "
     "One ball is selected randomly. What is the probability it is red?"),

    ("Counting & Probability",
     "How many diagonals does a decagon have?"),

    ("Counting & Probability",
     "In how many ways can 8 people be divided into two groups of 4?"),

    ("Counting & Probability",
     "Three fair coins are tossed. "
     "What is the probability of getting at least one head?"),

    ("Counting & Probability",
     "How many subsets does a set containing 8 elements have?"),

    ("Counting & Probability",
     "A committee of 4 is selected from 6 men and 5 women. "
     "How many committees contain exactly 2 men?"),

    ("Counting & Probability",
     "How many paths from (0,0) to (4,3) use only steps right or up?"),

    ("Counting & Probability",
     "A standard deck contains 52 cards. "
     "What is the probability of drawing an ace?"),


    # --------------------------------------------------------
    # PRECALCULUS
    # --------------------------------------------------------

    ("Precalculus",
     "If sin(theta) = 3/5 and theta is acute, find cos(2 theta)."),

    ("Precalculus",
     "Find sin(30 degrees)."),

    ("Precalculus",
     "Find cos(60 degrees)."),

    ("Precalculus",
     "Solve sin(x) = 1/2 for 0 <= x < 360 degrees."),

    ("Precalculus",
     "If tan(theta) = 3/4 and theta is acute, find sin(theta)."),

    ("Precalculus",
     "Simplify sin^2(x) + cos^2(x)."),

    ("Precalculus",
     "Find the amplitude of y = 3 sin(2x)."),

    ("Precalculus",
     "Find the period of y = sin(2x)."),

    ("Precalculus",
     "Convert 150 degrees to radians."),

    ("Precalculus",
     "If cos(theta) = -1/2 and theta lies in quadrant II, "
     "find sin(theta)."),

    ("Precalculus",
     "Find the exact value of sin(45 degrees) cos(45 degrees)."),

    ("Precalculus",
     "Solve 2 cos(x) = 1 for 0 <= x < 360 degrees."),

    ("Precalculus",
     "If sec(theta) = 5/4 and theta is acute, find tan(theta)."),
]


# ============================================================
# RUN TESTS
# ============================================================

correct = 0

category_total = defaultdict(int)
category_correct = defaultdict(int)

for number, (expected, problem) in enumerate(tests, start=1):

    predictions = classify(problem, top_k=3)

    predicted = predictions[0][0]

    category_total[expected] += 1

    is_correct = predicted == expected

    if is_correct:
        correct += 1
        category_correct[expected] += 1

    result = "PASS" if is_correct else "FAIL"

    print(f"\nTEST {number}")
    print(f"Problem:  {problem}")
    print(f"Expected: {expected}")
    print(f"Result:   {result}")

    print("\nTop predictions:")

    for rank, (label, confidence) in enumerate(predictions, start=1):
        print(
            f"  {rank}. {label:<25} "
            f"{confidence:>8.4%}"
        )

    print("-" * 70)


# ============================================================
# SUMMARY
# ============================================================

total = len(tests)
accuracy = correct / total

print("\n")
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(f"\nTests:    {total}")
print(f"Correct:  {correct}")
print(f"Wrong:    {total - correct}")
print(f"Accuracy: {accuracy:.2%}")

print("\nPER-CATEGORY ACCURACY")
print("-" * 70)

for label in ID_TO_LABEL.values():

    total_for_category = category_total[label]
    correct_for_category = category_correct[label]

    if total_for_category == 0:
        continue

    category_accuracy = correct_for_category / total_for_category

    print(
        f"{label:<25} "
        f"{correct_for_category:>2}/{total_for_category:<2} "
        f"{category_accuracy:>8.2%}"
    )