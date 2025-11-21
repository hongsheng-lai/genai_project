import random
import math


def generate_multiplication():
    a = random.randint(100, 5000)
    b = random.randint(100, 5000)
    q = f"{a} x {b}"
    ans = a * b
    return q, ans

def generate_division():
    a = random.randint(1000, 50000)
    b = random.randint(10, 500)
    q = f"{a} / {b}"
    ans = a / b
    return q, ans

def generate_exponent():
    a = random.randint(2, 20)
    b = random.randint(2, 8)
    q = f"{a}^{b}"
    ans = a ** b
    return q, ans

def generate_log():
    base = random.randint(2, 10)
    exp = random.randint(2, 7)
    arg = base ** exp              
    q = f"log_{base}({arg})"
    ans = math.log(arg, base)
    return q, ans

def generate_sqrt():
    x = random.randint(100, 5000)
    q = f"sqrt({x})"
    ans = math.sqrt(x)
    return q, ans


def generate_random_question():
    funcs = [
        generate_multiplication,
        generate_division,
        generate_exponent,
        generate_log,
        generate_sqrt
    ]
    return random.choice(funcs)()

questions = []
answers = []

NUM = 300  

for i in range(1, NUM + 1):
    q, ans = generate_random_question()
    questions.append((i, q, ans))
output_path = "/Users/hairuow/Documents/cmu_course/10623_gen_ai/project/hidden_answer.txt"

with open(output_path, "w") as f:
    f.write("=== QUESTIONS ===\n")
    for qid, q, _ in questions:
        f.write(f"{q} = ? (Answer in [{qid}])\n")
    f.write("\n=== ANSWERS ===\n")
    for qid, _, ans in questions:
        f.write(f"[{qid}] Answer: {ans}\n")

output_path
