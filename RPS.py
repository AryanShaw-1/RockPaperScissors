import random
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# ---------- Mappings ----------
moves = ["rock", "paper", "scissors"]
move_to_int = {"rock": 0, "paper": 1, "scissors": 2}
int_to_move = {0: "rock", 1: "paper", 2: "scissors"}

# History for machine learning
history = []
labels = []

# AI Model
model = KNeighborsClassifier(n_neighbors=3)

# Scores
user_score = 0
computer_score = 0

print("\nWelcome to AI-Based Rock, Paper, Scissors!")
print("The AI learns your patterns and predicts your next move.")
print("Type 'exit' anytime to stop.\n")

while True:
    user_move = input("Enter rock / paper / scissors: ").lower()

    if user_move == "exit":
        print("\nGame Over!")
        print(f"Final Score → You: {user_score} | Computer: {computer_score}")
        break

    if user_move not in moves:
        print("Invalid choice. Try again.\n")
        continue

    user_int = move_to_int[user_move]

    # ----------- AI PREDICTION -----------
    if len(history) < 5:
        # Not enough past data → random
        comp_int = random.randint(0, 2)
        comp_move = int_to_move[comp_int]
    else:
        # Train ML model on your past moves
        model.fit(history, labels)

        # Predict your next move using your last move
        predicted_user_move = model.predict([history[-1]])[0]

        # AI chooses move to defeat your predicted move
        comp_int = (predicted_user_move + 1) % 3
        comp_move = int_to_move[comp_int]

    print(f"Computer chose: {comp_move}")

    # Add current user move to training data
    history.append([user_int])
    labels.append(user_int)

    # ----------- WINNER LOGIC -----------
    if user_int == comp_int:
        print("Result: It's a Tie!")
    elif (user_int - comp_int) % 3 == 1:
        print("Result: You Win!")
        user_score += 1
    else:
        print("Result: Computer Wins!")
        computer_score += 1

    print(f"Score → You: {user_score} | Computer: {computer_score}\n")
