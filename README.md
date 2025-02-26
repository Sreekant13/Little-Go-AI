# Little-Go-AI
Projects based on course load(CSCI561)

# Overview
### The assignment focuses on developing AI agents for a small version of the Go game called Little-Go (5x5 board).
### The agents will compete in online tournaments on Vocareum.com.
### The task involves implementing Search, Game Playing, and Reinforcement Learning techniques.
### The goal is to create an AI agent capable of playing against various predefined AI agents.

# Summary of the Code
### The Stage 1 agent is built using Minimax with Alpha-Beta Pruning, optimized for fast decision-making with a branching factor of 5.
### The Stage 2 agent improves upon it by increasing the branching factor to 12, allowing for better strategy execution.
### The AI follows Go rules strictly (KO & Liberty).
### The program carefully selects valid moves, evaluates board positions, and chooses the best possible move using Minimax.
### The higher branching factor in Stage 2 makes the AI more competitive in tougher matches against Q-Learning and Championship agents.
### This ensures a well-balanced AI capable of playing strategically and efficiently within the given time constraints.
