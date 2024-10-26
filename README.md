# Little Go AI — Minimax with Alpha-Beta Pruning

An AI agent that plays **5×5 Go (Little Go)** using depth-limited Minimax search with Alpha-Beta Pruning. Built in two stages with progressively stronger search, tested against random, greedy, minimax, Q-learning, and championship agents.

---

## Approach

### Stage 1 — Baseline Agent (Branching Factor: 5)
Minimax search with Alpha-Beta Pruning, depth-limited to 5 levels. Explores up to 5 candidate moves per position, staying well within the 10-second time budget.

### Stage 2 — Enhanced Agent (Branching Factor: 12)
Expanded move exploration with a **move heuristic** that prioritizes stone captures and board center control. The wider search (12 branches) finds significantly stronger lines while still completing within time limits.

---

## Key Engineering

| Feature | Detail |
|---|---|
| **Search** | Minimax with Alpha-Beta Pruning — prunes branches provably worse than known alternatives |
| **Depth** | 5-ply look-ahead |
| **Time management** | Hard 9.8s cutoff per move — avoids tournament timeout |
| **Move ordering** | Heuristic prioritizes captures + center squares for better pruning efficiency |
| **Ko rule** | Full Ko detection — compares board state against previous position |
| **Liberty checking** | Validates moves against self-capture; removes dead stones after each play |
| **Board evaluation** | Scores position by piece count delta + liberty advantage |

---

## Project Structure

```
stage1/
└── stage1.py          ← Baseline Minimax agent (branching factor 5)

stage2/
├── stage2.py          ← Enhanced agent (branching factor 12)
├── board.py           ← Board state representation
└── input_output.py    ← Input parsing and move output
```

---

## Tech

- **Language**: Python 3
- **Algorithm**: Minimax with Alpha-Beta Pruning
- **Game**: 5×5 Go (Little Go) — Ko rule, liberty rule, stone capture
