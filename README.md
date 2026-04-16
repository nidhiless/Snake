## AI Snake Pathfinding Simulator

An AI-based Snake game that dynamically selects pathfinding algorithms based on the current game environment instead of relying on a fixed strategy.

## Overview

This project demonstrates how different pathfinding algorithms behave under varying conditions such as obstacle density, available space, and distance to the target.  
The system evaluates the game state and selects the most suitable algorithm at runtime, while also visualizing its decisions.


## Features

- Dynamic selection of pathfinding algorithms based on game conditions  
- Implemented algorithms:
  - A* Search  
  - Breadth-First Search (BFS)  
  - Depth-First Search (DFS)  
  - Uniform Cost Search (UCS)  
  - Greedy Best-First Search  
  - Bidirectional Search  
  - Iterative Deepening DFS  
  - Hill Climbing  
  - Simulated Annealing  

- Real-time visualization of:
  - Snake movement  
  - Planned path  
  - Obstacles  

- Backend fallback:
  - Uses FastAPI for decision-making when available  
  - Falls back to local logic if backend is unavailable  


## Tech Stack

- Backend: FastAPI (Python)  
- Frontend: HTML, CSS, JavaScript  
- Visualization: Canvas API  

---

## How It Works

The system evaluates the current state using simple heuristics such as:
- Distance to the target  
- Obstacle layout  
- Available free space  
- Risk near the snake’s body  

Based on these factors, it selects an appropriate algorithm to compute the next move.

---

## Run Locally

### Clone the repository
```bash
git clone https://github.com/nidhiless/Snake.git
cd Snake
````

### Create virtual environment

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run backend

```bash
uvicorn main:app --reload
```

### Run frontend

Open `index.html` in your browser
(or use `python -m http.server`)


## Notes

This project focuses on algorithm selection and visualization in a dynamic environment rather than traditional gameplay optimization.



## Author

Nidhi Patel

