- Epsilon-Greedy exploration strategy combined with Lévy flight to the SPBO algorithm for feature selection:

This enhancement integrates two exploration mechanisms into the SPBO algorithm:
1. Epsilon-Greedy exploration strategy
2. Lévy Flight-based long-range exploration

The primary objective is to improve the balance between exploration and exploitation during feature selection.

In the original SPBO algorithm, students update their positions using predefined learning behaviors. Although this can effectively exploit promising regions of the search space, the population may lose diversity and become trapped around a local optimum.

The proposed enhancement introduces controlled randomness and occasional long-distance movements to help SPBO explore unexplored regions while preserving its original exploitation behavior.

- Main idea:
Original SPBO
     │
     ├── Average Student Update
     │       └── Move toward group mean
     │
     └── Random Student Update
             └── Random movement


- Enhanced SPBO:
     │
     ├── Epsilon-Greedy Decision
     │       │
     │       ├── Exploitation
     │       │      └── Original SPBO behavior
     │       │
     │       └── Exploration
     │              │
     │              ├── Lévy Flight
     │              └── Random Uniform Spread
     │
     └── Continue normal SPBO learning behavior

The enhancement is designed to preserve the original SPBO mechanism while adding exploration only when the epsilon-greedy policy selects it.

--------------------------------------------------------------------------------------------

1. Levy Flight ::
- A Lévy Flight is a random walk in which most movements are relatively small, but occasionally the algorithm performs a large jump.

- Use "Mantegna's algorithm" for generating Lévy-distributed random steps

- Why Levy Flight ?
- SPBO can converge toward promising solutions relatively quickly.
- Enables occasional long jumps in the search space, preventing premature convergence
- Uses heavy-tailed Lévy distribution (Mantegna implementation)
- Parameter beta (default 1.5) controls distribution shape

--------------------------------------------------------------------------------------------

2. Epsilon-Greedy ::
- Epsilon-Greedy is a simple exploration/exploitation strategy.
- The purpose of epsilon-greedy is to control when the enhanced exploration mechanisms are activated.


- 2.1 Epsilon-Greedy Average Student Update ::
- Replaces the original update_average_student() with update_average_student_epsilon()
- Flow ::
                For each student in average group:
                For each feature dimension:
                    
                    if random() < epsilon (exploration probability):
                    # EXPLORATION MODE
                    if random() >= Pthreshold:
                        → Apply Lévy flight step (occasional long jumps)
                    else:
                        → Random spread within [min, max] of population
                    
                    else:
                    # EXPLOITATION MODE (original behavior)
                    → Move towards the group mean

- eps (default 0.1): Probability of choosing exploration vs exploitation
- levy_beta (default 1.5): Shape of Lévy distribution
- levy_A (default 1.0): Amplitude/scaling of Lévy step
- Pthreshold (default 0.6): Probability threshold for using Lévy vs random spread in exploration

--------------------------------------------------------------------------------------------

- 2.2 Epsilon-Greedy Random Student Update ::
- Replaces update_random_student() with update_random_student_epsilon()
- Flow:
                For each random student:
                For each feature dimension:
                    
                    if random() < epsilon:
                    # EXPLORATION
                    if random() >= Pthreshold:
                        → Apply Lévy flight (long jump)
                    else:
                        → Random uniform spread in search space
                    
                    else:
                    # EXPLOITATION  
                    → Random uniform spread in search space (keeps existing behavior)
--------------------------------------------------------------------------------------------
