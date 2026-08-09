- Epsilon-Greedy exploration strategy combined with Lévy flight to the SPBO algorithm for feature selection. 

1. Levy Flight ::
- Use "Mantegna's algorithm" for generating Lévy-distributed random steps

- Why Levy Flight ?
- Enables occasional long jumps in the search space, preventing premature convergence
- Uses heavy-tailed Lévy distribution (Mantegna implementation)
- Parameter beta (default 1.5) controls distribution shape

2. Epsilon-Greedy ::

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

