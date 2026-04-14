# The Bellman Equation

## The Core Idea

The value of being in a situation equals what you gain right now, plus the best value you can reach from there.

## The Equation

```
Q(s, a) = r + γ · max Q(s', a')
```

| Symbol | Meaning |
|--------|---------|
| `Q(s, a)` | Value of taking action `a` in state `s` |
| `r` | Immediate reward you receive |
| `γ` | Discount factor (e.g. 0.99) — future rewards are worth slightly less |
| `s'` | The next state after taking action `a` |
| `max Q(s', a')` | The best value achievable from the next state onward |

## Why It's True

Q-value is defined as the sum of all future rewards:

```
Q(s, a) = r₁ + γ·r₂ + γ²·r₃ + γ³·r₄ + ...
```

Factor out one step:

```
Q(s, a) = r₁ + γ · (r₂ + γ·r₃ + γ²·r₄ + ...)
                     └─────────────────────────┘
                       this is Q(s', best action)
```

So the equation is just a consequence of the definition — Q-values are recursive by nature.

## Rat and Cheese Example

The rat is 2 steps away from the cheese. Cheese gives reward +1, all other moves give 0. γ = 0.99.

```
[rat] [ ] [cheese]
  s2  s1    s0
```

Working backwards from the cheese:

```
Q(s0, right) = +1                        ← eat the cheese
Q(s1, right) =  0 + 0.99 × 1.0  = 0.99  ← one step away
Q(s2, right) =  0 + 0.99 × 0.99 = 0.98  ← two steps away
```

Even though the reward at s2 is 0, the Q-value is high because the network "knows" cheese is near.

## The Key Insight

- **Reward** is what the environment gives you immediately — it's fixed.
- **Q-value** is the total future reward from this point onward — it's what we want to learn.

The Bellman equation is a self-consistency condition: if you know the Q-value of every next state, you automatically know the Q-value of the current state. Training a neural network to satisfy this equation is exactly what DQN does.

The DQN is trying to approximate the Bellman equation.
