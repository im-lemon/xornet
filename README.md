# xornet

xornet is my first ML model i trained from scratch,
it was trained on a CPU, and reached a loss of 0.0004 on epoch 1000.

The model consistently predicts the right XOR output.

XOR works like this: <br>
Takes two binary inputs, if both are
the same (0, 0) or (1, 1) return 0, else 1.

You can test the model with ```ìnference.py```, it looks like this:
```
x 1> 0
x 2> 1
1 # Correct
x 1> 1
x 2> 1
0 # Correct
x 1> 0
x 2> 0
0 # Correct
```

<footer>
<span>&copy; 2026 im-lemon | Lode Brusten | gatenets</span>
</footer>