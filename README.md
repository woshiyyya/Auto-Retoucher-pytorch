# Auto-Retoucher(ART)--A Framework for Background Replacement and Foreground adjustment
Given someone's photo, generates a new image with best matching background, and find the best spatial location and scale.

A PyTorch implementation of ART framework.

### Example foreground images:

![resource/test_1.png](resource/test_1.png)
![resource/test_2.png](resource/test_2.png)

### demo:
![resource/result1.png](resource/demo2.png)
![resource/result1.png](resource/result1.png)

### moving sequence:
Adjustment procedure guided by model's gradient. 

Fg moves from a random initial location to a plausible position

![resource/result1_seq.png](resource/result1_seq.png)


## Framework:
![](resource/.jpg)


## Training:

```
python train.py
```

## Inference:

```
python Inference.py
```
