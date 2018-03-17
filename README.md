
# Installation 

## Packages
```
pip install -r requirements.txt
```

## Get the training data

Go to [https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data] and click Download.

Move the file into the root directory of this project and then run

```
tar xf fer2013.tar

```

# Training

```
wandb run python train_emotion_classifier.py
```

# Testing
```
python run_classifier.py test.jpg
```

# Things to try:

0. Look at the data!
  - How is it distributed?  How well would a random model do?
  - Can the model learn on a tiny subset of the data?
1. Better models
  - Better architecture
2. Data cleanup
  - Normalize
3. Data generation
  - https://keras.io/preprocessing/image/
4. Reduce learning rate on plateau
  - https://keras.io/callbacks/#reducelronplateau
5. Find more training data online?
6. Anything else?
