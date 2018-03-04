# Machine Learning framework for Go

This project is intended to be a framework for testing different architectures
and features for how good different _Tensorflow graphs_ are at predicting the
next few moves and the who will win the game.

The core parts of the framework is written in [Cython](http://cython.org/), and
the machine learning parts are written in [Tensorflow](http://tensorflow.org) so
performance should be good.

This project is intended to be a sandbox, and is not intended for public use
beyond proof of concept.

## Dev Dependencies

* [Python 3.6](https://www.python.org/)
* [Cython](http://cython.org)
* [Tensorflow 1.6](https://www.tensorflow.org/)

## Usage

The framework will load all files matching the glob `data/*.sgf` and train a
neural network to predict the next few moves, as well as the winner of the game.
The script will run for 819,200 steps, exactly how long this will take depends
on the hardware of the computer but it should take about 8 hours on an NVIDIA
GTX 1080 Ti:

```bash
make run
```

The framework will periodically write the trained models and logs to `models/`.
You can monitor these logs using tensorboard:

```bash
tensorboard --logdir models/
```

## Experimental Results

See the wiki for test results as it is cumbersome to have to do a commit for
every update to the article.
