# This repo tries to reproduce the model of [DeepMatch](https://arxiv.org/abs/1808.06576)
## Dependencies
python >= 3.6

tensorflow >= 1.11

## Usage
first run:
~~~
make test
~~~

### training procedure
do train test set split(nodup)
~~~
python train_test_split.py
~~~

then convert data to tfrecord

~~~
make prep
~~~

then start training

~~~
make prep
~~~

### testing reranking performance

~~~
make infer
~~~