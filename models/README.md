# Models

## Use Google's model

Download [Google's pretrained network](https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip) first, and then unpack the `tensorflow_inception_graph.pb` file from the archive.

You can use the following code in the command line.

``` bash
wget https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip && unzip inception5h.zip
```

The network we use here is the [GoogLeNet](https://arxiv.org/abs/1409.4842) architecture, trained to classify images into one of 1000 categories of the [ImageNet](http://image-net.org/) dataset. It consists of a set of layers that apply a sequence of transformations to the input image. The parameters of these transformations were determined during the training process by a variant of gradient descent algorithm.

## Use your own model

Of course you can use your own model. Put your neural network model here, and set its filename to [`model_fn`](../dream.py#L12) variable in `dream.py`.

Note that you probably also need to update those layers' names and related functions in the code instead of using the ones.
