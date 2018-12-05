Classifix
=========

Experiementation with classifier


# Experience 1a

* embed the class label inside the training data `img[112][112] = y` and see if the classifier is able to pick it up during training
   * Accuracy should be ~100%
* We test the behaviour by embedding the wrong label in the test image
   * Accuracy should ne   ~0% if successful

This would should overfitting at work. A single pixel is responsible for the classification.
This should highlight the issue with modern classification deep neural nets and ultimatly what make them
hackable through forged samples.

# Experience 1b

* Change the cost function to include a regularization member
    * `cost = cross_entropy(y, y_hat) + lambda * (x - x_hat)^2`
* The regularization forces the data retained by the network to be sufficient to rebuild a close enough version of the original idea
    * forces the network to extract meaningful features from the image.
    * A single pixel will not be enough

Hopefully this cost function should generate a more robust network
 
# Experience 2

* Change the labels from one-hot encoding to a simple drawing of the label
    * `y=0...1000` => `y=Tensor[1, 128, 128]`
* Forces the network to find the similarities between the training data and the drawing
    * Forces meaningful feature extraction
    * Single pixel will be discarded entirely

Hopefully this cost function should be robust to attack.



