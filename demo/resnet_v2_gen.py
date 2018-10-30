from random import shuffle
import os
import sys
import mxnet as mx
from mxnet import autograd, ndarray as nd
import numpy as np
import matplotlib.pyplot as plt
from time import time

def patch_path(path):
    return os.path.join(os.path.dirname(__file__), path)


def classifier_loss(X, classifier): 
    # Get label for Jazz
    from mxnet_audio.library.utility.gtzan_loader import gtzan_labels
    jazz_id = 5
    assert(gtzan_labels[jazz_id] == 'jazz')
    
    # Perform backpropagation
    X.attach_grad()
    with autograd.record():
       output = classifier.model(X)
       output = output[0, 3] #jazz_id]
    output.backward()

    return (output.asscalar(), X.grad)


def perceptual_loss(X):
    # Calculate ||X^TX||_* = tr(X^TX) = sum_ij (X.^2)_ij
    rank_est = nd.sum(nd.sum(nd.multiply(X, X), axis=2), axis=2)
    rank_est = rank_est[0, 0]

    # Calculate gradient
    grad = nd.multiply(2.0, X)
    return (rank_est.asscalar(), grad)


def style_loss(X):
    X = X.asnumpy()

    # Shift by one row / column
    XN = np.roll(X, -1, axis=2)
    XE = np.roll(X,  1, axis=3)
    XS = np.roll(X,  1, axis=2)
    XW = np.roll(X, -1, axis=3)
    diffN = np.subtract(X, XN)
    diffE = np.subtract(X, XE)
    diffS = np.subtract(X, XS)
    diffW = np.subtract(X, XW)

    # Compute anisotropic TV
    diffVert = np.sum(np.sum(np.abs(diffE), axis=2), axis=2)
    diffHorz = np.sum(np.sum(np.abs(diffS), axis=2), axis=2)
    TV = diffVert[0, 0] + diffHorz[0, 0]

    # Compute gradient
    gradVert = np.add(np.sign(diffN), np.sign(diffS))
    gradHorz = np.add(np.sign(diffE), np.sign(diffW))
    grad = np.multiply(-1.0, np.add(gradVert, gradHorz))
    return (-np.asscalar(TV), nd.array(grad))


def l1_regularization(X):
    return (-nd.norm(X).asscalar(), nd.multiply(-1.0, nd.sign(X)))


def load_audio_path_label_pairs(max_allowed_pairs=None):
    from mxnet_audio.library.utility.gtzan_loader import download_gtzan_genres_if_not_found
    download_gtzan_genres_if_not_found(patch_path('very_large_data/gtzan'))
    audio_paths = []
    with open(patch_path('data/lists/test_songs_gtzan_list.txt'), 'rt') as file:
        for line in file:
            audio_path = patch_path('very_large_data/' + line.strip())
            audio_paths.append(audio_path)
    pairs = []
    with open(patch_path('data/lists/test_gt_gtzan_list.txt'), 'rt') as file:
        for line in file:
            label = int(line)
            if max_allowed_pairs is None or len(pairs) < max_allowed_pairs:
                pairs.append((audio_paths[len(pairs)], label))
            else:
                break
    return pairs


def GD(classifier, alpha, beta, gamma, max_iterations = 100,
       learning_rate = 50.0, learning_rate_decay = 0.9, momentum = 0.5):
    # Random initialization
    #X = nd.abs(nd.random_normal(scale=1, shape=(1, *classifier.input_shape)))
    audio_path_label_pairs = load_audio_path_label_pairs()
    shuffle(audio_path_label_pairs)
    audio_path, actual_label_id = audio_path_label_pairs[0]
    mg = classifier.compute_melgram(audio_path)
    X = nd.array(np.expand_dims(mg, axis=0), ctx=classifier.model_ctx)
    X = X.as_in_context(classifier.model_ctx)

    # GD with momentum
    eta = -1.0 * learning_rate
    prev_grad = nd.zeros(shape=X.shape)
    losses = []
    cls_losses = []
    sty_losses = []
    pct_losses = []
    l1s = []
    for t in range(max_iterations):
        # Projection
        X = nd.maximum(X, 0.0) 
        X = nd.minimum(X, 1.0) 

        # Save as .csv        
        img = X[0, 0, :, :].asnumpy()
        np.savetxt('./temp/iter%d.csv' % t, img)

        # Calculate losses and gradients
        cls_loss = classifier_loss(X, classifier)
        sty_loss = style_loss(X)
        pct_loss = perceptual_loss(X)
        l1 = l1_regularization(X)
        
        # Weighting
        loss = cls_loss[0] + alpha * sty_loss[0] + beta * pct_loss[0] + gamma * l1[0]
        grad = cls_loss[1] + alpha * sty_loss[1] + beta * pct_loss[1] + gamma * l1[1]

        # Store losses
        print("Iteration %d: %.2f | (%.2f, %.2f, %.2f, %.2f)" % (t, loss, cls_loss[0], sty_loss[0], pct_loss[0],l1[0]))
        #print("Iteration %d: %.2f | (%.2f, %.2f, %.2f)" % (t, loss, cls_loss[0], sty_loss[0], pct_loss[0]))
        losses.append(loss)
        cls_losses.append(cls_loss[0])
        sty_losses.append(sty_loss[0])
        pct_losses.append(pct_loss[0])
        l1s.append(l1[0])

        # Update
        X = X - eta * (nd.array(grad) + momentum * prev_grad)

        eta = eta * learning_rate_decay
        prev_grad = grad
    
    #img = X[0, 0, :, :].asnumpy()
    #plt.figure()
    #plt.imshow(img)
    #plt.savefig('./cepstrogram.png')
    #__import__('code').interact(local=locals())

def main():
    sys.path.append(patch_path('..'))

    # Load pre-trained classifier
    from mxnet_audio.library.resnet_v2 import ResNetV2AudioClassifier
    classifier = ResNetV2AudioClassifier()
    classifier.load_model(model_dir_path=patch_path('models'))

    start = time()
    # Perform projected gradient descent with momentum
    #GD(classifier, 0.004, 0.0008, 0.005)
    GD(classifier, 0.008, 0.0008, 0.005)
    print("Time spent: %.2fs" % (time() - start))


if __name__ == '__main__':
    main()
