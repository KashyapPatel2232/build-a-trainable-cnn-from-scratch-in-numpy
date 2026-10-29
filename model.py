"""
Build a Trainable CNN from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - argmax_rows
def argmax_rows(matrix):
    # TODO: return the index of the largest element in each row of a 2D array
    return np.argmax(matrix, axis = 1)

# Step 2 - row_max
import numpy as np

def row_max(matrix):
    # TODO: return the maximum value of each row of `matrix` with keepdims True for broadcasting.
    return np.max(matrix, axis = 1, keepdims = True)

# Step 3 - row_sum
import numpy as np

def row_sum(matrix):
    """Return per-row sums of a 2D array with shape (N, 1)."""
    # TODO: return the sum along axis 1 keeping the reduced dimension
    return np.sum(matrix, axis = 1, keepdims = True)

# Step 4 - exp_shifted
import numpy as np

def exp_shifted(logits):
    """Subtract per-row max from logits and exponentiate elementwise."""
    # TODO: shift each row of logits by its max and return elementwise exp
    return np.exp(logits - np.max(logits, axis = 1, keepdims = True))

# Step 5 - stable_softmax
def stable_softmax(logits):
    # TODO: Compute a numerically stable softmax row-wise over (N, C) logits.
    maxi = row_max(logits)
    sub_logits = logits - maxi
    num = exp_shifted(sub_logits)
    deno = row_sum(num)
    softmax = num/deno 
    return softmax

# Step 6 - one_hot
def one_hot(labels, num_classes):
    # TODO: convert integer labels into a (N, num_classes) one-hot float matrix
    hot_encoded = np.zeros(shape = (len(labels), num_classes))
    for i in range(len(labels)):
        hot_encoded[i, labels[i]] = 1.0

    return hot_encoded

# Step 7 - gather_true_class_probs
def gather_true_class_probs(probs, labels):
    # TODO: return probs[i, labels[i]] for every row i as a 1D length-N array.
    ls = np.ndarray(shape = len(probs))
    for i in range(len(probs)):
        ls[i] = probs[i, labels[i]]
    return ls

# Step 8 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(probs, labels, eps=1e-12):
    # TODO: return the mean negative log-likelihood of the true-class probabilities
    logits = gather_true_class_probs(probs, labels)
    loss = -1* np.mean(np.log(logits + eps))
    return loss

# Step 9 - accuracy
def accuracy(logits_or_probs, labels):
    # TODO: return the fraction of rows whose argmax matches the integer label.
    predict_labels = argmax_rows(logits_or_probs)
    accuracy = np.mean(predict_labels == labels)
    return accuracy

# Step 10 - he_std
def he_std(fan_in):
    # TODO: return the He initialization standard deviation sqrt(2 / fan_in).
    return np.sqrt(2/fan_in)

# Step 11 - he_init
def he_init(shape, fan_in, seed):
    # TODO: sample a weight tensor from a normal distribution scaled by He std using the seed.
    np.random.seed(seed)
    weight = np.random.randn(*shape) * he_std(fan_in)
    return weight

# Step 12 - init_zero_bias
import numpy as np

def init_zero_bias(length):
    # TODO: return a 1D float array of zeros with the given length.
    return np.zeros(length)

# Step 13 - pad_2d
def pad_2d(images, pad):
    # TODO: zero-pad the spatial (H, W) dims of a 4D (N, C, H, W) tensor by `pad` on each side.
    return np.pad(images, pad_width=((0,0),(0,0),(pad, pad),(pad,pad)), mode = 'constant')

# Step 14 - output_spatial_size
def output_spatial_size(input_size, kernel, stride, padding):
    # TODO: return the conv/pool output spatial dimension from input_size, kernel, stride, padding
    return (input_size + 2* padding - kernel) // stride + 1

# Step 15 - im2col
import numpy as np

def im2col(images, kernel_h, kernel_w, stride, padding):
    N, C, H, W = images.shape
    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)

    output = np.zeros((N * out_h * out_w, C * kernel_h * kernel_w), dtype=images.dtype)
    images_padded = pad_2d(images, padding)

    k = 0
    for batch in range(N):
        for y in range(out_h):
            for x in range(out_w):
                start_y = y * stride
                start_x = x * stride
                end_y = start_y + kernel_h
                end_x = start_x + kernel_w
                
                patch = images_padded[batch, :, start_y:end_y, start_x:end_x]
                output[k, :] = patch.flatten()
                k += 1

    return output

# Step 16 - col2im
def col2im(cols, input_shape, kernel_h, kernel_w, stride, padding):
    # TODO: re-roll a (N*out_h*out_w, C*kh*kw) column matrix back into a (N, C, H, W) tensor
    N, C, H, W = input_shape
    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)

    padded_h = H + 2 * padding
    padded_w = W + 2 * padding
    output = np.zeros((N, C, padded_h, padded_w), dtype=cols.dtype)

    for n in range(N):
        for y in range(out_h):
            for x in range(out_w):
                idx = n * out_h * out_w + y*out_w + x
                patch = cols[idx].reshape(C, kernel_h, kernel_w)
                
                start_x = x * stride
                start_y = y * stride
                end_x = start_x + kernel_w
                end_y = start_y + kernel_h
                output[n,:,start_y:end_y, start_x:end_x] += patch
                    
    if padding != 0:
        output = output[:, :, padding:padding + H, padding:padding + W]

    return output

# Step 17 - conv2d_forward
def conv2d_forward(x, weights, bias, stride, padding):
    # TODO: convolve x with weights using im2col, add bias, return output and a backprop cache.
    N, C_in, H, W = x.shape
    C_out, _, kernel_h, kernel_w = weights.shape
    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)
    x_flat = im2col(x,kernel_h, kernel_w, stride, padding)
    w_flat = weights.reshape(C_out, -1)
    output = x_flat @ w_flat.T + bias
    output_reshape = output.reshape(N, out_h, out_w, C_out).transpose(0,3,1,2)
    cache = {'cols': x_flat, 'kernel_h':kernel_h,'kernel_w':kernel_w,'padding':padding,'stride':stride, 'weights':weights,'x_shape':x.shape}
    return output_reshape, cache

# Step 18 - conv2d_grad_input
def conv2d_grad_input(d_out, cache):
    # TODO: backprop d_out through the conv input using col2im
    _, C_out, _, _ = d_out.shape
    weights = cache['weights']
    _, _, kernel_h, kernel_w = weights.shape
    stride = cache['stride']
    padding = cache['padding']

    dout_cols = d_out.transpose(0,2,3,1).reshape(-1, C_out)
    patch_grads = dout_cols @ weights.reshape(C_out, -1)
    output = col2im(patch_grads, cache["x_shape"], kernel_h, kernel_w, stride, padding)
    return output

# Step 19 - conv2d_grad_weights
def conv2d_grad_weights(d_out, cache):
    # TODO: return dL/dW shaped (C_out, C_in, kH, kW) from d_out and the im2col cache.
    _, C_out, _, _ = d_out.shape
    weights = cache['weights']
    _, C_in, kernel_h, kernel_w = weights.shape
    input_patch = cache['cols']
    dout_cols = d_out.transpose(0,2,3,1).reshape(-1, C_out)
    patch_grads = dout_cols.T @ input_patch
    output = patch_grads.reshape(C_out,C_in, kernel_h, kernel_w)
    return output

# Step 20 - conv2d_grad_bias
def conv2d_grad_bias(d_out):
    # TODO: return a length C_out gradient by reducing d_out over batch and spatial axes
    _, cout, _, _ = d_out.shape
    return np.sum(d_out, axis = (0,2,3))

# Step 21 - conv2d_backward
def conv2d_backward(d_out, cache):
    # TODO: return (dx, dW, db) using the conv2d gradient helpers and the forward cache
    dx = conv2d_grad_input(d_out, cache)
    dW = conv2d_grad_weights(d_out, cache)
    db = conv2d_grad_bias(d_out)
    return dx, dW, db

# Step 22 - maxpool2d_forward
def maxpool2d_forward(x, kernel, stride):
    # TODO: run 2D max pooling and cache the in-window argmax of each output cell.
    N, C, H, W = x.shape
    out_h = (H - kernel ) // stride + 1
    out_w = (W - kernel ) // stride + 1
    
    out = np.zeros((N, C, out_h, out_w), dtype=x.dtype)
    argmax = np.zeros((N, C, out_h, out_w), dtype= int)

    for n in range(N):
        for c in range(C):
            for y_in in range(out_h):
                for x_in in range(out_w):
                    start_x = x_in* stride
                    start_y = y_in*stride
                    end_x = start_x + kernel
                    end_y = start_y + kernel

                    out[n, c, y_in, x_in] = np.max(x[n,c, start_y: end_y, start_x: end_x])
                    argmax[n, c, y_in, x_in] = np.argmax(x[n,c, start_y: end_y, start_x: end_x].flatten())

    cache = {'x_shape': x.shape, 'argmax':argmax, 'kernel':kernel, 'stride':stride}
    return out, cache

# Step 23 - scatter_grad_window
import numpy as np

def scatter_grad_window(grad_value, argmax_index, kernel):
    # TODO: place grad_value at the argmax position within a (kernel, kernel) zero array.
    output = np.zeros((kernel, kernel))
    row = argmax_index // kernel
    col = argmax_index % kernel
    output[row][col] = grad_value
    return output

# Step 24 - maxpool2d_backward
def maxpool2d_backward(d_out, cache):
    # TODO: scatter each d_out value to the cached argmax position in its window
    N, C, H, W = cache['x_shape']
    stride = cache['stride']
    argmax = cache['argmax']
    kernel = cache['kernel']
    out_h = output_spatial_size(H, kernel, stride, 0)
    out_w = output_spatial_size(W, kernel, stride, 0)
    
    output = np.zeros((N, C, H, W), dtype= d_out.dtype)

    for n in range(N):
            for c in range(C):
                for y_in in range(out_h):
                    for x_in in range(out_w):
                        start_x = x_in* stride
                        start_y = y_in*stride
                        end_x = start_x + kernel
                        end_y = start_y + kernel

                        window_grad = scatter_grad_window(d_out[n, c, y_in, x_in], argmax[n, c, y_in, x_in], kernel,)
                        output[n, c, start_y:end_y, start_x:end_x] += window_grad
    return output

# Step 25 - relu_forward
def relu_forward(x):
    # TODO: Compute the elementwise ReLU and cache the input for backprop.
    output = np.maximum(0., x)
    cache = {'x': x}
    return output, cache

# Step 26 - relu_backward
def relu_backward(d_out, cache):
    # TODO: mask the upstream gradient by the positive entries of the cached input.
    x = cache['x']
    output = d_out * (x > 0)
    return output

# Step 27 - flatten_forward
def flatten_forward(x):
    # TODO: reshape a 4D feature map into a 2D batch matrix and cache the original shape
    N, _, _, _ = x.shape
    output = x.reshape(N, -1)
    cache = {'x_shape':x.shape}
    return output, cache

# Step 28 - flatten_backward
import numpy as np

def flatten_backward(d_out, cache):
    # TODO: reshape the upstream gradient back to the original 4D feature map shape.
    N, C, H, W = cache['x_shape']
    output = d_out.reshape(N, C, H, W)
    return output

# Step 29 - linear_forward
def linear_forward(x, weights, bias):
    # TODO: compute X @ W + b and cache the inputs needed for backprop.
    output = x @ weights + bias
    cache = {'weights':weights, 'x': x}
    return output, cache

# Step 30 - linear_grad_input
import numpy as np

def linear_grad_input(d_out, cache):
    """Gradient of a linear layer w.r.t. its input X."""
    # TODO: return dL/dX given d_out (N, D_out) and cache['weights'] (D_in, D_out)
    weights = cache['weights']
    return d_out @ weights.T

# Step 31 - linear_grad_weights
import numpy as np

def linear_grad_weights(x, dout):
    """Gradient of loss wrt linear-layer weights W of shape (D_in, D_out)."""
    # TODO: Compute the gradient of a linear layer's loss wrt its weight matrix W.
    return x.T @ dout

# Step 32 - linear_grad_bias
import numpy as np

def linear_grad_bias(dout):
    # TODO: Compute the bias gradient of a linear layer given upstream gradient dout.
    return np.sum(dout, axis = 0)

# Step 33 - linear_backward
def linear_backward(dout, cache):
    # combine input, weight, and bias gradients for a linear layer using the cache
    inp = cache['x']
    dx = linear_grad_input(dout, cache)
    dW = linear_grad_weights(inp, dout)
    db = linear_grad_bias(dout)
    return dx, dW, db

# Step 34 - softmax_cross_entropy_forward
def softmax_cross_entropy_forward(logits, y):
    # TODO: return the mean cross-entropy loss for logits (N, C) and integer labels y (N,).
    probs = stable_softmax(logits)
    out = cross_entropy_loss(probs, y)
    return float(max(0.0, out))

# Step 35 - softmax_cross_entropy_backward
def softmax_cross_entropy_backward(logits, y):
    # TODO: return the fused softmax-cross-entropy gradient of shape (N, C).
    N, C = logits.shape
    return (stable_softmax(logits) - one_hot(y, C)) / N

# Step 36 - sgd_step
import numpy as np

def sgd_step(param, grad, lr):
    # TODO: return the SGD-updated parameter array (param - lr * grad).
    return param - lr * grad

# Step 37 - adam_update_m
import numpy as np

def adam_update_m(m, grad, beta_one):
    # TODO: return the updated first moment estimate using beta_one and grad.
    return m * beta_one + (1- beta_one) * grad

# Step 38 - adam_update_v
import numpy as np

def adam_update_v(v, grad, beta_two):
    # TODO: return the updated Adam second moment estimate as an EMA of squared gradients.
    return beta_two * v + (1-beta_two) * (grad ** 2)

# Step 39 - adam_bias_correct
def adam_bias_correct(moment, beta, t):
    # TODO: return moment divided by (1 - beta**t) to undo Adam's zero-init bias.
    return moment / (1- beta ** t)

# Step 40 - adam_param_step
import numpy as np

def adam_param_step(param, m_hat, v_hat, lr, eps):
    # TODO: apply one Adam parameter update using bias-corrected moments
    param = param - lr* (m_hat / (np.sqrt(v_hat) +eps))
    return param

# Step 41 - adam_step
import numpy as np

def adam_step(param, grad, m, v, t, lr, beta_one, beta_two, eps):
    # TODO: chain the four Adam helpers and return (new_param, new_m, new_v)
    new_m = adam_update_m(m, grad, beta_one)
    new_v = adam_update_v(v, grad, beta_two)
    m_hat = adam_bias_correct(new_m, beta_one, t)
    v_hat = adam_bias_correct(new_v, beta_two, t)
    new_param = adam_param_step(param, m_hat, v_hat, lr, eps)
    return new_param, new_m, new_v

# Step 42 - init_conv_layer
def init_conv_layer(out_channels, in_channels, kernel_size, seed=0):
    shape = (out_channels, in_channels, kernel_size, kernel_size)
    W = he_init(shape, in_channels * kernel_size * kernel_size, seed)
    b = init_zero_bias(out_channels)
    return {'W': W, 'b': b}

# Step 43 - init_linear_layer
def init_linear_layer(in_features, out_features, seed=0):
    # TODO: return {'W': He-init matrix (in_features, out_features), 'b': zero bias (out_features,)}
    shape = (out_features, in_features)
    W = he_init(shape, in_features, seed)
    b = init_zero_bias(out_features)
    return {'W': W.T, 'b': b}

# Step 44 - init_lenet
def init_lenet(in_channels, num_classes, seed=0):
    # TODO: build conv1, conv2, fc1, fc2 with the right shapes and return them in a dict.
    conv1 = init_conv_layer(6, in_channels, 5, seed)
    conv2 = init_conv_layer(16, 6, 5, seed)
    fc1 = init_linear_layer(256, 120, seed)
    fc2 = init_linear_layer(120, num_classes, seed)
    return {'conv1':conv1, 'conv2':conv2, 'fc1':fc1, 'fc2':fc2}

# Step 45 - forward_conv_block
def forward_conv_block(x, W, b, pool_size, stride, pad):
    # TODO: run conv2d -> relu -> maxpool2d and return (out, cache_dict)
    out, conv_cache = conv2d_forward(x, W, b, stride, pad)
    out, pool_cache = maxpool2d_forward(out, pool_size, pool_size)
    out, relu_cache = relu_forward(out)

    cache_out = {'conv_cache': conv_cache, 'pool_cache':pool_cache,'relu_cache':relu_cache}
    
    return out, cache_out

# Step 46 - forward_classifier_block
def forward_classifier_block(x, fc1, fc2):
    # TODO: run flatten -> linear -> relu -> linear and return logits plus a cache dict.
    output_1, cache_1 = flatten_forward(x)
    output_2, cache_2 = linear_forward(output_1, fc1['W'], fc1['b'])
    output_3, cache_3 = relu_forward(output_2)
    output_final, cache_final = linear_forward(output_3, fc2['W'], fc2['b'])
    cache ={'flatten_cache': cache_1, 'fc1_cache':cache_2, 'relu_cache':cache_3,'fc2_cache':cache_final}
    return output_final, cache

# Step 47 - lenet_forward
def lenet_forward(x, params):
    # TODO: run two conv blocks then the classifier block and return (logits, caches).
    output_block1, cache_block1 = forward_conv_block(x, params['conv1']['W'],params['conv1']['b'], 2, 1, 0)
    output_block2, cache_block2 = forward_conv_block(output_block1, params['conv2']['W'],params['conv2']['b'], 2, 1, 0)
    output_classifier, cache_classifier = forward_classifier_block(output_block2,params['fc1'],params['fc2'])
    cache = {'block1':cache_block1, 'block2':cache_block2, 'classifier':cache_classifier}
    return output_classifier, cache

# Step 48 - backward_conv_block
def backward_conv_block(dout, cache):
    # TODO: backprop dout through the cached pool, relu, and conv layers in reverse order.
    out_relu = relu_backward(dout, cache['relu_cache'])
    out_maxpool = maxpool2d_backward(out_relu, cache['pool_cache'])
    dx, dW, db = conv2d_backward(out_maxpool, cache['conv_cache'])
    return dx, dW, db

# Step 49 - backward_classifier_block
def backward_classifier_block(dlogits, cache):
    # TODO: backprop through fc2 -> relu -> fc1 -> flatten using the cached values
    dx2, dW2, db2 = linear_backward(dlogits, cache['fc2_cache'])
    fc2 = {'dW':dW2, 'db':db2}
    dx_relu = relu_backward(dx2, cache['relu_cache'])
    dx1, dW1, db1 = linear_backward(dx_relu, cache['fc1_cache'])
    fc1 = {'dW':dW1, 'db':db1}
    output_final = flatten_backward(dx1, cache['flatten_cache'])
    return {'dx':output_final,'fc1':fc1,'fc2':fc2}

# Step 50 - lenet_backward
def lenet_backward(dlogits, caches):
    # TODO: walk classifier and conv block caches in reverse to assemble all gradients
    cache_classifier = backward_classifier_block(dlogits, caches['classifier'])
    dx2, dW2, db2 = backward_conv_block(cache_classifier['dx'], caches['block2'])
    dx1, dW1, db1 = backward_conv_block(dx2, caches['block1'])
    conv1 = {'dW':dW1, 'db':db1}
    conv2 = {'dW':dW2, 'db':db2}
    fc1 = cache_classifier['fc1']
    fc2 = cache_classifier['fc2']
    grads = {'conv1':conv1, 'conv2':conv2, 'fc1':fc1, 'fc2':fc2}
    return grads

# Step 51 - lenet_predict
def lenet_predict(x, params):
    # TODO: Return the argmax class index per sample from a LeNet forward pass.
    output, cache = lenet_forward(x, params)
    predicted_class = argmax_rows(output)
    return predicted_class

# Step 52 - build_synthetic_image_dataset
def build_synthetic_image_dataset(num_samples, num_classes, image_size, in_channels=1, seed=0):
    # TODO: Return (x, y) for a reproducible synthetic NCHW image dataset.
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, in_channels, image_size, image_size))
    y = rng.integers(0, num_classes, num_samples)
    for i in range(num_samples):
        k = y[i]
        shift = k - (num_classes - 1) / 2
        x[i] += shift
    return x,y

# Step 53 - shuffle_indices (not yet solved)
# TODO: implement

# Step 54 - train_test_split (not yet solved)
# TODO: implement

# Step 55 - iterate_minibatches (not yet solved)
# TODO: implement

# Step 56 - train_step (not yet solved)
# TODO: implement

# Step 57 - train_one_epoch (not yet solved)
# TODO: implement

# Step 58 - train_loop (not yet solved)
# TODO: implement

# Step 59 - evaluate (not yet solved)
# TODO: implement

