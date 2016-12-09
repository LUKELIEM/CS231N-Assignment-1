import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4, verbose=False):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)
    if verbose:
        print self.params

  def loss(self, X, y=None, reg=0.0, verbose=False):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape
    num_train = N
    num_classes = D

    # Initialize intermediate values and local gradients for the stages
    # of the computational graph implementation of SVM
    
    dW1 = np.zeros(W1.shape)
    db1 = np.zeros(b1.shape)
    dW2 = np.zeros(W2.shape)
    db2 = np.zeros(b2.shape)
    
    scores = np.zeros((num_train,num_classes))  # intermediate value scores
    dscores = np.zeros((num_train,num_classes)) # and its gradient

    norm_scores = np.zeros(scores.shape)  # intermediate value margins
    dnorm_scores = np.zeros(scores.shape)  # and its gradient
    dnorm_scores_1 = np.zeros(scores.shape)  # and its gradient

    correct_scores = np.zeros(num_train)  # intermediate value margins
    dcorrect_scores = np.zeros(num_train)  # and its gradient
    
    exp_scores = np.zeros(scores.shape)  # intermediate value margins
    dexp_scores = np.zeros(scores.shape)  # and its gradient
    lg_exp_scores  = np.zeros(scores.shape) # its local gradient

    sums = np.zeros(num_train)  # intermediate value margins
    dsums = np.zeros(num_train)  # and its gradient

    log_sums = np.zeros(num_train)  # intermediate value margins
    dlog_sums = np.zeros(num_train) # its gradient
    lg_log_sums = np.zeros(num_train) # its local gradient
    
    losses = np.zeros(num_train)  # loss and
    dlosses = np.zeros(num_train) # its gradient
    data_loss = 0
    reg_loss = 0
    
    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################

    """
    We dealing with a 2-layer NN followed by a softmax classifier with the following 
    architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax
    
    The loss and gradient computation will be performed in 2 passes:
       - In the forward pass, the loss calculation is decomposed into 10 stages
         At the end of the forward pass, loss is calculated
       - In the backward pass, we apply chain rule to calculate the gradients
         from Stage 10 back to Stage 1
         At the end of the backward pass, the gradients dW1 and dW2 are calculated
       
    The input to the 2-layer NN is X, and the output is scores.
    The scores is then fed into the softmax classifier, which is defined as:
    
        L(i) = -Score_Correct_Class + log(summation(exp(Scores_of_all_Classes)))

    """
    # 2-layer Neural Network Section:
    #  Input - X
    #  Output - scores
    
    # Stage 1 - X.dot(W1)+b1 --> o1                               #[1]    
    o1= X.dot(W1) + b1
    
    # Stage 2 - ReLU(o1) --> h1                                   #[2]
    h1 = np.maximum(o1, np.zeros(o1.shape))
    lg_h1 = np.zeros(o1.shape)
    lg_h1[np.nonzero(h1)] = 1          # local gradient of ReLU is 1
                                       # for any element which is >0

    # Stage 3 - h1.dot(W2)+b2 --> scores                          #[3]    
    scores = h1.dot(W2) + b2

    if verbose:
        print "\nNeural Network Layer 1:"
        print X.shape
        print W1.shape
        print o1
        print h1
        print lg_h1
        print "\nNeural Network Layer 2:"
        print h1.shape
        print W2.shape
        print scores
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################

    # Softmax Classifier Section:
    #  Input - scores
    #  Output - data_loss
    
    # Stage 4 - scores - scale_const --> norm_scores                     #[4]
    # The scaling constant is an arbitrary constant. In this case, we pick the
    # maximum value in scores, so that elements of norm_scores are all 0 or  
    # less. This is to prevent overflow error in subsequent calculation.
    # The local gradient in this gate is thus d(x-c)/dx = 1
    scale_const = np.max(scores)
    norm_scores = scores - scale_const

    # Stage 5 - score of correct class --> correct_scores                #[5]    
    correct_scores = norm_scores[np.arange(num_train), y]

    # Stage 6 - exp(norm_scores) --> exp_scores                          #[6]    
    exp_scores = np.exp(norm_scores)
    lg_exp_scores = np.exp(norm_scores)  # local gradient of exp(x) = exp(x)

    # Stage 7 - sum(exp_scores) --> sums                                 #[7]
    sums = np.sum(exp_scores, axis=1)
    
    # Stage 8 - log(sums) --> log_sums                                   #[8]
    log_sums = np.log(sums)
    lg_log_sums = 1.0/sums               # local gradient of log(x) = 1/x
    
    # Stage 9 - log_sums - correct_scores --> losses                     #[9]
    losses = log_sums - correct_scores

    # Stage 10 - sum(losses)/num_train --> loss                         #[10]
    data_loss = np.sum(losses)/num_train

    # Reg Forward Pass 
    reg_loss = 0.5 * reg * (np.sum(W1*W1)+np.sum(W2*W2))
    
    loss = data_loss + reg_loss
    
    if verbose:
        print "Training data - Forward Pass" 
        print "\n Scores:"
        print scores
        print "\n Scaling Constant:"
        print scale_const
        print "\n Normalized Scores:"
        print norm_scores
        print "\n True Labels (y):"
        print y
        print "\n Scores of correct classes:"        
        print correct_scores
        print "\n Exp(Normalized Scores):" 
        print exp_scores
        print "\n Sum(Exp(Normalized Scores)):" 
        print sums
        print "\n Log(Sum):"         
        print log_sums
        print "\n Losses:" 
        print losses
        print "\n Data Loss:"
        print data_loss
        print "\n Reg Loss:"
        print reg_loss
        print "\n Loss:"
        print loss
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    
    # Softmax Segment 
    # Backward Pass - 8 Stages in Reverse 

    # Stage 10 - Backprop sum(losses)/num_train --> loss                #[10]
    dlosses = np.zeros((num_train)) + 1.0/num_train
    
    # Stage 9 - Backprop log_sums - correct_scores --> losses            #[9]
    dlog_sums = 1 * dlosses
    dcorrect_scores = -1 * dlosses
    
    # Stage 8 - Backprop log(sums) --> log_sums                          #[8]
    dsums = lg_log_sums * dlog_sums
    
    # Stage 7 - Backprop sum(exp_scores) --> sums                        #[7]
    dexp_scores = np.zeros(scores.shape) + dsums.reshape(-1,1)    # Broadcast operation

    # Stage 6 - Backprop exp(scores) --> exp_scores                      #[6]    
    dnorm_scores = lg_exp_scores * dexp_scores
    dnorm_scores_1 = np.copy(dnorm_scores)   # Store d_norm_scores before adding
                                  # back dcorrect_scores
    
    # Stage 5 - Backprop score of correct class --> correct_scores       #[5]    
    dnorm_scores[np.arange(num_train), y] += dcorrect_scores

    # Stage 4 - Backprop scores - max(scores) --> norm_scores            #[4]  
    dscores = dnorm_scores

    
    # 2-layer Neural Network Section:
    # Backward Pass - 3 Stages in Reverse

    # Stage 3 - Backprop h1.dot(W2)+b2 --> scores                        #[3] 
    dW2 += h1.T.dot(dscores)
    dh1 = dscores.dot(W2.T)
    db2 = np.sum(dscores, axis=0)

    # Stage 2 - Backprop ReLU(o1) --> h1                                 #[2]
    do1 = lg_h1 * dh1
    
    # Stage 1 - Backprop  X.dot(W1)+b1 --> o1                            #[1]    
    dW1 += X.T.dot(do1)
    db1 = np.sum(do1, axis=0)

    # Reg Backward Pass
    dW1 += 2 * 0.5 * reg * W1
    dW2 += 2 * 0.5 * reg * W2
    
    grads['W1'] = dW1
    grads['W2'] = dW2
    grads['b1'] = db1
    grads['b2'] = db2
    
    if verbose:
        print "\n Training data - Backward Pass" 
        print "\n dloss:"
        print dlosses
        print "\n dcorrect_scores:"
        print dcorrect_scores
        print "\n dlog_sums:"         
        print dlog_sums 
        print "\n lg_log_sums:"         
        print lg_log_sums
        print "\n sdums:" 
        print dsums 
        print "\n dexp_scores:" 
        print dexp_scores  
        print "\n lg_exp_scores:" 
        print lg_exp_scores
        print "\n dnorm_scores (before addback):" 
        print dnorm_scores_1
        print "\n dnorm_scores (after addback):" 
        print dnorm_scores 
        print "\n dscores:"
        print dscores
        print dscores.shape
        
        print "\n dW2:"
        print dW2
        print dW2.shape
        print "\n dh2:"
        print dh1
        print dh1.shape
        print "\n db2:"
        print db2
        print db2.shape        
        print "\n do1:"
        print do1
        print "\n dW1:"
        print dW1
        print "\n db1:"
        print db1
        print db1.shape
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads



  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      
      if num_train < batch_size:
            batch_size = num_train
            
      indx = np.random.choice(np.arange(num_train), batch_size, replace=False)
      X_batch = X[indx,:]
      y_batch = y[indx]

                    
      if verbose:
        print "\n Interation %d" % (it+1)
        print "Input Batch:"
        print X_batch
        print y_batch
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      # perform parameter update
      self.params['W1'] -= learning_rate*grads['W1']
      self.params['b1'] -= learning_rate*grads['b1']
      self.params['W2'] -= learning_rate*grads['W2']
      self.params['b2'] -= learning_rate*grads['b2']
      
      if verbose:
        print "Parameters W1, b1, W2, b2:"
        print self.params['W1']
        print self.params['b1']
        print self.params['W2']
        print self.params['b2']
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################



      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch, verbose=verbose) == y_batch).mean()
        val_acc = (self.predict(X_val, verbose=verbose) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        
      if verbose:   # and it % 100 == 0: 
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)
        print train_acc
        print val_acc
        
        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X, verbose=False):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape
    num_train = N
    num_classes = D
    
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    y_pred = np.zeros(X.shape[1])

    # Stage 1 - X.dot(W1)+b1 --> o1                               #[1]    
    o1= X.dot(W1) + b1
    
    # Stage 2 - ReLU(o1) --> h1                                   #[2]
    h1 = np.maximum(o1, np.zeros(o1.shape))

    # Stage 3 - h1.dot(W2)+b2 --> scores                          #[3]    
    scores = h1.dot(W2) + b2
    
    y_pred = np.argmax(scores, axis=1)
        
    if verbose:
      print scores
      print y_pred
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


