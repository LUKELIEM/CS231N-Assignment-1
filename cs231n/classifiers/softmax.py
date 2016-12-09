import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg, verbose=False):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  # Initialize intermediate values and local gradients for the 6 stages
  # of the computational graph implementation of SVM

  scores = np.zeros((num_train,num_classes))  # intermediate value scores
  dscores = np.zeros((num_train,num_classes)) # and its gradient

  norm_scores = np.zeros(scores.shape)  # intermediate value margins
  dnorm_scores = np.zeros(scores.shape)  # and its gradient
  
  dnorm_scores_1  = np.zeros(scores.shape)  # and its gradient

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
 

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  """
  The Softmax loss and gradient computation will be performed in 2 passes:
     - In the forward pass, the SVM loss calculation is decomposed into
       8 stages.
       At the end of the forward pass, loss is calculated
     - In the backward pass, we use chain rule to calculate the gradients
       from Stage 8 back to Stage 1
       At the end of the backward pass, the gradient dW is calculated
       
       L(i) = -Score_Correct_Class + ln(Summation(exp(Scores_of_all_Classes)))
  """
  
  # Softmax Forward Pass - 8 Stages in Total  

  for i in xrange(num_train):
    # Stage 1 - dot product of X and W --> scores                        #[1]
    scores[i] = X[i].dot(W)

    # Stage 2 - scores - scale_const --> norm_scores                     #[2]
    # The scaling constant is an arbitrary constant. In our implementation, we
    # pick the maximum in scores, so that norm_scores are all 0 or less. This 
    # is to prevent overflow error in subsequent calculation.
    # The local gradient in this gate is thus d(x+c)/dx = 1
    scale_const = np.max(scores[i])
    norm_scores[i] = scores[i] - scale_const

    # Stage 3 - score of correct class --> correct_scores                #[3]    
    correct_scores[i] = norm_scores[i, y[i]]

    # Stage 4 - exp(scores) --> exp_scores                               #[4]    
    exp_scores[i] = np.exp(norm_scores[i])
    lg_exp_scores[i] = np.exp(norm_scores[i])  # local gradient of exp(x) = exp(x)

    # Stage 5 - sum(exp_scores) --> sums                                 #[5]
    sums[i] = np.sum(exp_scores[i])
    
    # Stage 6 - log(sums) --> log_sums                                   #[6]
    log_sums[i] = np.log(sums[i])
    lg_log_sums[i] = 1.0/sums[i]  # local gradient of ln(x) = 1/x
    
    # Stage 7 - log_sums - correct_scores --> losses                     #[7]
    losses[i] = log_sums[i] - correct_scores[i]

    # Stage 8 - sum(losses)/num_train --> loss                           #[8]
    data_loss = np.sum(losses)/num_train

  """
  The regularization loss and grad computation will also be performed in 2 passes. 
  Each can be performed in a single matrix operation
  """
  # Reg Forward Pass 
  reg_loss = 0.5 * reg * np.sum(W*W)
    
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

        
  # Softmax Backward Pass - 8 Stages in Reverse 

  for i in xrange(num_train):
        
    # Stage 8 - Backprop sum(losses)/num_train --> loss                    #[8]
    dlosses[i] = 1.0/num_train
    
    # Stage 7 - Backprop log_sums - correct_scores --> losses              #[7]
    dlog_sums[i] = 1 * dlosses[i]
    dcorrect_scores[i] = -1 * dlosses[i]
    
    # Stage 6 - Backprop og(sums) --> log_sums                             #[6]
    dsums[i] = lg_log_sums[i] * dlog_sums[i]
    
    # Stage 5 - Backprop sum(exp_scores) --> sums                          #[5]
    dexp_scores[i] = dsums[i]   # Broadcast operation

    # Stage 4 - Backprop exp(scores) --> exp_scores                        #[4]    
    dnorm_scores[i] = lg_exp_scores[i] * dexp_scores[i]
    dnorm_scores_1[i] = dnorm_scores[i]   # Store d_norm_scores before adding
                                          # back dcorrect_scores
    
    # Stage 3 - Backprop score of correct class --> correct_scores         #[3]    
    dnorm_scores[i, y[i]] += dcorrect_scores[i]

    # Stage 2 - Backprop scores - max(scores) --> norm_scores              #[2]  
    dscores[i] = dnorm_scores[i]

    # Stage 1 - Backprop dot product of X and W --> scores                 #[1]
    dW += X[i].reshape(-1,1).dot(dscores[i].reshape(1,-1))
    
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
      print "dW (Gradient) Before Regularization:\n"
      print dW
  
  # Reg Backward Pass 
  dW += 2 * 0.5 * reg * W

  if verbose:
    print "dW (Gradient) after Regularization:\n"
    print dW    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg, verbose=False):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  # Initialize intermediate values and local gradients for the 6 stages
  # of the computational graph implementation of SVM
    
  W_squared = np.zeros(W.shape)   # intermediate value W_squared

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
 

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  """
  The Softmax loss and gradient computation will be performed in 2 passes:
     - In the forward pass, the SVM loss calculation is decomposed into
       8 stages.
       At the end of the forward pass, loss is calculated
     - In the backward pass, we use chain rule to calculate the gradients
       from Stage 8 back to Stage 1
       At the end of the backward pass, the gradient dW is calculated
       
       L(i) = -Score_Correct_Class + ln(Summation(exp(Scores_of_all_Classes)))
  """
  # Stage 1 - dot product of X and W --> scores                        #[1]
  scores = X.dot(W)

  # Stage 2 - scores - scale_const --> norm_scores                     #[2]
  # The scaling constant is an arbitrary constant. In our implementation, we
  # pick the maximum in scores, so that norm_scores are all 0 or less. This 
  # is to prevent overflow error in subsequent calculation.
  # The local gradient in this gate is thus d(x+c)/dx = 1
  scale_const = np.max(scores)
  norm_scores = scores - scale_const

  # Stage 3 - score of correct class --> correct_scores                #[3]    
  correct_scores = norm_scores[np.arange(num_train), y]

  # Stage 4 - exp(scores) --> exp_scores                               #[4]    
  exp_scores = np.exp(norm_scores)
  lg_exp_scores = np.exp(norm_scores)  # local gradient of exp(x) = exp(x)

  # Stage 5 - sum(exp_scores) --> sums                                 #[5]
  sums = np.sum(exp_scores, axis=1)
    
  # Stage 6 - log(sums) --> log_sums                                   #[6]
  log_sums = np.log(sums)
  lg_log_sums = 1.0/sums               # local gradient of ln(x) = 1/x
    
  # Stage 7 - log_sums - correct_scores --> losses                     #[7]
  losses = log_sums - correct_scores

  # Stage 8 - sum(losses)/num_train --> loss                           #[8]
  data_loss = np.sum(losses)/num_train

  # Reg Forward Pass 
  reg_loss = 0.5 * reg * np.sum(W*W)
    
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

  # Softmax Backward Pass - 8 Stages in Reverse 

  # Stage 8 - Backprop sum(losses)/num_train --> loss                    #[8]
  dlosses = np.zeros((num_train)) + 1.0/num_train
    
  # Stage 7 - Backprop log_sums - correct_scores --> losses              #[7]
  dlog_sums = 1 * dlosses
  dcorrect_scores = -1 * dlosses
    
  # Stage 6 - Backprop log(sums) --> log_sums                             #[6]
  dsums = lg_log_sums * dlog_sums
    
  # Stage 5 - Backprop sum(exp_scores) --> sums                          #[5]
  dexp_scores = np.zeros(scores.shape) + dsums.reshape(-1,1)    # Broadcast operation

  # Stage 4 - Backprop exp(scores) --> exp_scores                        #[4]    
  dnorm_scores = lg_exp_scores * dexp_scores
  dnorm_scores_1 = np.copy(dnorm_scores)   # Store d_norm_scores before adding
                                  # back dcorrect_scores
           
  # Stage 3 - Backprop score of correct class --> correct_scores         #[3]    
  dnorm_scores[np.arange(num_train), y] += dcorrect_scores
    
  # Stage 2 - Backprop scores - max(scores) --> norm_scores              #[2]  
  dscores = dnorm_scores

  # Stage 1 - Backprop dot product of X and W --> scores                 #[1]
  dW += X.T.dot(dscores)
    
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
      print "dW (Gradient) Before Regularization:\n"
      print dW
  
  # Reg Backward Pass 
  dW += 2 * 0.5 * reg * W

  if verbose:
    print "dW (Gradient) after Regularization:\n"
    print dW    

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

