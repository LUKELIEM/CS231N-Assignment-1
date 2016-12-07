import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg, verbose=False):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  # Initialize intermediate values and local gradients for the 6 stages
  # of the computational graph implementation of SVM
    
  dW = np.zeros(W.shape) # the final gradient dW
  W_squared = np.zeros(W.shape)   # intermediate value W_squared

  scores = np.zeros((num_train,num_classes))  # intermediate value scores
  dscores = np.zeros((num_train,num_classes)) # and its gradient
    
  margins = np.zeros(scores.shape)  # intermediate value margins
  dmargins = np.zeros(scores.shape)  # and its gradient

  maxes = np.zeros(scores.shape)  # intermediate value maxes
  dmaxes = np.zeros(scores.shape)  # its gradient
  lg_maxes = np.zeros(scores.shape)  # its local gradient

  losses = np.zeros((num_train))  # loss and
  dlosses = np.zeros((num_train)) # its gradient
  data_loss = 0
  reg_loss = 0
  loss_check = 0

  """
  The SVM loss and gradient computation will be performed in 2 passes:
     - In the forward pass, the SVM loss calculation is decomposed into
       6 stages.
       At the end of the forward pass, loss is calculated
     - In the backward pass, we use chain rule to calculate the gradients
       from Stage 6 back to Stage 1
       At the end of the backward pass, the gradient dW is calculated
  """
    
  # SVM Forward Pass - 6 Stages in Total
  for i in xrange(num_train):
    
    # Stage 1 - dot product of X and W --> Scores                   #[1]
    scores[i] = X[i].dot(W)
    
    """
    The score vector consists of score of the correct class, which is
    referenced by y[i], and the scores of the incorrect classes. This
    2 types of scores are treated differently in Stage 2 and 3
    """
    
    # Stage 2 - correct_class_score * -1                            #[2]
    correct_class_score = scores[i, y[i]]   # Score of the correct 
                                            # class is referenced by y[i]
    minus_val = -1*correct_class_score
        
    # Stage 3 - scores + minus_val + 1 --> margins                  #[3]
    # This operation is performed only n the scores of the incorrect classes
    margins[i] = scores[i] + minus_val + 1
    margins[i, y[i]] = 0   # We zero out the margin of correct class
    
    # Stage 4 - max(0,margins) ==> maxes                            #[4]
    maxes[i] = np.maximum(margins[i],0)
    lg_maxes[i] = np.greater(maxes[i],0)*1  # local gradient for max operation

    # Stage 5 - sum(margins) ==> losses vector                      #[5]
    losses[i] = np.sum(maxes[i])

    # Debug Code - Check against original implementation
    correct = scores[i, y[i]]
    for j in xrange(num_classes):
      if j == y[i]: 
        continue
      m = scores[i,j] - correct + 1 
      if m > 0:
        loss_check += m
   
    if verbose:
        print "Training data %d - Forward Pass\n" %i 
        print scores[i]
        print correct_class_score
        print margins[i]
        print maxes[i]
        print lg_maxes[i]
        print losses[i]

  # Stage 6 - sum(Losses)/num_train ==> loss                        #[6]
  data_loss = np.sum(losses)/num_train

   
  """
  The regularization loss computation will also be performed in 2 passes:
     - In the forward pass, the regularizationloss calculation is decomposed
       into 3 stages.
       At the end of the forward pass, regularization loss is calculated
     - In the backward pass, we use chain rule to calculate the gradients
       from Stage 3 back to Stage 1
       At the end of the backward pass, the regularization component to the
       gradient dW is calculated
  """

  # Reg Forward Pass - 3 Stages in Total
  # Stage 1 - W*W (elementwise squaring)                            #[R1]
  W_squared = W * W

  # Stage 2 - np.sum(W_squared) --> sum_val                         #[R2]
  sum_val = np.sum(W_squared)

  # Stage 3 - sum_val * 0.5 * reg --> loss                          #[R3]
  reg_loss = 0.5 * reg * sum_val
  loss = data_loss + reg_loss  
  
  # Debug Code - Check against original implementation
  loss_check /= num_train
  if verbose:
    print "\nThese following two numbers should be equal"
    print data_loss
    print loss_check
    print reg_loss
    print loss
    
  # SVM Backward Pass - The same 6 Stages in reverse
  for i in xrange(num_train):
        
    # Stage 6 - Backprop sum(Losses)/num_train ==> loss             #[6]
    dlosses[i] = 1.0/num_train
 
    # Stage 5 - Backprop sum(margins) ==> losses                    #[5]
    dmargins[i] = dlosses[i]     
    
    # Stage 4 - Backprop max(0,margins1) ==> maxes                  #[4]
    dmaxes[i] = dmargins[i] * lg_maxes[i]  # Local gradient for each element of
                    # maxes vector is 1 if that element is > 0
     
    # We need to treat the gradient of the score of the correct class
    # differently than those of the incorrect classes
    
    # Stage 3  - Backprop scores + minus_score + 1 --> margins      #[3]
    # The local gradient of the score of the incorrect class is simply 1
    dscores[i] = 1 * dmaxes[i]
    # The local gradient of the score of the correct class is equal to the
    # number of incorrect classes whereby: 
    #        scores + minus_score + 1 > 0
    dscores[i,y[i]] = np.sum(dmaxes[i])    
    
    # Stage 2 - backprop correct_class_score * -1                   #[2]
    # local gradient of *-1 is -1
    dscores[i,y[i]] *= -1
        
    # backprop dot product of X and W
    dW += X[i].reshape(-1,1).dot(dscores[i].reshape(1,-1))          #[1]
    
    if verbose:
        print "Training data %d - Backward Pass\n" %i 
        print dlosses[i]
        print dmargins[i]
        print dmaxes[i]     
        print dscores[i]
  
  if verbose:
    print "dW (Gradient) Before Regularization:\n"
    print dW
    
  # Reg Backward Pass - 3 Stages in reverse

  # Stage 3 - Backprop sum_val * 0.5 * reg --> loss                 #[R3]
  dsum_val = 0.5 * reg 

  # Stage 2 - Backprop  np.sum(W_squared) --> sum_val               #[R2]
  dW_squared = dsum_val
    
  # Stage 1 - W*W (elementwise squaring)                            #[R1]
  dW += 2 * W * dW_squared  
    
  if verbose:
    print "dW (Gradient) after Regularization:\n"
    print dW
    print dsum_val
    print dW_squared
    print 2 * W * dW_squared 

  return loss, dW


def svm_loss_vectorized(W, X, y, reg, verbose=False):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """

  num_classes = W.shape[1]
  num_train = X.shape[0]


  # Initialize intermediate values and local gradients for the 6 stages
  # of the computational graph implementation of SVM
    
  dW = np.zeros(W.shape) # the final gradient dW
  W_squared = np.zeros(W.shape)   # intermediate value W_squared

  scores = np.zeros((num_train,num_classes))  # intermediate value scores
  dscores = np.zeros((num_train,num_classes)) # and its gradient
    
  margins = np.zeros(scores.shape)  # intermediate value margins
  dmargins = np.zeros(scores.shape)  # and its gradient

  maxes = np.zeros(scores.shape)  # intermediate value maxes
  dmaxes = np.zeros(scores.shape)  # its gradient
  lg_maxes = np.zeros(scores.shape)  # its local gradient

  losses = np.zeros((num_train))  # loss and
  dlosses = np.zeros((num_train)) # its gradient
  data_loss = 0
  reg_loss = 0
  loss = 0.0

  """
  The SVM loss and gradient computation will be performed in 2 passes:
     - In the forward pass, the SVM loss calculation is decomposed into
       6 stages.
       At the end of the forward pass, loss is calculated
     - In the backward pass, we use chain rule to calculate the gradients
       from Stage 6 back to Stage 1
       At the end of the backward pass, the gradient dW is calculated
  """
  
  # SVM Forward Pass - 6 Stages in Total  

  # Stage 1 - dot product of X and W --> Scores                   #[1]
  scores = X.dot(W)
    
  """
  The score vector consists of score of the correct class, which is
  referenced by y[i], and the scores of the incorrect classes. This
  2 types of scores are treated differently in Stage 2 and 3
  """
 
  # Stage 2 - correct_class_score * -1                            #[2]
  # Scores of the correct class for training data is indexed by y
  correct_class_score = scores[np.arange(num_train), y]   
  minus_val = -1 * correct_class_score

        
  # Stage 3 - scores + minus_val + 1 --> margins                  #[3]
  # This operation is performed only on the scores of the incorrect classes
  margins = scores + np.reshape(minus_val, (-1, 1)) + 1
  margins[np.arange(num_train), y] = 0   # zero out the margin of correct class
 
  # Stage 4 - max(0,margins) ==> maxes                            #[4]
  maxes = np.maximum(margins,0)
  lg_maxes = np.greater(maxes,0)*1  # local gradient for max operation

  # Stage 5 - sum(margins) ==> losses vector                      #[5]
  losses = np.sum(maxes, axis = 1)

  # Stage 6 - sum(Losses)/num_train ==> loss                      #[6]
  data_loss = np.sum(losses)/num_train

  """
  The regularization loss and grad computation will also be performed in 2 passes. 
  But each can be performed in a single matrix operation
  """
  # Reg Forward Pass 
  reg_loss = 0.5 * reg * np.sum(W*W)
    
  loss = data_loss + reg_loss
  
  if verbose:
    print "Training data %d - Forward Pass\n"
    print "Parameters:"
    print W
    print "\nTraining Data:"
    print X
    print "\nTrue Labels:"
    print y
    print "\nScore (XW):"
    print scores
    print correct_class_score
    print minus_val
    print margins
    print maxes
    print lg_maxes
    print losses
    print data_loss
    print reg_loss

  # SVM Backward Pass - The same 6 Stages in reverse

  # Stage 6 - Backprop sum(Losses)/num_train ==> loss             #[6]
  dlosses = np.zeros((num_train)) + 1.0/num_train

  # Stage 5 - Backprop sum(margins) ==> losses                    #[5]
  dmargins = np.zeros(scores.shape) + dlosses.reshape(-1,1)    

       
  # Stage 4 - Backprop max(0,margins1) ==> maxes                  #[4]
  dmaxes = dmargins * lg_maxes  # Local gradient for each element of
                    # maxes vector is 1 if that element is > 0
  
  # We need to treat the gradient of the score of the correct class
  # differently than those of the incorrect classes
    
  # Stage 3  - Backprop scores + minus_score + 1 --> margins      #[3]
  # The local gradient of the score of the incorrect class is simply 1
  dscores = 1 * dmaxes
  # The local gradient of the score of the correct class is equal to the
  # number of incorrect classes whereby: 
  #        scores + minus_score + 1 > 0
  dscores[np.arange(num_train), y] = np.sum(dmaxes,axis=1).reshape(1,-1)   
    
 
  # Stage 2 - backprop correct_class_score * -1                   #[2]
  # local gradient of *-1 is -1
  dscores[np.arange(num_train), y] *= -1
     
  # backprop dot product of X and W
  dW += X.T.dot(dscores)                                          #[1]
    
  if verbose:
    print "Training data - Backward Pass\n" 
    print dlosses
    print dmargins
    print dmaxes
    print dscores
    print dscores.shape
    print X.shape
    print W.shape
    print "dW (Gradient) Before Regularization:\n"
    print dW
 
  # Reg Backward Pass 
  dW += 2 * 0.5 * reg * W

  if verbose:
    print "dW (Gradient) after Regularization:\n"
    print dW
    
  return loss, dW
