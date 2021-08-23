'''
    Machine learning python project 1
    Michal Keren 
    Itamar Eyal 
    Libraries: numpy, matplotlib & sklearn.
    After running the file, 2 plots will show for sections 8,9.
    Results for section 10 will be printed to the console.
    Maximal number of epochs is 10**5.
    Running time of about 35 sec.
'''


'''
    IMPORTS
'''
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import numpy as np

'''
    DEFINES
'''
NUMBER_OF_LABELS = 10  # change to classes
IMG_SIZE = 28

'''
    IMPLEMENTATIONS
'''
def hot_vectors(t):
    h = np.zeros((t.size, 10))
    h[np.arange(t.size), t] = 1
    return h

def softmax(W,X):
    a = W.dot(X.T).T.astype('float64')
    exp_a = np.exp(a - np.max(a, axis=1, keepdims=True))
    return np.divide(exp_a , np.sum(exp_a, axis=1, keepdims=True))

def cross_entropy_loss(y, t):
    return -np.sum(np.log(y[t==1]))

def get_set_accuracy(W,X,t):
    y = softmax(W, X)
    guesses = y.argmax(axis=1)
    answers = t.argmax(axis=1)
    corrects_mat = np.equal(guesses, answers)
    corrects = np.sum(corrects_mat)  # num of correct guesses
    accuracy = corrects / guesses.shape[0] * 100  # correct guesses in %
    return accuracy

'''
    EXECUTION
'''
# 1 & 2- load MNIST DataBase & flatten the data from pictures to vectors.
mnist = fetch_openml('mnist_784')
X = mnist['data'].astype('float64')#each row is a 28x28 photo. (flatten the data from pictures to vectors.)
t = mnist['target'].astype('int64')
print("DB loaded.")

t= hot_vectors(t) #transform t into hot vectors array.

# shuffle the DataBase
random_state = check_random_state(1)
permutation = random_state.permutation(X.shape[0]) # a random arrangement of the indexes in the range [0,X.shape[0]]
X = X[permutation] # shuffling the order of the pictures.
t = t[permutation] # arranging the correct labelflatten the data from pictures to vectorss accordingly.

# 3- construct the X matrix.
X= np.c_[X, np.ones(X.shape[0]).astype('float16')] # adding '1' to the end of each photo vector.
                                                   #X=[x0^T,x1^T...,x9^T]^T (10 rows)

# 4- split the DataBse into: training set- 60%, validation set- 20%, test set-20%.
X_train, X_test, t_train, t_test = train_test_split(X, t, train_size= 0.6)
X_test, X_val,t_test, t_val= train_test_split(X_test, t_test, test_size= 0.5) # split half of the test_set into validation set.

# 5 - initialize the Wights vectors, values [0,0.1]
W = np.random.uniform(low=0, high=0.1, size=(NUMBER_OF_LABELS,IMG_SIZE*IMG_SIZE + 1)).astype('float64')

# The next lines standardize the images
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

# lists to hold plotting data
E_lst = []
VS_accuracy_lst = []

eta = 0.00001
precision = 0.001 #The condition for convergence
max_epoch = 10**5# maximum number of iterations
i =0  # index of iteration
accuracy_diff= precision+1 #init

# The GD iteration loop. every iteration trains the W matrix and calculates the validationSet accuracy.
while accuracy_diff > precision and i < max_epoch:
    y = softmax(W,X_train)

    # 6- the Error function:
    E_lst.append(cross_entropy_loss(y, t_train)) # add to list for plotting

    grad_E = X_train.T.dot(y-t_train) # the gradient of loss.
    # 7- gradient descent
    W = W - eta*grad_E.T # update W

    # calc Validation Set accuracy
    valSet_accuracy =get_set_accuracy(W,X_val,t_val)
    if(len(VS_accuracy_lst)>0):
        accuracy_diff = abs(VS_accuracy_lst[-1]-valSet_accuracy)
    VS_accuracy_lst.append(valSet_accuracy)
    print("Epoch #"+str(i))
    i += 1
i_lst = np.arange(0, i)

# 8- Cross Entropy Loss on train set as a function of iteration
plt.scatter(i_lst, E_lst)
plt.plot(i_lst, E_lst)
plt.suptitle('Training Set Loss Vs. Epoch', fontsize=14, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")

# 9- Precision as a function of iteration for validation set
plot_cp= plt.figure(2)
plt.scatter(i_lst, VS_accuracy_lst)
plt.plot(i_lst, VS_accuracy_lst)
plt.suptitle('validation set accuracy Vs. Epoch', fontsize=14, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("accuracy[%]")

# 10- results on every set after last iteration
trainSet_accuracy =str("%.2f" % get_set_accuracy(W,X_train,t_train))
testSet_accuracy =str("%.2f" % get_set_accuracy(W,X_test,t_test))
print("------final accuracy Values:------")
print("Training Set accuracy: "+trainSet_accuracy+"%")
print("Test Set accuracy: "+testSet_accuracy+"%")
print("Validation Set accuracy: "+str("%.2f" % VS_accuracy_lst[-1])+"%")

# present plots
plt.show()
