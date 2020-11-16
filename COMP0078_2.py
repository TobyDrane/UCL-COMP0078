# Feature maps old data [(x1, y1), (x2, y2)]
# to new -> [((x1 * k, ..., xk * k), y1) ...]
def basis_feature_map(x, k):
    # Blank empty 2D array of rows of input and cols of k base size
    X = numpy.empty(shape=(len(x), k))
    
    for elem in range(len(x)):
        item = x[elem]
        new_element = []
        # This basis is xk ** (k-1), NOTE: No -1 due to loop starting at 0
        for i in range(k):
            new_element.append(item ** i)
            
        X[elem] = new_element

    X = numpy.matrix(X)
    return X.astype(numpy.float32)

# Similar to setup to that of above but new basis function is
# k sin x
def sin_basis_feature_map(x, k):
    X = numpy.empty(shape=(len(x), k))
    
    for elem in range(len(x)):
        item = x[elem]
        new_element = []
        
        for i in range(1, k + 1):
            new_element.append(numpy.sin(i * numpy.pi * item))
        
        X[elem] = new_element
    
    X = numpy.matrix(X)
    return X.astype(numpy.float32)
 
# Calculate the weight value based upon the matrices
# (XTX)^-1(XT)Y
def calculate_weight(X, Y):
    XT = numpy.transpose(X)
    w = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(XT, X)), XT), Y)
    return w

# MSE is sum of squares / n
def calculate_MSE(X, Y, w):
    # Sum of sqaure errors
    SSE = 0
    # Our predicted y value
    yPred = X@w
    
    for i in range(len(yPred)):
        SSE += math.pow((Y[i] - yPred[i]), 2)
    
    # MSE is SSE / m
    return SSE / len(Y)

# A more general function to that of above
def calculate_ridge_MSE(Y, YPred):
    SSE = 0.0
    for i in range(len(Y)):
        SSE += math.pow((Y[i] - YPred[i]), 2)
    return SSE / len(Y)

def gx(x):
    # Function produces the g(x) sin^2(2PIX)
    y = numpy.square(numpy.sin(2 * numpy.pi * x))
    return y

def q1_1_2():
    noise = numpy.random.normal(0, 0.07, 30)
    x = numpy.random.uniform(0, 1, 30)
    y = gx(x) + noise

    Y = numpy.matrix(y).T
    
    xd = numpy.linspace(0, 1, 50)
    
    plot.scatter(x, y)
    plot.plot(xd, gx(xd), color='black')
    plot.xlabel('x')
    plot.ylabel('y')
    plot.show()
    
    w2 = calculate_weight(basis_feature_map(x, 2), Y)
    w5 = calculate_weight(basis_feature_map(x, 5), Y)
    w10 = calculate_weight(basis_feature_map(x, 10), Y)
    w14 = calculate_weight(basis_feature_map(x, 14), Y)
    w18 = calculate_weight(basis_feature_map(x, 18), Y)
        
    plot.scatter(x, y)
    
    p1, = plot.plot(xd, basis_feature_map(xd, 2)@w2)
    p2, = plot.plot(xd, basis_feature_map(xd, 5)@w5)
    p3, = plot.plot(xd, basis_feature_map(xd, 10)@w10)
    p4, = plot.plot(xd, basis_feature_map(xd, 14)@w14)
    p5, = plot.plot(xd, basis_feature_map(xd, 18)@w18)
    
    plot.xlabel('x')
    plot.ylabel('y')
    plot.ylim((-0.5, 1.5))
    plot.xlim((0, 1))
    plot.legend(handles=[p1, p2, p3, p4, p5], labels=['k=2', 'k=5','k=10','k=14', 'k=18'], loc='best')
    plot.show()
               
def q1_1_2_bc():
    noise = numpy.random.normal(0, 0.07, 30)
    x = numpy.random.uniform(0, 1, 30)
    y = gx(x) + noise
    Y = numpy.matrix(y).T
        
    tnoise = numpy.random.normal(0, 0.07, 1000)
    tx = numpy.random.uniform(0, 1, 1000)
    ty = gx(tx) + tnoise
    tY = numpy.matrix(ty).T
    
    errors = numpy.zeros(shape=(18, 1))
    errors_test = numpy.zeros(shape=(18, 1))

    for k in range(1, 19):
        fmap = numpy.around(basis_feature_map(x, k).astype(numpy.float64), decimals=7)
        w = calculate_weight(fmap, Y)
        errors[k-1] = numpy.log(calculate_MSE(fmap, Y, w))
        
        tfmap = numpy.around(basis_feature_map(tx, k).astype(numpy.float64), decimals=7)
        errors_test[k-1] = numpy.log(calculate_MSE(tfmap, tY, w))
    
    plot.plot(numpy.linspace(0, 18, 18), errors)
    plot.xlabel('k')
    plot.ylabel('ln(MSE)')
    plot.xlim(1, 18)
    plot.title(label='ln(MSE) of training data')
    plot.show()
    
    plot.plot(numpy.linspace(0, 18, 18), errors_test)
    plot.xlabel('k')
    plot.ylabel('ln(MSE)')
    plot.title(label='ln(MSE) of test data')
    plot.xlim(1, 18)
    plot.show()
    
def q1_1_3():
    noise = numpy.random.normal(0, 0.07, 30)
    x = numpy.random.uniform(0, 1, 30)
    y = gx(x) + noise
    Y = numpy.matrix(y).T
        
    tnoise = numpy.random.normal(0, 0.07, 1000)
    tx = numpy.random.uniform(0, 1, 1000)
    ty = gx(tx) + tnoise
    tY = numpy.matrix(ty).T 
    
    errors = numpy.zeros(shape=(18, 1))
    errors_test = numpy.zeros(shape=(18, 1))


    for k in range(1, 19):
        fmap = numpy.around(sin_basis_feature_map(x, k).astype(numpy.float64), decimals=7)
        w = calculate_weight(fmap, Y)
        errors[k-1] = numpy.log(calculate_MSE(fmap, Y, w))
        
        tfmap = numpy.around(sin_basis_feature_map(tx, k).astype(numpy.float64), decimals=7)
        errors_test[k-1] = numpy.log(calculate_MSE(tfmap, tY, w))
    
    plot.plot(numpy.linspace(0, 18, 18), errors)
    plot.xlabel('k')
    plot.ylabel('ln(MSE)')
    plot.xlim(1, 18)
    plot.title(label='ln(MSE) of training data')
    plot.show()
    
    plot.plot(numpy.linspace(0, 18, 18), errors_test)
    plot.xlabel('k')
    plot.ylabel('ln(MSE)')
    plot.title(label='ln(MSE) of test data')
    plot.xlim(1, 18)
    plot.show()
    
    def single_1000_MSE_new_basis(x, y, tx, ty):
        Y = numpy.matrix(y).T
        tY = numpy.matrix(ty).T
        
        errors = numpy.zeros(shape=(18, 1))
        errors_test = numpy.zeros(shape=(18, 1))
       
        for k in range(1, 19):
            fmap = numpy.around(sin_basis_feature_map(x, k).astype(numpy.float64), decimals=7)
            tfmap = numpy.around(sin_basis_feature_map(tx, k).astype(numpy.float64), decimals=7)
            
            w = calculate_weight(fmap, Y)
            
            errors[k-1] = calculate_MSE(fmap, Y, w)
            errors_test[k-1] = calculate_MSE(tfmap, tY, w)
        
        return errors, errors_test
    
    """
    
    The repeat of part d for question 3
    
    """
    errors_sum_train = numpy.zeros(shape=(100, 18))
    errors_sum_test = numpy.zeros(shape=(100, 18))

    for r in tqdm(range(100)):
        noise = numpy.random.normal(0, 0.07, 30)
        x = numpy.random.uniform(0, 1, 30)
        y = gx(x) + noise
    
        tnoise = numpy.random.normal(0, 0.07, 1000)
        tx = numpy.random.uniform(0, 1, 1000)
        ty = gx(tx) + tnoise
        
        e, te = single_1000_MSE_new_basis(x, y, tx, ty)
        errors_sum_train[r] = e.T
        errors_sum_test[r] = te.T
        
    errors_train = numpy.log(numpy.average(errors_sum_train, axis=0))
    errors_test = numpy.log(numpy.average(errors_sum_test, axis=0))

    spacing = numpy.linspace(0, 18, 18)
    p1, = plot.plot(spacing, errors_train)
    p2, = plot.plot(spacing, errors_test)
    plot.xlabel('k')
    plot.ylabel('natural log of MSE')
    plot.xlim(1, 18)
    plot.legend(handles=[p1, p2], labels=["train", "test"], loc="best")
    plot.show()
        

def single_1000_MSE(x, y, tx, ty):
    Y = numpy.matrix(y).T
    tY = numpy.matrix(ty).T
    
    errors = numpy.zeros(shape=(18, 1))
    errors_test = numpy.zeros(shape=(18, 1))
   
    for k in range(1, 19):
        fmap = numpy.around(basis_feature_map(x, k).astype(numpy.float64), decimals=7)
        tfmap = numpy.around(basis_feature_map(tx, k).astype(numpy.float64), decimals=7)
        
        w = calculate_weight(fmap, Y)
        
        errors[k-1] = calculate_MSE(fmap, Y, w)
        errors_test[k-1] = calculate_MSE(tfmap, tY, w)
    
    return errors, errors_test

def q1_1_2_d():
    errors_sum_train = numpy.zeros(shape=(100, 18))
    errors_sum_test = numpy.zeros(shape=(100, 18))

    for r in tqdm(range(100)):
        noise = numpy.random.normal(0, 0.07, 30)
        x = numpy.random.uniform(0, 1, 30)
        y = gx(x) + noise
    
        tnoise = numpy.random.normal(0, 0.07, 1000)
        tx = numpy.random.uniform(0, 1, 1000)
        ty = gx(tx) + tnoise
        
        e, te = single_1000_MSE(x, y, tx, ty)
        errors_sum_train[r] = e.T
        errors_sum_test[r] = te.T
        
    errors_train = numpy.log(numpy.average(errors_sum_train, axis=0))
    errors_test = numpy.log(numpy.average(errors_sum_test, axis=0))

    spacing = numpy.linspace(0, 18, 18)
    p1, = plot.plot(spacing, errors_train)
    p2, = plot.plot(spacing, errors_test)
    plot.xlabel('k')
    plot.ylabel('natural log of MSE')
    plot.xlim(1, 18)
    plot.legend(handles=[p1, p2], labels=["train", "test"], loc="best")
    plot.show()

def q1_1():
    data = numpy.array([(1, 3), (2, 2), (3, 0), (4, 5)])
    
    x = data[:, 0]
    y = data[:, 1]
    Y = numpy.matrix(y).T
    
    # Calculate all the required weights
    w1 = calculate_weight(basis_feature_map(x, 1), Y)
    w2 = calculate_weight(basis_feature_map(x, 2), Y)
    w3 = calculate_weight(basis_feature_map(x, 3), Y)
    w4 = calculate_weight(basis_feature_map(x, 4), Y)
    
    # Even data to produce curves from    
    xd = numpy.linspace(0, 4.5, 100)
    
    # Let's draw the scatter diagram
    plot.scatter(x, y)
    
    p1, = plot.plot(xd, basis_feature_map(xd, 1)@w1)
    p2, = plot.plot(xd, basis_feature_map(xd, 2)@w2)
    p3, = plot.plot(xd, basis_feature_map(xd, 3)@w3)
    p4, = plot.plot(xd, basis_feature_map(xd, 4)@w4)
    
    plot.xlabel('x')
    plot.ylabel('y')
    plot.legend(handles=[p1, p2, p3, p4], labels=['k=1', 'k=2','k=3','k=4'], loc='best')

    print(basis_feature_map(x, 4).shape, Y.shape, w4.shape)

    print('MSE k1 = ', calculate_MSE(basis_feature_map(x, 1), Y, w1))
    print('MSE k2 = ', calculate_MSE(basis_feature_map(x, 2), Y, w2))
    print('MSE k3 = ', calculate_MSE(basis_feature_map(x, 3), Y, w3))
    print('MSE k4 = ', calculate_MSE(basis_feature_map(x, 4), Y, w4))

    plot.show()
    
    
"""

Handle the Boston Filtered Data

"""
url = 'http://www.cs.ucl.ac.uk/staff/M.Herbster/boston-filter/Boston-filtered.csv'
data = pandas.read_csv(url)
# Pandas can just spit out a numpy array
data_numpy_array = data.to_numpy()

# The first 0-12 attributes are the attributes used to construct a predicition
boston_X = data_numpy_array[:, : -1]
# Last attribute is the output value, i.e. what we are trying to predict
boston_Y = data_numpy_array[:, -1 :]    

def q1_2_a():
    train_error = []
    test_error = []

    for i in range(20):
        X_train, X_test, Y_train, Y_test = train_test_split(boston_X, boston_Y, test_size=0.33)
        train_ones = numpy.asmatrix(numpy.ones(len(X_train))).T
        test_ones = numpy.asmatrix(numpy.ones(len(X_test))).T
        
        train_error.append(calculate_MSE(train_ones, Y_train, calculate_weight(train_ones, Y_train)))
        test_error.append(calculate_MSE(test_ones, Y_test, calculate_weight(test_ones, Y_test)))

    print('Train MSE error: ', (numpy.sum(train_error) / 20), 'Test MSE error: ', (numpy.sum(test_error) / 20))
    print(numpy.std(train_error), ' Test: ', numpy.std(test_error))

def q1_2_c():
    for a in range(12):
        train_error = []
        test_error = []      

        for i in range(20):
            train_data, test_data = train_test_split(data_numpy_array, test_size=0.33)
            X_train = numpy.matrix([train_data[:, a], numpy.ones(len(train_data))]).T
            Y_train = train_data[:, -1 :]

            X_test = numpy.matrix([test_data[:, a], numpy.ones(len(test_data))]).T
            Y_test = test_data[:, -1 :]

            train_error.append(calculate_MSE(X_train, Y_train, calculate_weight(X_train, Y_train)))
            test_error.append(calculate_MSE(X_test, Y_test, calculate_weight(X_test, Y_test)))
        
        print('Attribute ', (a + 1), ' Train MSE error: ', (numpy.sum(train_error) / 20), ' Test MSE error: ', (numpy.sum(test_error) / 20))
        print(numpy.std(train_error), ' Test: ', numpy.std(test_error))

def q1_2_d():
    train_error = []
    test_error = []

    train_data, test_data = train_test_split(data_numpy_array, test_size=0.33)

    for i in range(20):
        train_data, test_data = train_test_split(data_numpy_array, test_size=0.33)
        X_train = numpy.hstack((train_data[:, : -1], numpy.ones(shape=(len(train_data), 1))))
        Y_train = train_data[:, -1 :]
        
        X_test = numpy.hstack((test_data[:, : -1], numpy.ones(shape=(len(test_data), 1))))
        Y_test = test_data[:, -1 :]
        
        train_error.append(calculate_MSE(X_train, Y_train, calculate_weight(X_train, Y_train)))
        test_error.append(calculate_MSE(X_test, Y_test, calculate_weight(X_test, Y_test)))

    print('Train MSE error: ', (numpy.sum(train_error) / 20), 'Test MSE error: ', (numpy.sum(test_error) / 20))
    print(numpy.std(train_error), ' Test: ', numpy.std(test_error))

# A single valued Guass kernel function, this DOES not perform a Guass kernel
# for an enitre data set, alas per item only
# For use with the create_kernel function
def Gaussian_kernel(xi, xj, sigma):
    # Return a single Kernel value
    norm = numpy.linalg.norm((xi - xj), ord=2)
    K = numpy.exp(-(math.pow(norm, 2)) / (2 * math.pow(sigma, 2)))
    return K

def ridge_regression(K, Y, gamma):
    l = len(K)
    z = numpy.dot(gamma, numpy.dot(l, numpy.identity(l)))
    alpha = numpy.dot(numpy.linalg.inv(K + z), Y)
    
    # An array of values
    return alpha

def ridge_regression_prediction(X, X_test, alpha, sigma):
    predicted = []
    for t in range(len(X_test)):
        current_prediction = 0
        for i in range(len(X)):
            current_prediction += numpy.dot(alpha[i], Gaussian_kernel(X[i], X_test[t], sigma)).item()
        
        predicted.append(current_prediction)
    
    return numpy.array(predicted)
 
def create_kernel(X, sigma):
    m = X.shape[0]
    
    # The Kernel m x m matrix
    K = numpy.zeros(shape=(m, m))
    
    for i in range(m):
        for j in range(m):   
            K[i][j] = Gaussian_kernel(X[i], X[j], sigma)
            
    return numpy.matrix(K)
   
def K_Fold(X, Y, k):
    # Note here X is the data where we split by axis 0
    # Need to copy as we delete rows from the data here
    X_copy = X
    Y_copy = Y
    
    split_size = int(X_copy.shape[0] / k)
    
    X_folds = []
    Y_folds = []
    
    for i in range(k):
        X_split = []
        Y_split = []
        
        while len(X_split) < split_size:
            index = random.randrange(X_copy.shape[0])
            X_split.append(X_copy[index])
            Y_split.append(Y_copy[index])

            # Delete item from array copy so we don't pick it up again in the future
            numpy.delete(X_copy, index, 0)
            numpy.delete(Y_copy, index, 0)
        
        X_folds.append(numpy.asarray(X_split))
        Y_folds.append(numpy.asarray(Y_split))
    
    # Returns a shape of (k, X.shape[0] / k, X.shape[1])
    # e.g. for K = 5 (5, 67, 12)
    return numpy.asarray(X_folds), numpy.asarray(Y_folds)
    
def q1_3():
    K_splits = 5
    
    X_train, X_test, Y_train, Y_test = train_test_split(boston_X, boston_Y, test_size=0.33)

    gamma = numpy.array([math.pow(2, i) for i in numpy.arange(-40, -25, 1)])
    sigma = numpy.array([math.pow(2, i) for i in numpy.arange(7, 13.5, 0.5)])
        
    Z = numpy.zeros((len(gamma), len(sigma)))
    
    best_mse = numpy.Infinity
    best_gamma_index = -1
    best_sigma_index = -1
    
    X_fold, Y_fold = K_Fold(X_train, Y_train, K_splits)
    for g in tqdm(range(len(gamma))):
        for s in range(len(sigma)):
            _gamma = gamma[g]
            _sigma = sigma[s]
            
            MSE = []
            for i in range(K_splits):
                X_copy = X_fold
                Y_copy = Y_fold
                XT = X_copy[i]
                YT = Y_copy[i]
                X = numpy.row_stack(numpy.delete(X_copy, i, 0))
                Y = numpy.row_stack(numpy.delete(Y_copy, i, 0))
                
                K_train = create_kernel(X, _sigma)
                alpha = ridge_regression(K_train, Y, _gamma)
                predicted = ridge_regression_prediction(X, XT, alpha, _sigma)
                error = calculate_ridge_MSE(YT, predicted)
                MSE.append(error)
            
            mean_mse = numpy.mean(MSE).item()    
            Z[g, s] = mean_mse
            
            if mean_mse < best_mse:
                best_mse = mean_mse
                best_gamma_index = g
                best_sigma_index = s

    print('Best MSE: ', best_mse, ' Best gamma: ', gamma[best_gamma_index], ' Best sigma: ', sigma[best_sigma_index])
    fig = plot.figure()
    ax = plot.axes(projection='3d')
    
    X, Y = numpy.meshgrid(sigma, gamma)

    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_ylabel('gamma')
    ax.set_xlabel('sigma')
    plot.xlim(2**7,2**13)
    plot.ylim(2**(-40),2**(-26))
    ax.set_zlabel('error')
    plot.show()
    
    plot.matshow(Z)
    plot.show()

def q1_3_d():
    K_splits = 5
    
    X_train, X_test, Y_train, Y_test = train_test_split(boston_X, boston_Y, test_size=0.33)

    gamma = numpy.array([math.pow(2, i) for i in numpy.arange(-40, -25, 1)])
    sigma = numpy.array([math.pow(2, i) for i in numpy.arange(7, 13.5, 0.5)])
        
    Z = numpy.zeros((len(gamma), len(sigma)))
    
    best_mse = numpy.Infinity
    best_gamma_index = -1
    best_sigma_index = -1
    
    X_fold, Y_fold = K_Fold(X_train, Y_train, K_splits)
    for g in tqdm(range(len(gamma))):
        for s in range(len(sigma)):
            _gamma = gamma[g]
            _sigma = sigma[s]
            
            MSE = []
            for i in range(K_splits):
                X_copy = X_fold
                Y_copy = Y_fold
                XT = X_copy[i]
                YT = Y_copy[i]
                X = numpy.row_stack(numpy.delete(X_copy, i, 0))
                Y = numpy.row_stack(numpy.delete(Y_copy, i, 0))
                
                K_train = create_kernel(X, _sigma)
                alpha = ridge_regression(K_train, Y, _gamma)
                predicted = ridge_regression_prediction(X, XT, alpha, _sigma)
                error = calculate_ridge_MSE(YT, predicted)
                MSE.append(error)
            
            mean_mse = numpy.mean(MSE).item()    
            Z[g, s] = mean_mse
            
            if mean_mse < best_mse:
                best_mse = mean_mse
                best_gamma_index = g
                best_sigma_index = s
    
    
    _gamma = gamma[best_gamma_index]
    _sigma = sigma[best_sigma_index]
    # gamma = math.pow(2, -33)
    # sigma = math.pow(2, 10)
    train_error = []
    test_error = []

    for i in tqdm(range(20)):
        train_data, test_data = train_test_split(data_numpy_array, test_size=0.33)
        X_train = numpy.hstack((train_data[:, : -1], numpy.ones(shape=(len(train_data), 1))))
        Y_train = train_data[:, -1 :]
        
        X_test = numpy.hstack((test_data[:, : -1], numpy.ones(shape=(len(test_data), 1))))
        Y_test = test_data[:, -1 :]
        
        K = create_kernel(X_train, _sigma)
        alpha = ridge_regression(K, Y_train, _gamma)
        predicition = ridge_regression_prediction(X_train, X_train, alpha, _sigma)
        error = calculate_ridge_MSE(Y_train, predicition)
        train_error.append(error)
        
        KT = create_kernel(X_test, _sigma)
        alphaT = ridge_regression(KT, Y_test, _gamma)
        predT = ridge_regression_prediction(X_test, X_test, alphaT, _sigma)
        test_error.append(calculate_ridge_MSE(Y_test, predT))
        
    print('Train MSE: ', (numpy.sum(train_error) / 20), ' Test MSE: ', (numpy.sum(test_error) / 20))
    print(numpy.std(train_error), 'Test: ', numpy.std(test_error))

def run():
    q1_1()
    q1_1_2()
    
    q1_1_2_bc()
    q1_1_2_d()
    
    #q1_1_3()
    
    #q1_2_a()
    #q1_2_c()
    #q1_2_d()
    #q1_3()
    #q1_3_d()

if __name__ == '__main__':
    run()