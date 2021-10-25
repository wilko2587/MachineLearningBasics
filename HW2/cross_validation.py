
def generate_CV(data,N):
    '''
    returns a list of N unique sets of (training_data, validation_data) based on the 1D list "data"
    '''
    validation_size = int(len(data)/N)
    cv_sets = []
    for i in range(N):
        _validation = data[i*validation_size:(i+1)*validation_size]
        _training = data[:i*validation_size] + data[(i+1)*validation_size:]
        cv_sets.append( (_validation, _training))
    return cv_sets

