import parse

def get_numerical_split(data, var, target='Class'):

    split0 = []
    for example in data:
        print('example: ',example)
        print('example[var]: ',example[var])
        print('example[target]: ', example[target])
        if int(example[var]) == 0:
            print('    IF STATEMENT ENTERED!')
            split0.append(example[target])
        else:
            print("    IF STATEMENT AVOIDED")
            pass
        print('new split0: ',split0)
        input('Press rtn key to example...')
        print('')

    splits = [split0]
    return splits

file = "tennis.data"
data = parse.parse(file)
split = get_numerical_split(data,"Wind",target="Class")