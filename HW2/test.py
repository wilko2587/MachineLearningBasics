import matplotlib.pyplot as plt
from random import random

def mean(data):
    '''
    returns mean of floats/integers in a list "data"
    '''
    _data = data
    _sum = sum(_data)
    return _sum / len(data)

N = 10000

Pverbal = 0.1
Pgun = 0.01

first_verbal_complaint = []
first_gun_complaint = []
all_verbal_complaint = []
all_gun_complaint = []

t = range(0, 5000, 365)

for n in range(N):
    this_officers_verbal_complaints = []
    this_officers_gun_complaints = []
    for year in t:  # simulate time spent in force, one instance per year
        verbal_complaint = True if random() <= Pverbal else False
        gun_complaint = True if random() <= Pgun else False
        if verbal_complaint:
            this_officers_verbal_complaints.append(year)
            all_verbal_complaint.append(year)
            if len(this_officers_verbal_complaints) == 1:
                first_verbal_complaint.append(year)
        if gun_complaint:
            this_officers_gun_complaints.append(year)
            all_gun_complaint.append(year)
            if len(this_officers_gun_complaints) == 1:
                first_gun_complaint.append(year)

plt.hist(first_verbal_complaint,alpha=0.5,bins=t)
plt.hist(first_gun_complaint,alpha=0.5,bins=t)
plt.show()

print('means: ')
print("average time to first verbal complaint: ",int(mean(first_verbal_complaint)),"days")
print("average time to first gun complaint: ",int(mean(first_gun_complaint)),"days")
