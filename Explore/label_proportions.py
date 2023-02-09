import matplotlib.pyplot as plt
from data.get_labelled_data import get_labelled_data

y = get_labelled_data()[1]
counts = {}
labels = ['cardboard','glass','metal','paper','plastic','trash']
for i in y:
    counts[labels[i]] = counts.get(labels[i],0) +1

print(counts)
plt.pie(counts.values(), labels = labels)
plt.show()
