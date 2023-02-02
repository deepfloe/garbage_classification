import matplotlib.pyplot as plt
from data.get_labelled_data import get_labelled_data

labels = get_labelled_data()[1]
plt.hist(labels)
plt.show()
