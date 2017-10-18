import matplotlib.pyplot as plt

x = [1,2,3,4]
y = x
c = ((1.0, 0.0, 0.0), (0.8, 0.1, 0.1), (0.6, 0.2, 0.6), (0.4, 0.3, 0.3))

plt.plot(x,y, color=c, marker='s')
plt.show()
