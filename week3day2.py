import matplotlib.pyplot as plt
import numpy as np

# Create an array of t values (you can adjust the range as needed)
t = np.linspace(-5, 5, 100)  # From -5 to 5, with 100 points
y = 2 - t

# Plot the function
plt.plot(t, y)

# Add labels and title
plt.xlabel('t')
plt.ylabel('y')
plt.title('Plot of y = 2 - t')

# Add grid lines for better readability
plt.grid(True)

# Add axes lines
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

# Show the plot
plt.show()