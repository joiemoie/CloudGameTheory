import numpy as np
import functions
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Set these parameters
num_providers = 8
num_users = 1000
num_resource_types = 1

# Set the max prices for the users
random_prices = np.random.random((num_users, num_resource_types)) * 2 + 10
random_prices[:, 0] += 5


# Set the initial starting prices for the companies
final_prices = np.random.random((num_providers, num_resource_types)) * 2 + 10

# Set the user preferences
random_preferences = np.random.random((num_users, num_providers)) * .5
random_preferences[:,0] += np.random.random((num_users)) * .3

# function which converts price to quantity based on the demand function
def price_to_quantity(prices):
  slope = (2.0 - 5.0) / (15.0 - 10.0)

  return prices * slope + 16.2

# Sets the quantities demanded
quantities = price_to_quantity(random_prices)

# stores the final prices for each iteration
results = np.zeros((1000, num_resource_types))

# information for setting the numpy plot
colors = cm.rainbow(np.linspace(0, 1, 8))
fig=plt.figure(figsize=(16, 16))
columns = 2
rows = 1

#iterates through 1000 iterations.
for step in range(1000):
  permutation = np.random.permutation(num_providers)
  for perm in range(num_providers):

    # random user makes the first move
    i = permutation[perm]

    #iterates through each resource type
    for k in range(num_resource_types):

      new_price = functions.updated_price(final_prices , i, k, random_prices, quantities, random_preferences)

      final_prices[i, k] = new_price

    # determines the final assignments
    assignments = functions.user_assignments(final_prices, random_prices, quantities, random_preferences)

    # prints to the graph every 10 iterations
    if (step % 10 == 0):

      # computes the provider profits
      prov_profits = functions.provider_profits(final_prices, assignments)

      # prints some useful intormation
      print(i, prov_profits, final_prices, np.mean(random_preferences, axis=0))

      # stores the results
      results[step] = final_prices[i]

      # plots the data
      fig.add_subplot(rows, columns, 1)
      plt.scatter(np.full((num_resource_types), step), results[step], c=colors[i])
      fig.add_subplot(rows, columns, 2)
      plt.scatter(step, prov_profits[i], c=colors[i])

      plt.draw()

      plt.pause(0.05)

plt.show()