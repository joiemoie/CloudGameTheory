import numpy as np
import functions
import matplotlib.pyplot as plt
import matplotlib.cm as cm

num_providers = 8
num_users = 1000
num_resource_types = 2

random_prices = np.random.random((num_users, num_resource_types)) * 5 + 10

final_prices = np.random.random((num_providers, num_resource_types)) * 10
random_preferences = np.random.random((num_users, num_providers)) * 5
random_preferences[:,0] += np.random.random((num_users)) * 5

def price_to_quantity(prices):
  slope = (2.0 - 5.0) / (15.0 - 10.0)

  return prices * slope + 16.2

quantities = price_to_quantity(random_prices)
random_prices[:, 0] += 5

lr = .1

results = np.zeros((1000, num_resource_types))
#plt.axis([0, 1000, 0, 10])
colors = cm.rainbow(np.linspace(0, 1, 8))
fig=plt.figure(figsize=(16, 16))
columns = 2
rows = 1

for step in range(1000):
  permutation = np.random.permutation(num_providers)
  for perm in range(num_providers):

    # random user makes the first move
    i = permutation[perm]

    for k in range(num_resource_types):

      grad = functions.provider_gradients(final_prices , i, k, random_prices, quantities, random_preferences)

      final_prices[i, k] = grad

    assignments = functions.user_assignments(final_prices, random_prices, quantities, random_preferences)

    if (step % 10 == 0):
      prov_profits = functions.provider_profits(final_prices, assignments)
      print(i, prov_profits, final_prices, np.mean(random_preferences, axis=0))
      results[step] = final_prices[i]
      print(np.sum(final_prices[i]))
      fig.add_subplot(rows, columns, 1)
      plt.scatter(np.full((num_resource_types), step), results[step], c=colors[i])
      fig.add_subplot(rows, columns, 2)
      plt.scatter(step, prov_profits[i], c=colors[i])

      plt.draw()

      plt.pause(0.05)

plt.show()