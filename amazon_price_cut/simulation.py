import numpy as np
import functions
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches

# Set these parameters
num_providers = 4
num_users = 1000
num_resource_types = 1

# Set the max prices for the users
random_prices = np.random.random((num_users, num_resource_types)) * (0.14 - 0.1) + 0.1


# Set the initial starting prices for the companies
final_prices = np.random.random((num_providers, num_resource_types)) * (0.14 - 0.1) + 0.1
final_prices[0] = .14
final_prices[1] = .13
final_prices[2] = .12
final_prices[3] = .11
final_costs = np.full((num_providers, num_resource_types), 0.08)
# amazon cut
#final_costs[0, :] -= 0.004
# google cut
final_costs[2, :] -= 0.004

# Set the user preferences
random_preferences = np.array([functions.get_preference(np.random.random()) for _ in range(num_users)])
#random_preferences[:,0] += np.random.random((num_users)) * 3

# function which converts price to quantity based on the demand function

# Sets the quantities demanded
quantities = np.array([[np.random.randint(10, 13) for _ in range(num_resource_types)] for _ in range(num_users)])

# stores the final prices for each iteration
results = np.zeros((1000, num_resource_types))

# information for setting the numpy plot
colors = cm.rainbow(np.linspace(0, 1, num_providers))
fig=plt.figure(figsize=(16, 16))
columns = 2
rows = 1
labels = ["AWS", "Azure", "Google", "IBM"]
handles = []
for i in range(num_providers):
    handles.append(mpatches.Patch(color=colors[i], label=labels[i]))

#iterates through 1000 iterations.
for step in range(1000):
  permutation = np.random.permutation(num_providers)
  for perm in range(num_providers):

    # random user makes the first move
    i = permutation[perm]

    #iterates through each resource type
    for k in range(num_resource_types):

      new_price = functions.updated_price(final_prices,final_costs , i, k, random_prices, quantities, random_preferences)

      final_prices[i, k] = new_price

    # determines the final assignments
    assignments = functions.user_assignments(final_prices, random_prices, quantities, random_preferences)

    # prints to the graph every 10 iterations
    if (step % 10 == 0):

      # computes the provider profits
      prov_profits = functions.provider_profits(final_prices, quantities,final_costs, assignments)

      # prints some useful intormation
      print(i, prov_profits, final_prices, np.mean(random_preferences, axis=0))

      # stores the results
      results[step] = final_prices[i]

      # plots the data
      fig.add_subplot(rows, columns, 1)
      plt.scatter(np.full((num_resource_types), step), results[step], c=colors[i])
      plt.ylim(0.08, 0.15)
      plt.xlabel("Iterations")
      plt.ylabel("Final prices")
      plt.legend(handles=handles)
      
      fig.add_subplot(rows, columns, 2)
      plt.scatter(step, prov_profits[i], c=colors[i])
      plt.xlabel("Iterations")
      plt.ylabel("Provider profits")
      plt.legend(handles=handles)

      plt.draw()

      plt.pause(0.05)

plt.show()
