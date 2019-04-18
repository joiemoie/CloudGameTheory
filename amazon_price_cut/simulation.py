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
final_costs = np.full((num_providers, num_resource_types), 0.1)
# amazon cut
#final_costs[0, :] -= 0.005
# google cut
final_costs[2, :] -= 0.005

# Set the user preferences
random_preferences = np.random.random((num_users, num_providers)) * 0.02 - 0.01
random_user_value = np.random.random((num_users))
for i in range(num_users):
    if random_user_value[i] < 0.5:
        random_preferences[i, 0] = np.random.random() * (0.035 - 0.025) + 0.025
    elif random_user_value[i] < 0.79:
        random_preferences[i, 1] = np.random.random() * (0.035 - 0.025) + 0.025
    elif random_user_value[i] < 0.92:
        random_preferences[i, 2] = np.random.random() * (0.035 - 0.025) + 0.025
    else:
        random_preferences[i, 3] = np.random.random() * (0.035 - 0.025) + 0.025
#random_preferences[:,0] += np.random.random((num_users)) * 3

# function which converts price to quantity based on the demand function

# Sets the quantities demanded
quantities = np.random.randint(low = 10, high = 13, size = (num_users, num_resource_types))

# stores the final prices for each iteration
results = np.zeros((1000, num_resource_types))

# information for setting the numpy plot
colors = cm.rainbow(np.linspace(0, 1, 4))
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

fig2 = plt.figure()
prov_market_share = prov_profits / sum(prov_profits)
plt.pie(prov_market_share, labels=labels, colors=colors, autopct='%1.1f%%')
plt.axis('equal')

plt.show()
