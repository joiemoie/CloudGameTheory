import numpy as np
import functions
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import csv

bufs = []

for index in range(10):
  # Set these parameters
  num_providers = 4
  num_users = 2000
  num_resource_types = 1

  # Set the max prices for the users
  random_prices = np.random.random((num_users, num_resource_types)) * 0.040 + 0.100

  # Set the initial starting prices for the companies
  final_prices = np.random.random((num_providers, num_resource_types)) * 0.040 + 0.100
  final_prices[0] = .14
  final_prices[1] = .13
  final_prices[2] = .12
  final_prices[3] = .11
  final_costs = np.full((num_providers, num_resource_types), 0.08)

  # Set the user preferences
  random_preferences = np.array([functions.get_preference(np.random.random()) for _ in range(num_users)])
  random_preferences[int(num_users/2):, :] = np.random.random() * .01 - .01

  # function which converts price to quantity based on the demand function

  # Sets the quantities demanded
  quantities = np.array([[np.random.randint(10, 13) for _ in range(num_resource_types)] for _ in range(num_users)])

  # stores the final prices for each iteration
  results = np.zeros((num_users, num_resource_types))

  # information for setting the numpy plot
  colors = cm.rainbow(np.linspace(0, 1, num_providers))
  fig=plt.figure(figsize=(8, 4))
  columns = 2
  rows = 1
  labels = ["AWS", "Azure", "Google", "IBM"]
  handles = []
  for i in range(num_providers):
      handles.append(mpatches.Patch(color=colors[i], label=labels[i]))

  # iterates through 1000 iterations.
  for step in range(1000):
    permutation = np.random.permutation(num_providers)
    for perm in range(num_providers):

      # random user makes the first move
      i = permutation[perm]

      #iterates through each resource type
      for k in range(num_resource_types):

        new_price = functions.updated_price(final_prices, final_costs, i, k, random_prices, quantities, random_preferences)

        final_prices[i, k] = new_price

      # determines the final assignments
      assignments = functions.user_assignments(final_prices, random_prices, quantities, random_preferences)

      # prints to the graph every 10 iterations
      if (step % 10 == 0):

        # computes the provider profits
        prov_profits = functions.provider_profits(final_prices, quantities, final_costs, assignments)

        # prints some useful intormation
        print(i, prov_profits, final_prices, np.mean(random_preferences, axis=0))

        # stores the results
        results[step] = final_prices[i]

        # plots the data
        fig.add_subplot(rows, columns, 1)
        plt.ylim((0.080, 0.150))
        plt.scatter(np.full((num_resource_types), step), results[step], c=colors[i])
        plt.xlabel("Iterations")
        plt.ylabel("Final prices")
        plt.legend(handles=handles, loc="best")

        fig.add_subplot(rows, columns, 2)
        plt.scatter(step, prov_profits[i], c=colors[i])
        plt.xlabel("Iterations")
        plt.ylabel("Provider profits")
        plt.legend(handles=handles, loc="best")

        plt.draw()

        plt.pause(0.01)
  plt.show()
        #fig1_path = "/Users/apple/Desktop/18859/project/simulation/real_preferences/real16_" + str(index) + "_1.png"
  #plt.savefig(fig1_path)
  buf = [index] + [profit for profit in prov_profits] + [price[0] for price in final_prices]
  bufs.append(buf)


  ## Market share results (2017)
  fig = plt.figure(figsize=(12, 4))

  # predict market share using model
  final_profits = prov_profits
  fig.add_subplot(1, 3, 1)
  plt.title("Prediction on model")
  plt.pie(final_profits, labels=labels, autopct = '%3.2f%%', colors=colors)

  # predict market share using real prices
  real_prices = np.array([[0.133], [0.100], [0.136], [0.137]]) # 2017 w SSD
  assignments = functions.user_assignments(real_prices, random_prices, quantities, random_preferences)
  final_profits = functions.provider_profits(real_prices, quantities, final_costs, assignments)
  fig.add_subplot(1, 3, 2)
  plt.title("Prediction on real prices")
  plt.pie(final_profits, labels=labels, autopct = '%3.2f%%', colors=colors)

  # real market share
  real_market_share = [0.6, 0.2, 0.0706, 0.1294]
  fig.add_subplot(1, 3, 3)
  plt.title("Real market share in 2017")
  plt.pie(real_market_share, labels=labels, autopct = '%3.2f%%', colors=colors)
  fig2_path = "/Users/apple/Desktop/18859/project/simulation/real_preferences/real16_" + str(index) + "_2.png"
  plt.savefig(fig2_path)

# write data into .csv file
file_path = "/Users/apple/Desktop/18859/project/simulation/real_preferences/test.csv"
with open(file_path,"w") as csvfile: 
  writer = csv.writer(csvfile)
  # write column names
  writer.writerow(["No.", "AWS", "Azure", "Google", "IBM", "AWS", "Azure", "Google", "IBM"])
  # write rows
  writer.writerows(bufs)
