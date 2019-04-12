import tensorflow as tf
import numpy as np
import functions

num_providers = 4
num_users = 100

random_prices = np.random.random((num_users)) * 5 + 10
#random_prices = np.full((num_users), 12.5)
final_prices = np.full(num_providers, 10.0)
random_preferences = np.random.random((num_users, num_providers)) * 10
random_preferences[:,0] += np.random.random((num_users)) * 25

lr = .1

for step in range(1000):
  permutation = np.random.permutation(num_providers)
  for perm in range(num_providers):

    # random user makes the first move
    i = permutation[perm]
    temp_final_prices1 = np.copy(final_prices)
    temp_final_prices2 = np.copy(temp_final_prices1)

    grad = functions.provider_gradients(temp_final_prices1, i, random_prices, random_preferences)

    ## user theorizes what happens if he makes the first move
    temp_final_prices1[i] = grad

    ## user theorizes what happens if all opponents react to his move aggressively
    for perm2 in range(num_providers):
      if (perm2 != perm):
        j = permutation[perm2]
        grad2 = functions.provider_gradients(temp_final_prices1, j, random_prices, random_preferences)

        temp_final_prices2[j] = grad2

    grad = functions.provider_gradients(temp_final_prices2, i, random_prices, random_preferences)

    final_prices[i] = grad

    assignments = functions.user_assignments(final_prices, random_prices, random_preferences)

    try:
      print(i, functions.provider_profits(final_prices, assignments), final_prices, np.mean(random_preferences, axis=0))
      
    except:
      pass
