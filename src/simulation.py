import tensorflow as tf
import numpy as np
import functions

num_providers = 4
num_users = 100

# tf Graph input

selected_provider = tf.placeholder(tf.int32, [None])
provider_prices = tf.placeholder(tf.float32, [None, num_providers])
max_prices = tf.placeholder(tf.float32, [None, num_users])
user_preferences = tf.placeholder(tf.float32, [None, num_users, num_providers])

provider_grads = functions.provider_gradients(provider_prices, selected_provider, max_prices, user_preferences)

# Initialize the variables (i.e.  assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

  # Run the initializer
  sess.run(init)

  random_prices = np.random.random((num_users)) * 5 + 10
  final_prices = np.full(num_providers, 100.0)
  random_preferences = np.random.random((num_users, num_providers)) * .1 + 10
  lr = .1

  for step in range(500):
    permutation = np.random.permutation(num_providers)
    for perm in range(num_providers):

      # random user makes the first move
      i = permutation[perm]
      temp_final_prices1 = np.copy(final_prices)
      temp_final_prices2 = np.copy(temp_final_prices1)

      grad= sess.run(provider_grads, feed_dict = {max_prices: [random_prices], 
                                      provider_prices: [temp_final_prices1], 
                                      selected_provider: [i],
                                      user_preferences: [random_preferences]})

      # user theorizes what happens if he makes the first move
      temp_final_prices1[i] = functions.gradient_update(temp_final_prices1, random_prices, random_preferences, i, grad, lr)

      # user theorizes what happens if all opponents react to his move aggressively
      for perm2 in range(num_providers):
        if (perm2 != perm):
          j = permutation[perm2]
          grad2= sess.run(provider_grads, feed_dict = {max_prices: [random_prices], 
                                      provider_prices: [temp_final_prices1], 
                                      selected_provider: [j],
                                      user_preferences: [random_preferences]})
          temp_final_prices2[j] = functions.gradient_update(temp_final_prices1, random_prices, random_preferences, j, grad2, lr)

      # user reacts based on the aggressive actions of the actors
      grad = sess.run(provider_grads, feed_dict = {max_prices: [random_prices], 
                                      provider_prices: [temp_final_prices2], 
                                      selected_provider: [i],
                                      user_preferences: [random_preferences]})
      final_prices[i] = functions.gradient_update(temp_final_prices2, random_prices, random_preferences, i, grad, lr)

    assignments = functions.user_assignments(final_prices, random_prices, random_preferences)

    print(functions.provider_profits(final_prices, assignments), final_prices, np.mean(random_preferences, axis=0))

  #isIncreasing = True
  #curr_max = np.sum((final_price < random_prices) * final_price)
  #while (isIncreasing):
    #temp_price = final_price + .01
    #curr = np.sum((temp_price < random_prices) * temp_price)
    #if (curr < curr_max):
      #break
    #final_price = temp_price
