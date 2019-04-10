import tensorflow as tf
import numpy as np

def provider_gradients(provider_prices, selected_provider, max_prices, user_preferences):

  num_providers = provider_prices.shape[1]
  num_users = max_prices.shape[1]

  # user profits across all the providers
  user_profits = tf.tile(tf.expand_dims(max_prices,2), [1, 1, num_providers]) -\
                 tf.tile(tf.expand_dims(provider_prices,1), [1 , num_users, 1]) +\
                 user_preferences

    # user profits across all the providers
  user_monetary_profits = tf.tile(tf.expand_dims(max_prices,2), [1, 1, num_providers]) -\
                 tf.tile(tf.expand_dims(provider_prices,1), [1 , num_users, 1])

  # gets the max of the profits
  user_max_profit = tf.reduce_max(user_profits, 2)

  # gets the price of a single provider
  provider_price = tf.gather(provider_prices, selected_provider, axis=1)

  # gets all of the users profits for a single provider
  user_profit_by_provider = tf.squeeze(tf.gather(user_profits, selected_provider, axis=2), axis=2)

  # checks if the provider profits are greater than the highest profit from other companies
  isValid = tf.to_float(tf.math.equal(user_profit_by_provider, user_max_profit))

  # checks if the user will benefit from buying
  isValid = isValid * tf.to_float(tf.math.greater(user_profit_by_provider, 0))

  isNotValid = (isValid * -1 + 1)

  highest_user_preferences = tf.reduce_max(user_preferences, axis=2)

  # ignores users who do not even have a preference for me and I cannot retrieve them anymore
  ignoreExtras= tf.to_float(tf.math.greater(highest_user_preferences, provider_price)) *\
    tf.to_float(tf.math.not_equal(selected_provider,tf.cast(tf.math.argmax(user_profits, axis=2), tf.int32)))

  isNotValid *= (ignoreExtras * -1 + 1)
  profit = provider_price * tf.reduce_mean(isValid)
  profit = tf.reduce_sum(profit, axis=1)
  loss = provider_price * max_prices * tf.reduce_mean(isNotValid)
  loss = tf.reduce_sum(loss, axis=1)
  
  gradients = tf.squeeze(tf.gradients(profit - loss, provider_price))
  return gradients

def user_utilities(max_prices, provider_prices, user_preferences):
  utilities = np.zeros((len(max_prices), len(provider_prices)))
  for i in range(len(max_prices)):
    utilities[i] = max_prices[i] - provider_prices + user_preferences[i]
  return utilities
# numpy function
def user_assignments(provider_prices, max_prices, user_preferences):

  assignments = np.full(max_prices.shape, -1)

  utilities = user_utilities(max_prices, provider_prices, user_preferences)
  for i in range(len(assignments)):
    user_benefits = utilities[i]
    if np.amax(user_benefits) >=0:
      assignments[i] = np.random.choice(np.flatnonzero(user_benefits == user_benefits.max()))
    else:
      assignments[i] = -1
  return assignments
    

# numpy function
def provider_profits(provider_prices, assignments):
  result = np.zeros(provider_prices.shape)
  for i in range(len(provider_prices)):
    result[i] = np.sum(assignments == i) * provider_prices[i]
  return result

def gradient_update(provider_prices, max_prices, user_preferences, provider_index, gradients, lr):
  assigns1 = user_assignments(provider_prices, max_prices, user_preferences)
  curr_profits = provider_profits(provider_prices, assigns1)[provider_index]
  updated_prices = np.copy(provider_prices)
  updated_prices[provider_index] += gradients * lr
  assigns2 = user_assignments(updated_prices, max_prices, user_preferences)
  updated_profits = provider_profits(updated_prices, assigns2)[provider_index]

  if updated_prices[provider_index] < 0:
    return provider_prices[provider_index] / 2

  if (gradients > 0 and (curr_profits >= updated_profits)):
    return provider_prices[provider_index]
  else:
    return updated_prices[provider_index]


