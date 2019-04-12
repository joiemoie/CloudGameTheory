import tensorflow as tf
import numpy as np

def provider_gradients(provider_prices, selected_provider, max_prices, user_preferences):

  num_providers = provider_prices.shape[1]
  num_users = max_prices.shape[1]

  provider_price = provider_prices[:, selected_provider[0]]
  # user profits across all the providers
  user_profits = tf.tile(tf.expand_dims(max_prices,2), [1, 1, num_providers]) +\
                 user_preferences
  # max_profit by provider
  max_profit_by_provider = user_profits[:, :, selected_provider[0]]

  user_profits = user_profits - provider_prices

  # gets the max of the profits
  user_max_profit = tf.reduce_max(user_profits, 2)

  # gets all of the users profits for a single provider
  user_profit_by_provider = user_profits[:, :, tf.to_int32(selected_provider[0])]

  # checks if the provider profits are greater than the highest profit from other companies
  #isValid = tf.to_float(tf.math.equal(user_profit_by_provider, user_max_profit))
  isValid = tf.to_float(tf.math.sigmoid(user_max_profit - user_profit_by_provider))


  # checks if the user will benefit from buying
  #isValid = isValid * tf.to_float(tf.math.greater(user_profit_by_provider, 0))
  isValid = isValid * tf.to_float(tf.sigmoid(user_profit_by_provider))


  isNotValid = (isValid * -1 + 1)

  profit = tf.reduce_sum(provider_price * isValid)
  loss = tf.reduce_sum((provider_price - (user_max_profit - user_profit_by_provider)) * isNotValid)
  loss -= tf.reduce_sum(((user_max_profit - user_profit_by_provider)) * isValid)

  potential = tf.reduce_sum((max_profit_by_provider - provider_price) * isValid)

  #gradients = tf.squeeze(tf.gradients(profit - loss - potential, provider_price))
  gradients = tf.squeeze(tf.gradients(profit -loss - potential, provider_price))
  temp = [tf.gradients(profit, provider_price)[0], tf.gradients(loss, provider_price)[0], tf.gradients(potential, provider_price)]
  #temp = isNotValid

  return [gradients, temp]

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
  updated_prices = np.copy(provider_prices)
  updated_prices[provider_index] += gradients * lr


  if updated_prices[provider_index] < 0:
    return provider_prices[provider_index] / 2

  return updated_prices[provider_index]



