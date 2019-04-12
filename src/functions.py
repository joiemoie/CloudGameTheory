import tensorflow as tf
import numpy as np

def sigmoid(x, derivative=False):
  return x*(1-x) if derivative else 1/(1+np.exp(-x))

def lower_profit(provider_prices, selected_provider, max_prices, user_preferences):
  num_providers = provider_prices.shape[0]
  num_users = max_prices.shape[0]

  provider_price = provider_prices[selected_provider]
  # user profits across all the providers
  user_profits = np.tile(np.expand_dims(max_prices,1), [1, num_providers]) +\
                 user_preferences

  user_profits = user_profits - provider_prices

  # gets the max of the profits
  user_max_profit = np.max(user_profits, 1)
  user_argmax_profit = np.argmax(user_profits, 1)

  # gets all of the users profits for a single provider
  user_profit_by_provider = user_profits[:, selected_provider]

  isValid = (selected_provider == user_argmax_profit) * 1.0
  isValid *= (user_profit_by_provider > 0) * 1.0
  
  num_count = np.sum(isValid)
  if (num_count == len(isValid)):
    
    return [0.0, 0.0]
  while (np.sum(isValid) == num_count and provider_price > 0):
    provider_price -= .01
    user_profits[:, selected_provider] += .01
      # gets the max of the profits
    user_max_profit = np.max(user_profits, 1)

    # gets all of the users profits for a single provider
    user_profit_by_provider = user_profits[:, selected_provider]

    isValid = (user_max_profit == user_profit_by_provider) * 1.0
    isValid *= (user_profit_by_provider > 0) * 1.0

  return [np.sum(isValid) * provider_price, provider_price]

def same_profit(provider_prices, selected_provider, max_prices, user_preferences):
  num_providers = provider_prices.shape[0]
  num_users = max_prices.shape[0]

  provider_price = provider_prices[selected_provider]

  # user profits across all the providers
  user_profits = np.tile(np.expand_dims(max_prices,1), [1, num_providers]) +\
                 user_preferences

  user_profits = user_profits - provider_prices

  # gets the max of the profits
  user_max_profit = np.max(user_profits, 1)
  user_argmax_profit = np.argmax(user_profits, 1)

  # gets all of the users profits for a single provider
  user_profit_by_provider = user_profits[:, selected_provider]

  isValid = (selected_provider == user_argmax_profit) * 1.0
  isValid *= (user_profit_by_provider > 0) * 1.0
  
  num_count = np.sum(isValid)
  if (num_count == 0):
    return [0.0, 0.0]

  while (np.sum(isValid) == num_count):
    #print(isValid, provider_price)
    provider_price += .01
    user_profits[:, selected_provider] -= .01
      # gets the max of the profits
    user_max_profit = np.max(user_profits, 1)

    # gets all of the users profits for a single provider
    user_profit_by_provider = user_profits[:, selected_provider]

    isValid = (user_max_profit == user_profit_by_provider) * 1.0
    isValid *= (user_profit_by_provider > 0) * 1.0
  provider_price -= .01

  return [num_count * provider_price, provider_price]

def higher_profit(provider_prices, selected_provider, max_prices, user_preferences):
  num_providers = provider_prices.shape[0]
  num_users = max_prices.shape[0]

  provider_price = provider_prices[selected_provider]

  # user profits across all the providers
  user_profits = np.tile(np.expand_dims(max_prices,1), [1, num_providers]) +\
                 user_preferences

  user_profits = user_profits - provider_prices

  # gets the max of the profits
  user_max_profit = np.max(user_profits, 1)
  user_argmax_profit = np.argmax(user_profits, 1)

  # gets all of the users profits for a single provider
  user_profit_by_provider = user_profits[:, selected_provider]

  isValid = (selected_provider == user_argmax_profit) * 1.0
  isValid *= (user_profit_by_provider > 0) * 1.0
  
  num_count = np.sum(isValid)
  if (num_count < 1):
    return [0.0, 0.0]
  old_prov_privce = provider_price
  while (np.sum(isValid) >= num_count - 1 and np.sum(isValid) > 0):
    provider_price += .01
    user_profits[:, selected_provider] -= .01
      # gets the max of the profits
    user_max_profit = np.max(user_profits, 1)

    # gets all of the users profits for a single provider
    user_profit_by_provider = user_profits[:, selected_provider]

    isValid = (user_max_profit == user_profit_by_provider) * 1.0
    isValid *= (user_profit_by_provider > 0) * 1.0

  if (provider_price-.01 != old_prov_privce):
    provider_price-= .01
  user_profits[:, selected_provider] += .01
    # gets the max of the profits
  user_max_profit = np.max(user_profits, 1)

  # gets all of the users profits for a single provider
  user_profit_by_provider = user_profits[:, selected_provider]

  isValid = (user_max_profit == user_profit_by_provider) * 1.0
  isValid *= (user_profit_by_provider > 0) * 1.0
  return [(np.sum(isValid)) * provider_price, provider_price]

def provider_gradients(provider_prices, selected_provider, max_prices, user_preferences):

  low_prof, low_price = lower_profit(provider_prices, selected_provider, max_prices, user_preferences)
  same_prof, same_price = same_profit(provider_prices, selected_provider, max_prices, user_preferences)

  high_prof, high_price = higher_profit(provider_prices, selected_provider, max_prices, user_preferences)
  #print(low_prof, same_prof, high_prof)
  #print(low_price, same_price, high_price)

  if (low_prof > same_prof and low_prof > high_prof):
    return low_price
  elif (high_prof >= same_prof):
    return high_price
  else:
    return same_price


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



