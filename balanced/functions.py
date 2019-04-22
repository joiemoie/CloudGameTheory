import numpy as np

update_rate = .0001

def get_preference(n):
  winner = np.random.uniform(0.025, 0.035)
  losers = [np.random.uniform(-0.01, 0.01) for _ in range(3)]
  if n < 0.25:
    return [winner] + losers
  elif n < 0.5:
    return losers[:1] + [winner] + losers[1:]
  elif n < 0.75:
    return losers[:2] + [winner] + losers[2:]
  else:
    return losers + [winner]

def computeIsValid(user_profits, selected_provider):
  # gets the max of the profits
  user_profits_summed = np.sum(user_profits, 1)
  user_max_profit = np.max(user_profits_summed, 1)
  user_argmax_profit = np.argmax(user_profits_summed, 1)

  # gets all of the users profits for a single provider
  user_profit_by_provider = user_profits_summed[:, selected_provider]

  isValid = (selected_provider == user_argmax_profit) * 1.0
  isValid *= (user_profit_by_provider > 0) * 1.0
  return isValid

def profit_helper(provider_prices, provider_costs, selected_provider, selected_resource, max_prices, quantities, user_preferences):
  num_providers = provider_prices.shape[0]
  num_users = max_prices.shape[0]

  provider_price = provider_prices[selected_provider, selected_resource]
  provider_cost = provider_costs[selected_provider, selected_resource]
  # user profits across all the providers
  user_profits = user_utilities(max_prices, quantities, provider_prices, user_preferences)

  isValid = computeIsValid(user_profits, selected_provider)
  
  return [isValid, provider_price, provider_cost, user_profits]


def lower_profit(provider_prices, provider_costs, selected_provider, selected_resource, max_prices, quantities, user_preferences):

  isValid, provider_price, provider_cost, user_profits = profit_helper(provider_prices, provider_costs, selected_provider, selected_resource, max_prices, quantities, user_preferences)

  num_count = np.sum(isValid)
  if (num_count == len(isValid)):
    return [0.0, 0.0]

  profit = np.sum(isValid) * (provider_price - provider_cost)

  while ((np.sum(isValid) == num_count) and profit >= 0):
    provider_price -= update_rate
    user_profits[:, selected_resource, selected_provider] += update_rate * quantities[:,selected_resource]
      # gets the max of the profits
    isValid = computeIsValid(user_profits, selected_provider)
    profit = np.sum(isValid) * (provider_price - provider_cost)

  if (profit < 0):
    provider_price += update_rate
    user_profits[:, selected_resource, selected_provider] += update_rate * quantities[:,selected_resource]
      # gets the max of the profits
    isValid = computeIsValid(user_profits, selected_provider)
    profit = np.sum(isValid) * (provider_price - provider_cost)
  return [profit, provider_price]

def same_profit(provider_prices,provider_costs, selected_provider, selected_resource, max_prices, quantities, user_preferences):
  isValid, provider_price,provider_cost, user_profits = profit_helper(provider_prices,provider_costs, selected_provider, selected_resource, max_prices, quantities, user_preferences)

  num_count = np.sum(isValid)
  if (num_count == 0):
    return [0.0, 0.0]


  profit = np.sum(isValid) * (provider_price - provider_cost)

  while (np.sum(isValid) == num_count and profit > 0):
    #print(isValid, provider_price)
    provider_price += update_rate

    user_profits[:, selected_resource, selected_provider] -= update_rate * quantities[:,selected_resource]
    isValid = computeIsValid(user_profits, selected_provider)
    profit = np.sum(isValid) * (provider_price - provider_cost)


  provider_price -= update_rate
  profit = np.sum(isValid) * (provider_price - provider_cost)


  return [profit, provider_price]

def higher_profit(provider_prices,provider_costs, selected_provider, selected_resource, max_prices, quantities, user_preferences):
  isValid, provider_price, provider_cost, user_profits = profit_helper(provider_prices, provider_costs, selected_provider, selected_resource, max_prices, quantities, user_preferences)

  num_count = np.sum(isValid)
  if (num_count < 1):
    return [0.0, 0.0]
  old_prov_privce = provider_price

  profit = np.sum(isValid) * (provider_price - provider_cost)

  while (np.sum(isValid) >= num_count - 1 and profit > 0):
    provider_price += update_rate

    user_profits[:, selected_resource, selected_provider] -= update_rate * quantities[:,selected_resource]
    isValid = computeIsValid(user_profits, selected_provider)
    profit = np.sum(isValid) * (provider_price - provider_cost)


  if (provider_price-update_rate != old_prov_privce):
    provider_price-= update_rate

  user_profits[:, selected_resource, selected_provider] += update_rate
  isValid = computeIsValid(user_profits, selected_provider)
  profit = np.sum(isValid) * (provider_price - provider_cost)


  return [profit, provider_price]

def updated_price(provider_prices,provider_costs, selected_provider, selected_resource, max_prices, quantities, user_preferences):

  low_prof, low_price = lower_profit(provider_prices,provider_costs, selected_provider, selected_resource, max_prices, quantities, user_preferences)
  same_prof, same_price = same_profit(provider_prices,provider_costs, selected_provider, selected_resource, max_prices, quantities, user_preferences)

  high_prof, high_price = higher_profit(provider_prices,provider_costs, selected_provider, selected_resource, max_prices, quantities, user_preferences)
  #print(low_prof, same_prof, high_prof)
  #print(low_price, same_price, high_price)

  if (low_prof > same_prof and low_prof > high_prof):
    return low_price
  elif (high_prof >= same_prof):
    return high_price
  else:
    return same_price


def user_utilities(max_prices, quantities, provider_prices, user_preferences):
  num_providers = provider_prices.shape[0]
  num_users = max_prices.shape[0]
  num_resource_types = max_prices.shape[1]

  user_profits = np.tile(np.expand_dims(max_prices,2), [1, 1, num_providers]) +\
                 np.tile(np.expand_dims(user_preferences,1), [1, num_resource_types, 1])
  user_profits = user_profits - np.tile(np.expand_dims(provider_prices.transpose(),0), [num_users, 1, 1])
  user_profits *= np.tile(np.expand_dims(quantities, 2), [1, 1, num_providers])

  return user_profits

# numpy function
def user_assignments(provider_prices, max_prices, quantities, user_preferences):

  assignments = np.full(max_prices.shape, -1)

  utilities = user_utilities(max_prices, quantities, provider_prices, user_preferences)
  utilities = np.sum(utilities, 1)
  for i in range(len(assignments)):
    user_benefits = utilities[i]
    if np.amax(user_benefits) >=0:
      assignments[i] = np.random.choice(np.flatnonzero(user_benefits == user_benefits.max()))
    else:
      assignments[i] = -1
  return assignments
    

# numpy function
def provider_profits(provider_prices, quantities,provider_costs, assignments):
  result = np.zeros(len(provider_prices))
  for i in range(len(provider_prices)):
    result[i] = np.sum((assignments == i) * quantities * (provider_prices[i] - provider_costs[i]))
  return result