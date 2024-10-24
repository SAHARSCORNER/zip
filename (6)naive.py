import pandas as pd
import numpy as np

# Create the DataFrame
df = pd.DataFrame({
    'income': ['very high', 'high', 'medium', 'high', 'very high', 'medium', 'high', 'medium', 'high', 'low'],
    'credit': ['excellent', 'good', 'excellent', 'good', 'good', 'excellent', 'bad', 'bad', 'bad', 'bad'],
    'decision': ['authorize', 'authorize', 'authorize', 'authorize', 'authorize', 'authorize', 'request id', 'request id', 'reject', 'call police']
})

# Calculate the prior probabilities
prior_counts = df['decision'].value_counts()
total_count = len(df)

# Prior probabilities
p_of_authorize = prior_counts['authorize'] / total_count
p_of_request_id = prior_counts['request id'] / total_count
p_of_reject = prior_counts['reject'] / total_count
p_of_call_police = prior_counts['call police'] / total_count

# Output prior probabilities
print("c1 probability:", p_of_authorize)
print("c2 probability:", p_of_request_id)
print("c3 probability:", p_of_reject)
print("c4 probability:", p_of_call_police)

# Tuple to classify
tuple_to_add = ('medium', 'good')

# Conditional probabilities
p_income_medium_given_authorize = df[(df['decision'] == 'authorize') & (df['income'] == 'medium')].shape[0] / prior_counts['authorize']
p_credit_good_given_authorize = df[(df['decision'] == 'authorize') & (df['credit'] == 'good')].shape[0] / prior_counts['authorize']

p_income_medium_given_request_id = df[(df['decision'] == 'request id') & (df['income'] == 'medium')].shape[0] / prior_counts['request id']
p_credit_good_given_request_id = df[(df['decision'] == 'request id') & (df['credit'] == 'good')].shape[0] / prior_counts['request id']

p_income_medium_given_reject = df[(df['decision'] == 'reject') & (df['income'] == 'medium')].shape[0] / prior_counts['reject']
p_credit_good_given_reject = df[(df['decision'] == 'reject') & (df['credit'] == 'good')].shape[0] / prior_counts['reject']

p_income_medium_given_call_police = df[(df['decision'] == 'call police') & (df['income'] == 'medium')].shape[0] / prior_counts['call police']
p_credit_good_given_call_police = df[(df['decision'] == 'call police') & (df['credit'] == 'good')].shape[0] / prior_counts['call police']

# Total likelihoods for the tuple
p_of_x_on_c1 = p_income_medium_given_authorize * p_credit_good_given_authorize
p_of_x_on_c2 = p_income_medium_given_request_id * p_credit_good_given_request_id
p_of_x_on_c3 = p_income_medium_given_reject * p_credit_good_given_reject
p_of_x_on_c4 = p_income_medium_given_call_police * p_credit_good_given_call_police

# Posterior probabilities
p_of_c1_on_x = p_of_authorize * p_of_x_on_c1
p_of_c2_on_x = p_of_request_id * p_of_x_on_c2
p_of_c3_on_x = p_of_reject * p_of_x_on_c3
p_of_c4_on_x = p_of_call_police * p_of_x_on_c4

posterior_probabilities = [p_of_c1_on_x, p_of_c2_on_x, p_of_c3_on_x, p_of_c4_on_x]

# Finding the maximum posterior probability
max_prob = max(posterior_probabilities)
print(f"Maximum posterior probability is {max_prob}")

# Classifying the tuple
for i, probability in enumerate(posterior_probabilities, start=1):
    if max_prob == probability:
        print(f"Tuple classified into c{i}")
        break
