import pandas as pd

# Create the DataFrame
df = pd.DataFrame({
    'income': ['very high', 'high', 'medium', 'high', 'very high', 'medium', 'high', 'medium', 'high', 'low'],
    'credit': ['excellent', 'good', 'excellent', 'good', 'good', 'excellent', 'bad', 'bad', 'bad', 'bad'],
    'decision': ['authorize', 'authorize', 'authorize', 'authorize', 'authorize', 'authorize', 'request id', 'request id', 'reject', 'call police']
})

# Calculate prior probabilities
prior_probs = df['decision'].value_counts(normalize=True)

# Conditional probabilities given the tuple ('medium', 'good')
tuple_to_add = ('medium', 'good')
likelihoods = {
    decision: (df[(df['decision'] == decision) & (df['income'] == tuple_to_add[0])].shape[0] / df['decision'].value_counts()[decision]) *
              (df[(df['decision'] == decision) & (df['credit'] == tuple_to_add[1])].shape[0] / df['decision'].value_counts()[decision])
    for decision in df['decision'].unique()
}

# Posterior probabilities
posteriors = {decision: prior_probs[decision] * likelihood for decision, likelihood in likelihoods.items()}

# Output classification
classified_as = max(posteriors, key=posteriors.get)

print(prior_probs)
print("Posteriors: ",posteriors)
print("Likelihoods: ",likelihoods)
print(f"Tuple classified into: {classified_as}")
