def predict_fraud_probability(G, node, trained_results):
    df = create_node_features_dataset(G)
    node_features = df.loc[node].drop('is_fraudster')

    scaled_features = trained_results['scaler'].transform(
        node_features.values.reshape(1, -1)
    )

    fraud_prob = trained_results['model'].predict_proba(scaled_features)[0][1]

    return fraud_prob