import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score
import matplotlib.pyplot as plt

rmse_list = []
mae_list = []
precision_list = []
recall_list = []

sample_size = 20000
num_sample = 10

df0 = pd.read_csv('RAW_interactions.csv')
df2 = pd.read_csv('RAW_recipes.csv')


def match_recipes(user_id, recommendations):
    print("Recommend for", user_id)
    nameid = recommendations.index.tolist()
    for k in nameid:
        print(df2.loc[df2['id'] == k, 'name'].tolist())
    print("end")


def items_based(user_id, df, num_recommendations=5):
    ratings_matrix = df.pivot_table(index='user_id', columns='recipe_id', values='rating', fill_value=0)

    # Calculate similarity (use cosine)
    item_similarity = cosine_similarity(ratings_matrix.T)
    item_similarity_df = pd.DataFrame(item_similarity, index=ratings_matrix.columns, columns=ratings_matrix.columns)

    # Check if user ID is in the dataframe
    if df[df['user_id'] == user_id].shape[0] > 0:
        # Rating dataset for user
        user_ratings = ratings_matrix.loc[user_id]

        # Save recommendations
        similar_items = pd.Series(dtype=float)

        for item, rating in user_ratings.items():
            if rating > 0:
                # Find similarity value
                similar_items = pd.concat([similar_items, item_similarity_df[item] * rating])

        # Sorting
        similar_items = similar_items.sort_values(ascending=False)
        # Exclude items that have already been rated
        similar_items = similar_items[~similar_items.index.isin(user_ratings[user_ratings > 0].index)]

        return similar_items.head(num_recommendations)
    else:
        similar_items = pd.Series(dtype=float)
        return similar_items.head(num_recommendations)


def evaluate_recommendations(test_data, predicted_ratings):
    # Match indices between test data and predicted ratings
    matched_indices = test_data['recipe_id'][test_data['recipe_id'].isin(predicted_ratings.index)]

    # Filter test data and predicted ratings for matched indices
    y_true = test_data.set_index('recipe_id').loc[matched_indices, 'rating']
    y_pred = predicted_ratings.loc[matched_indices]

    # Check if matched indices are empty
    if y_true.empty or y_pred.empty:
        print("No matching items between test data and recommendations. Evaluation skipped.")
        return

    # Ensure same order for evaluation
    y_true = y_true.sort_index()
    y_pred = y_pred.sort_index()

    # RMSE and MAE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Store metrics for visualization
    rmse_list.append(rmse)
    mae_list.append(mae)

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")

    # Precision and Recall
    # Convert ratings to binary (relevant if rating >= 4)
    y_true_binary = (y_true >= 4).astype(int)
    y_pred_binary = (y_pred >= 4).astype(int)

    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)

    # Store metrics for visualization
    precision_list.append(precision)
    recall_list.append(recall)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")


def sampling_items_based(s, num_recommendations=5):
    all_similar_items = pd.Series(dtype=float)

    for i in range(s):
        print("Sampling . . . ", i + 1)
        sample_df = df0.sample(n=sample_size, replace=False)

        # Since not all user IDs are used during the sampling process, select from the sampled users.
        if i == 0:
            value = sample_df.iloc[0, 0]

        similar_items = items_based(value, sample_df, num_recommendations)

        all_similar_items = pd.concat([all_similar_items, similar_items])

    # Sorting
    all_similar_items = all_similar_items.sort_values(ascending=False)

    # Evaluate recommendations
    test_data = df0[df0['user_id'] == value]  # Use the original dataset for test
    evaluate_recommendations(test_data, all_similar_items)

    # Final recommendations
    all_similar_items = all_similar_items.head(num_recommendations)
    match_recipes(value, all_similar_items)


def plot_combined_metrics_bar():
    # Metric names and corresponding values
    metrics = ['RMSE', 'MAE', 'Precision', 'Recall']
    values = [
        np.mean(rmse_list),  # 평균 RMSE
        np.mean(mae_list),  # 평균 MAE
        np.mean(precision_list),  # 평균 Precision
        np.mean(recall_list)  # 평균 Recall
    ]

    # Plot the bar chart
    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'salmon', 'gold'], edgecolor='black')

    # Annotate each bar with its value
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.3f}", ha='center', va='bottom')

    # Customize the plot
    plt.title("Performance Metrics")
    plt.ylabel("Value")
    plt.ylim(0, max(values) + 0.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.show()


# Run sampling and evaluation
sampling_items_based(s=num_sample)

# Plot the metrics as bar graphs
plot_combined_metrics_bar()



