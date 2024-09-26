# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import locale
from matplotlib.ticker import PercentFormatter


# Set the locale to Portuguese (Brazil)
locale.setlocale(locale.LC_ALL, "pt_BR.UTF-8")

canditatos_dic = {
    "André Fernandes": ["Against-André Fernandes", "Pro-André Fernandes"],
    "Evandro Leitão": ["Against-Evandro Leitão", "Pro-Evandro Leitão"],
    "José Sarto": ["Against-José Sarto", "Pro-José Sarto"],
    "Capitão Wagner": ["Against-Capitão Wagner", "Pro-Capitão Wagner"],
}
custom_palette = {
    "Capitão Wagner": "blue",
    "José Sarto": "#FFD700",  # "#FFEB3B",
    "Evandro Leitão": "red",
    "André Fernandes": "green",
}


def get_posts(input_csv, platform, candidates, classification):
    """
    Filters and processes posts from a CSV file based on platform, candidate, and classification criteria.

    This function reads a CSV file containing posts data, filters the posts by the specified platform,
    candidate list, and classification list, and returns a cleaned and sorted DataFrame. It ensures
    that the 'dt' column contains valid date values and processes Facebook post IDs to be integers.

    Parameters:
    ----------
    input_csv : str
        The file path to the input CSV containing the posts data. The CSV should contain at least the
        following columns:
        - 'platform': the platform where the post was made (e.g., 'Facebook', 'Twitter').
        - 'new_candidate': the candidate's name or identifier.
        - 'new_classification': the classification of the post (e.g., 'Pro', 'Against').
        - 'post_id': unique identifier for the post.
        - 'dt': the date and time of the post.
        - 'count': the number of posts or interactions.
        - 'likeCount': the number of likes on the post.

    platform : str
        The platform name to filter the posts (e.g., 'Facebook', 'Twitter').

    candidates : list
        A list of candidate names or identifiers to filter the posts.

    classification : list
        A list of classifications to filter the posts (e.g., 'Pro', 'Against').

    Returns:
    -------
    pandas.DataFrame
        A DataFrame containing the filtered and processed posts data with the following columns:
        - 'platform': the platform of the post.
        - 'new_candidate': the candidate associated with the post.
        - 'new_classification': the classification of the post.
        - 'post_id': the unique identifier for the post, converted to an integer if the platform is 'Facebook'.
        - 'dt': the date and time of the post.
        - 'count': the number of interactions for the post.
        - 'likeCount': the number of likes on the post.

    Hint:
    -----
    The returned value is a pandas DataFrame.

    Notes:
    ------
    - The function ensures that the 'dt' column is converted to a datetime format.
    - For Facebook posts, the 'post_id' is converted to an integer to maintain consistency.
    - Duplicates are dropped, and the result is sorted by date and candidate.
    """

    # read posts
    posts = pd.read_csv(
        input_csv, parse_dates=True, dtype={"count": int, "likeCount": int}
    )

    # filter platform
    posts_p = posts[posts["platform"] == platform].copy()

    # ensure 'dt' are dates
    posts_p["dt"] = pd.to_datetime(posts_p["dt"])
    if platform == "Facebook":
        posts_p["post_id"] = posts_p["post_id"].astype(float).astype(int)

    # cleanup
    posts_p.drop_duplicates(inplace=True)
    posts_p = posts_p[posts_p["new_candidate"].isin(candidates)]
    posts_p = posts_p[posts_p["new_classification"].isin(classification)]

    # sort
    posts_p = posts_p.sort_values(by=["dt", "new_candidate"])

    return posts_p


def posts_and_likes(post_p):
    print(posts_p["platform"].unique())
    df = posts_p.groupby("new_candidate")["post_id"].count().reset_index(name="c")
    df["cumm"] = np.cumsum(df["c"])
    display(df)

    df = posts_p.groupby("new_candidate")["count"].sum().reset_index(name="c")
    df["cumm"] = np.cumsum(df["c"])
    display(df)


def post_count(post_p, custom_palette=custom_palette):
    post_count = (
        posts_p.groupby(["new_candidate", "dt"]).size().reset_index(name="post_count")
    )

    plt.figure(figsize=(14, 7))
    sns.lineplot(
        data=post_count,
        x="dt",
        y="post_count",
        hue="new_candidate",
        palette=custom_palette,
        markers=True,
        dashes=False,
        style="new_candidate",
        markeredgewidth=1.5,
        markeredgecolor=None,
    )

    # Customize the plot

    # Set x-ticks to every week
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

    # Format the x-ticks to show the date
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gca().set_facecolor("gray")  # You can use any color name or hex code

    plt.title(f"Post Count Over Time in {platform} for Each New Candidate")
    plt.xlabel("Date")
    plt.ylabel("Post Count")
    plt.legend(title="New Candidate")
    plt.xticks(rotation=90)
    plt.tight_layout()


def likes_count(post_p, custom_palette=custom_palette):
    # Group by 'new_candidate' and 'dt', then count the number of posts for each group

    likes_count = (
        posts_p.groupby(["new_candidate", "dt"])["count"]
        .sum()
        .reset_index(name="likes_count")
    )

    likes_count["dt"] = pd.to_datetime(likes_count["dt"])

    # df2['cumsum_like_count'] = df2.groupby('new_candidate')['like_count'].cumsum()

    plt.figure(1, figsize=(14, 7))
    sns.lineplot(
        data=likes_count,
        x="dt",
        y="likes_count",
        hue="new_candidate",
        palette=custom_palette,
        markers=True,
        dashes=False,
        style="new_candidate",
        markeredgewidth=1.5,
        markeredgecolor=None,
    )

    # Customize the plot

    # Set x-ticks to every week
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

    # Format the x-ticks to show the date
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gca().set_facecolor("gray")  # You can use any color name or hex code

    plt.title(f"Likes Count Over Time in {platform} for Each New Candidate")
    plt.xlabel("Date")
    plt.ylabel("Likes Count")
    plt.legend(title="New Candidate")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    likes_count["likes_count_cumsum"] = likes_count.groupby("new_candidate")[
        "likes_count"
    ].cumsum()

    plt.figure(2, figsize=(14, 7))
    sns.lineplot(
        data=likes_count,
        x="dt",
        y="likes_count_cumsum",
        hue="new_candidate",
        palette=custom_palette,
        markers=True,
        dashes=False,
        style="new_candidate",
        markeredgewidth=1.5,
        markeredgecolor=None,
    )

    # Customize the plot

    # Set x-ticks to every week
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

    # Format the x-ticks to show the date
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gca().set_facecolor("gray")  # You can use any color name or hex code

    plt.title(f"Cumulative Likes Count Over Time in {platform} for Each New Candidate")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Likes Count")
    plt.legend(title="New Candidate")
    plt.xticks(rotation=90)
    plt.tight_layout(),  # %%


def relative_approval(canditades_info, min_dt, max_dt, window_size, tau=45):
    """
    Computes the relative approval score for each candidate within a specified time window,
    applying an exponential decay based on time proximity to the current date.

    Parameters:
    ----------
    candidates_info : pandas.DataFrame
        A DataFrame containing candidate data, with the following required columns:
        - 'new_candidate': the name or identifier of the candidate.
        - 'new_classification': classification of the post (e.g., 'Pro', 'Against', etc.).
        - 'dt': the date of the post.
        - 'count': the number of posts or interactions.

    min_dt : datetime
        The minimum date in the current time window.

    max_dt : datetime
        The maximum date in the current time window. This is considered the "reference date"
        for the time window.

    window_size : int
        The number of days in the current time window.

    Returns:
    -------
    approbation : list of lists
        A list containing the relative approval score for each candidate. Each element is a list
        with the following format:
        - candidate: the name or identifier of the candidate.
        - max_dt: the reference date (i.e., the maximum date in the time window).
        - relative_approval: the candidate's relative approval score, normalized so that
          the sum of all candidates' scores equals 1.

    Notes:
    ------
    - The function applies an exponential decay based on the proximity of each post's date
      to the maximum date in the window. Posts closer to `max_dt` are given higher weights.
    - The approval score is calculated by summing the weighted number of "Pro" posts and
      subtracting the weighted number of "Against" posts.
    - The scores are then normalized by subtracting the minimum score from all candidates'
      scores to ensure no negative values, and finally, each score is divided by the total
      sum to compute the relative approval for each candidate.

    Procedure:
    ----------
    1. An exponential decay weight is computed for each day in the time window, with the decay
       constant (`tau`) set to 45.
    2. The posts for each candidate are weighted by this exponential decay, giving higher
       importance to more recent posts.
    3. The weighted number of "Pro" and "Against" posts is computed for each candidate.
    4. The relative approval score is calculated for each candidate as the difference between
       the weighted "Pro" and "Against" posts, divided by the total weighted count.
    5. The relative approval scores are normalized by subtracting the minimum value and dividing
       by the total sum of the scores for all candidates.

    """
    approbation = []
    dic_cand_approval = {}

    # For each Candidate
    for candidate in canditades_info["new_candidate"].unique():
        # Filter users_info for the specified candidate
        candidate_data = canditades_info[canditades_info["new_candidate"] == candidate]

        weight = []
        for n, t in zip(np.arange(1, window_size + 1), pd.date_range(min_dt, max_dt)):
            weight.append([t, np.exp(-1 * ((window_size - n) / tau))])

        weight = pd.DataFrame(weight, columns=["dt", "weight"])
        candidate_data = pd.merge(candidate_data, weight, on="dt")

        candidate_data["count"] = candidate_data["count"] * candidate_data["weight"]

        candidate_pros_cons = (
            candidate_data.groupby("new_classification")["count"].sum().reset_index()
        )

        # Step 2: Separate classifications with 'Pros' and 'Against' (real numbers...)
        idx_pros = candidate_pros_cons["new_classification"].str.contains("Pro")
        pros = candidate_pros_cons[idx_pros]["count"].iloc[0]
        idx_cons = candidate_pros_cons["new_classification"].str.contains("Against")
        against = candidate_pros_cons[idx_cons]["count"].iloc[0]

        # Convert the result into a dictionary
        candidate_ = candidate_pros_cons["new_classification"].iloc[0].split("-")[1]
        dic_cand_approval[candidate_] = (pros - against) / candidate_data["count"].sum()

    # Find the minimum value
    min_value = min(dic_cand_approval.values())
    for candidate in dic_cand_approval:
        dic_cand_approval[candidate] = dic_cand_approval[candidate] - min_value

    # note that the reference date is max_dt
    # approbation: [candidate, date, relative aproval]
    for candidate in dic_cand_approval:
        approbation.append(
            [
                candidate,
                max_dt,
                dic_cand_approval[candidate] / sum(dic_cand_approval.values()),
            ]
        )

    return approbation


def smooth_relative_approval(res_final, N=0):
    # Setting the alpha value for EWMA calculation
    if N == 0:
        N = int(len(res_final) / 3)

    alpha = 2 / (N + 1)  # Adjust as needed

    # Compute EWMA for each candidate's predictions
    def ewma_group(group, alpha=0.3):
        return group.ewm(alpha=alpha, adjust=False).mean()

    # Sort by candidate and dt to ensure calculations are done in the correct order
    res_final = res_final.sort_values(by=["Candidate", "dt"])

    # Apply the EWMA calculation to each candidate's predictions
    col_name = f"vote_intention_ewma_{N}"
    col_name = "vote_intention_ewma"
    res_final[col_name] = res_final.groupby("Candidate")["vote_intention"].transform(
        lambda x: ewma_group(x, alpha)
    )

    return res_final


def compute_relative_approval(post_plat, N=0):
    """
    Computes the relative approval of candidates over a progressively increasing time window.

    This function processes a dataset of posts (`posts_p`) that includes candidate, classification,
    and count data over time. It divides the data into progressively larger time windows, samples
    posts for each candidate, and ensures that all classifications are represented within the sampled
    posts. The relative approval of each candidate is then calculated based on these samples.

    Parameters:
    ----------
    post_plat : pandas.DataFrame, optional
        A DataFrame containing the posts data. It must include the following columns:
        - 'dt': datetime, the date of the post.
        - 'post_id': unique identifier for each post.
        - 'new_candidate': candidate name or identifier.
        - 'new_classification': classification of the post (e.g., 'Pos', 'Against', etc.).
        - 'count': the number of posts or interactions.

    Returns:
    -------
    res_final : pandas.DataFrame
        A DataFrame containing the averaged relative approval of each candidate over time.
        The DataFrame has the following columns:
        - 'Candidate': name or identifier of the candidate.
        - 'dt': datetime, the date.
        - 'vote_intention': averaged relative approval (or vote intention) for the candidate.

    Notes:
    ------
    - The function progressively increases the time window starting from a minimum date (`min_dt`)
      and continues until the maximum date in the dataset.
    - For each time window, posts are sampled from each candidate such that a certain number of
      posts are selected, ensuring that all classifications are represented in the sample.
    - The relative approval is computed using an external function `relative_approval`, which
      should handle rescaling and summarizing the data for each candidate.
    - The final result is the average relative approval of each candidate for each date, calculated
      from multiple samples.

    Procedure:
    ----------
    1. The function calculates the minimum and maximum dates (`min_dt`, `max_dt`) in the dataset,
       and computes the total length in days.
    2. For each window size (in days), a subset of the posts is selected.
    3. The function samples posts for each candidate within the window. The sample size is
       determined by the smallest number of unique posts from any candidate, divided by 5.
    4. The sampled posts are filtered and processed to ensure all classifications are represented.
    5. The relative approval of each candidate is calculated based on the sampled data.
    6. This process is repeated multiple times for each window size, and the mean relative approval
       for each candidate is computed.
    7. The results from each window are concatenated into a final DataFrame.

    """
    rel_approval = []
    min_dt = post_plat["dt"].min()
    max_length = (post_plat["dt"].max() - min_dt).days
    print(f"min_dt={pd.to_datetime(min_dt)},  max_length={max_length}")

    # Loop through the windows - window_size increase progressivelly?

    for window_size in range(50, max_length + 1):
        max_dt_ = pd.to_datetime(min_dt) + pd.DateOffset(days=window_size)
        posts_in_window = post_plat[
            (post_plat["dt"] >= min_dt) & (post_plat["dt"] < max_dt_)
        ].copy()

        print(
            f"{window_size}    min={pd.to_datetime(min_dt)}, max={pd.to_datetime(max_dt_)} posts_in_window:{posts_in_window.shape}"
        )
        res = []
        for i in range(40):  # <--------------bootstrap # Loop for sampling
            # take the number of unique posts by candidate,
            # than sample size is one-fifth of the smallest number of unique posts
            sample_size = (
                posts_in_window.groupby("new_candidate")["post_id"].nunique().min() // 5
            )
            if i == 1:
                print(f"sample size= {sample_size}")

            def sample_post_ids(pposts):  # define sample_size in scope
                unique_post_ids = pposts["post_id"].unique()
                sampled = np.random.choice(
                    unique_post_ids, size=sample_size, replace=True
                )
                return sampled

            while True:
                # Apply the function to each candidate group and get a list of sampled post IDs
                posts_in_window_by_candidate = posts_in_window.groupby("new_candidate")

                # Apply the sampling function 'sample_post_ids' to each group
                sampled_postid_by_candidate = posts_in_window_by_candidate.apply(
                    sample_post_ids, include_groups=False
                )

                # Flatten the resulting series of lists into a single series
                flattened_ = sampled_postid_by_candidate.explode()

                # sample_size for each candidate
                sampled_post_ids = flattened_.reset_index(drop=True)

                # Filter the original DataFrame based on sampled post IDs
                sampled_posts_by_candidate = posts_in_window[
                    posts_in_window["post_id"].isin(sampled_post_ids)
                ]

                # Check if all 8 classifications are represented
                if sampled_posts_by_candidate["new_classification"].nunique() == len(
                    classification
                ):
                    break

            # Group by candidate, classification, and date, then sum the 'count'
            candidates_info = sampled_posts_by_candidate.groupby(
                ["new_candidate", "new_classification", "dt"], as_index=False
            )["count"].sum()
            if i == -10:
                print(candidates_info.sort_values(by="dt"))

            # Rescale the data (assuming `rescale_` is a predefined function)
            approval = relative_approval(candidates_info, min_dt, max_dt_, window_size)
            res.append(approval)

        # Flatten the nested list
        df = pd.DataFrame(
            [item for sublist in res for item in sublist],
            columns=["Candidate", "dt", "vote_intention"],
        )

        # Calculate the mean of rescaled data for each candidate and date
        rel_approval.append(df.groupby(["Candidate", "dt"], as_index=False).mean())

    # Concatenate all results into a final DataFrame
    rel_approval = pd.concat(rel_approval)

    return rel_approval


def plot_relative_approval(
    df,
    col,
    custom_palette,
    tit="",
    filename="none",
    ylabel="Aprovação Relativa (%)",
    y_shift={
        "André Fernandes": 0,
        "Evandro Leitão": 0,
        "Capitão Wagner": 0,
        "José Sarto": 0,
    },
    canditate_order=[
        "Evandro Leitão",
        "André Fernandes",
        "Capitão Wagner",
        "José Sarto",
    ],
):

    # Set the plot size
    fig, ax = plt.subplots(figsize=(12, 5))

    # Create the line plot
    sns.lineplot(
        data=df,
        x="dt",
        y=col,
        hue="Candidate",
        hue_order=canditate_order,
        markers=False,
        palette=custom_palette,
        linewidth=2.5,
    )
    # style="Candidate",
    # markersize=5,
    # markeredgewidth=1.5,
    # markeredgecolor=None,
    sns.set_theme(style="white", rc={"axes.grid": False})
    sns.despine()  # Only remove the left spine; keep the bottom spine


    # Add text labels at the last data point for each candidate
    x_shift = pd.Timedelta(days=1)  # Shift by 2 days, adjust as necessary
    for candidate in df["Candidate"].unique():
        candidate_data = df[df["Candidate"] == candidate]
        x = candidate_data["dt"].iloc[-1] + x_shift
        y = candidate_data[col].iloc[-1] + y_shift[candidate]/100
        yt = np.round(100 * candidate_data[col].iloc[-1], 0)
        texto = f"{yt:.0f}%"
        plt.text(x, y, texto, color=custom_palette[candidate], fontsize=10, ha="left")

    date_format = mdates.DateFormatter("%-d de %b")
    ax.xaxis.set_major_formatter(date_format)
    plt.xlabel("")
    plt.xlim(pd.Timestamp("2024-08-01"), np.max(df["dt"]) + pd.Timedelta(days=5))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())  # Set major ticks to weekly

    ax.yaxis.set_major_formatter(
        PercentFormatter(1.0, decimals=0)
    )  # 1.0 scale means values from 0 to 1 are formatted as percentages
    plt.ylabel(ylabel)  # Corrected label name for clarity

    plt.title(tit, fontdict={"fontsize": 14, "fontweight": "bold"}, pad=40)
    plt.legend(
        loc="upper center", frameon=False, ncol=4, bbox_to_anchor=(0.5, 1.15)
    )  # Add a legend with a title
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    if filename != "none":
        filename = f"{filename}_{tit}.png"
        filename = filename.replace(" ", "_")
        plt.savefig(filename, format="png", dpi=300)
        print(f"saved {filename}")
    ax.tick_params(axis='x', which='major', length=5, color='black')  # Ensure tick marks are shown

    # plt.subplots_adjust(top=0.85)
    plt.show()


def plot_relative_approval_grp(df, columns, custom_palette, tit="", filename="none"):

    # Set the plot size
    plt.figure(figsize=(12, 6))
    for col in columns:
        # Create the line plot
        sns.lineplot(
            data=df,
            x="dt",
            y=col,
            hue="Candidate",
            markers=True,
            palette=custom_palette,
            style="Candidate",
            markeredgewidth=1.5,
            markeredgecolor=None,
        )

        # Add text labels at the last data point for each candidate
        x_shift = pd.Timedelta(days=1)  # Shift by 2 days, adjust as necessary
        for candidate in df["Candidate"].unique():
            candidate_data = df[df["Candidate"] == candidate]
            x = candidate_data["dt"].iloc[-1] + x_shift
            y = candidate_data[col].iloc[-1]
            texto = f"{candidate_data[col].iloc[-1]:.2f}"
            plt.text(
                x, y, texto, color=custom_palette[candidate], fontsize=10, ha="left"
            )

    # Add labels and title
    plt.xlabel("Date")
    plt.ylabel("Relative Approval")  # Corrected label name for clarity
    plt.title(
        f"Relative Approval Over Time by Candidate {tit}"
    )  # Improved title for clarity

    # Display the plot with enhancements
    plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
    plt.grid(True)  # Add grid lines for better visual interpretation

    plt.gca().set_facecolor("lightgray")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    plt.legend(by_label.values(), by_label.keys())  # Add a legend with a title
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    if filename != "none":
        filename = f"{filename}_{tit}.png"
        filename = filename.replace(" ", "_")
        plt.savefig(filename, format="png", dpi=300)
        print(f"saved {filename}")
    plt.show()


def plot_renormalized(
    rel_approval, ncandidatos, custom_palette=custom_palette, col="vote_intention_ewma"
):
    filtered_data = rel_approval[rel_approval["Candidate"].isin(ncandidatos)].copy()

    # Normalize the ewma_prediction values so that the sum for each day equals 1
    filtered_data.loc[:, "normalized_ewma"] = filtered_data.groupby("dt")[
        col
    ].transform(lambda x: x / x.sum())

    # Set the plot size
    plt.figure(figsize=(12, 6))

    # Create the line plot for the normalized EWMA prediction
    sns.lineplot(
        data=filtered_data,
        x="dt",
        y="normalized_ewma",
        hue="Candidate",
        markers=True,
        palette=custom_palette,
        style="Candidate",
        markeredgewidth=1.5,
        markeredgecolor=None,
    )

    # Add labels and title
    plt.xlabel("Date")
    plt.ylabel("Normalized Vote Intention")
    plt.title(f"Renormalized Vote Intention Over Time by Candidate")

    # Enhance the plot display
    plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
    plt.grid(True)  # Add grid lines for better visual interpretation
    plt.legend(title="Candidate")  # Add a legend with a title
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.gca().set_facecolor("gray")  # You can use any color name or hex code

    # Add text labels at the last data point for each selected candidate
    x_shift = pd.Timedelta(days=1)  # Shift by 2 days, adjust as necessary

    for candidate in filtered_data["Candidate"].unique():
        candidate_data = filtered_data[filtered_data["Candidate"] == candidate]
        plt.text(
            candidate_data["dt"].iloc[-1] + x_shift,
            candidate_data["normalized_ewma"].iloc[-1],
            f'{candidate_data["normalized_ewma"].iloc[-1]:.2f}',
            color=custom_palette[candidate],
            fontsize=10,
            ha="left",
        )

    plt.show()


def combine_platforms(
    rel_approval_dic,
    alpha=0.5,
    custom_palette=custom_palette,
    col="vote_intention",
    tit="",
    filename="none",
):

    vi_Insta = rel_approval_dic["Instagram"]
    vi_Face = rel_approval_dic["Facebook"]

    merged_df = pd.merge(vi_Face, vi_Insta, on=["Candidate", "dt"], how="inner")
    colx = f"{col}_x"
    coly = f"{col}_y"
    if colx in merged_df.columns and coly in merged_df.columns:
        merged_df[col] = (
            alpha * merged_df[colx].values + (1 - alpha) * merged_df[coly].values
        )
    return merged_df


def plot_relative_combined(
    rel_approval_dic,
    ncandidatos,
    col,
    custom_palette,
    alpha=0.5,
    tit="",
    filename="none",
):

    ra_Insta = rel_approval_dic["Instagram"]
    filtered_Insta = ra_Insta[ra_Insta["Candidate"].isin(ncandidatos)].copy()

    # Normalize the ewma_prediction values so that the sum for each day equals 1
    filtered_Insta.loc[:, "normalized_ewma"] = filtered_Insta.groupby("dt")[
        col
    ].transform(lambda x: x / x.sum())

    ra_Face = rel_approval_dic["Facebook"]
    filtered_Face = ra_Face[ra_Face["Candidate"].isin(ncandidatos)].copy()

    # Normalize the ewma_prediction values so that the sum for each day equals 1
    filtered_Face.loc[:, "normalized_ewma"] = filtered_Face.groupby("dt")[
        col
    ].transform(lambda x: x / x.sum())

    merged_df = pd.merge(
        filtered_Insta, filtered_Face, on=["Candidate", "dt"], how="inner"
    )
    colx = f"{col}_x"
    coly = f"{col}_y"
    if colx in merged_df.columns and coly in merged_df.columns:
        merged_df[col] = (
            alpha * merged_df[colx].values + (1 - alpha) * merged_df[coly].values
        )

    # Set the plot size
    plt.figure(figsize=(12, 6))

    # Create the line plot for the normalized EWMA prediction
    sns.lineplot(
        data=merged_df,
        x="dt",
        y=col,
        hue="Candidate",
        markers=True,
        palette=custom_palette,
        style="Candidate",
        markeredgewidth=1.5,
        markeredgecolor=None,
    )

    # Add labels and title
    plt.xlabel("Date")
    plt.ylabel("Relative Approval")
    plt.title(
        "Relative Approval Over Time by Candidate Combined Instagram and Facebook"
    )

    # Enhance the plot display
    plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
    plt.grid(True)  # Add grid lines for better visual interpretation
    plt.legend(title="Candidate", loc="upper left")  # Add a legend with a title
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.gca().set_facecolor("lightgray")  # You can use any color name or hex code

    # Add text labels at the last data point for each selected candidate
    x_shift = pd.Timedelta(days=1)  # Shift by 2 days, adjust as necessary
    y_shift = {
        "André Fernandes": 0.0051,
        "Evandro Leitão": -0.0051,
        "Capitão Wagner": 0.0,
        "José Sarto": 0.0,
    }
    for candidate in merged_df["Candidate"].unique():
        candidate_data = merged_df[merged_df["Candidate"] == candidate]
        y = candidate_data[col].iloc[-1]
        plt.text(
            candidate_data["dt"].iloc[-1] + x_shift,
            candidate_data[col].iloc[-1] + y_shift[candidate],
            f"{y:.3f}",
            color=custom_palette[candidate],
            fontsize=10,
            ha="left",
        )

    if filename != "none":
        filename = f"{filename}_{tit}.png"
        filename = filename.replace(" ", "_")
        plt.savefig(filename, format="png", dpi=300)
        print(f"saved {filename}")
    plt.show()


# %%


candidates = ["Evandro Leitão", "André Fernandes", "José Sarto", "Capitão Wagner"]
# candidates = ['Evandro Leitão', 'André Fernandes',  'José Sarto']
classification = [item for sublist in candidates for item in canditatos_dic[sublist]]

input_csv = "./data/approval_17.zip"
input_csv = "./data/Aprovação_18.csv"

platforms = ["Instagram", "Facebook"]
platform = platforms[0]

rel_approval = {}
posts_p = {}
for platform in platforms:
    print(f"computing {platform} ==============================================")
    posts_p[platform] = get_posts(input_csv, platform, candidates, classification)
    rel_approval[platform] = compute_relative_approval(posts_p[platform], N=30)

for platform in platforms:
    rel_approval[platform] = smooth_relative_approval(rel_approval[platform], 40)

# %%  --------------------------------------------------------------------------

cols = ["vote_intention_ewma"]
platform = "Instagram"
for col in cols:
    plot_relative_approval(
        rel_approval[platform],
        col,
        custom_palette,
        tit=f"Aprovação Relativa Acumulada {platform}",
        filename="none",
    )

y_shift = {
    "André Fernandes": 0.0051,
    "Evandro Leitão": -0.0051,
    "Capitão Wagner": 0.0,
    "José Sarto": 0.0,
}
platform = "Facebook"
for col in cols:
    plot_relative_approval(
        rel_approval[platform],
        col,
        custom_palette,
        tit=f"Aprovação Relativa Acumulada {platform}",
        filename="none",
        y_shift=y_shift,
    )


# %%
candidatesred = ["Evandro Leitão", "André Fernandes", "José Sarto"]

for platform in platforms[0:1]:
    col = ["vote_intention_ewma"]
    plot_renormalized(
        rel_approval[platform], candidatesred, custom_palette=custom_palette, col=col
    )


# %%
age_social = pd.read_csv("./data/Insta_Face_users.csv")
age_I = age_social[["DS_FAIXA_ETARIA", "Instagram"]]
age_I["cat"] = "Instagram"

age_I = age_I.rename(columns={"Instagram": "frac"})

age_F = age_social[["DS_FAIXA_ETARIA", "Facebook"]]
age_F["cat"] = "Facebook"
age_F = age_F.rename(columns={"Facebook": "frac"})

age_voters = pd.read_csv("./data/eleitorado_2024_fortaleza.csv")
age_voters = age_voters[["DS_FAIXA_ETARIA2", "count"]]
age_voters = age_voters[age_voters["DS_FAIXA_ETARIA2"] != "0"]
age_voters = (
    age_voters.groupby("DS_FAIXA_ETARIA2").sum().reset_index(names="DS_FAIXA_ETARIA")
)
age_voters["frac"] = age_voters["count"] / np.sum(age_voters["count"])
age_voters = age_voters[["DS_FAIXA_ETARIA", "frac"]]
age_voters["cat"] = "Voters"

df_combined = pd.concat([age_I, age_F, age_voters], ignore_index=True)
print(df_combined)

# %%  --------------------------------------------------------------------------


plt.figure(figsize=(10, 6))
sns.barplot(data=df_combined, x="DS_FAIXA_ETARIA", y="frac", hue="cat")

# Rotate x labels for better readability
plt.xticks(rotation=45, ha="right")
plt.title("")
plt.tight_layout()
plt.show()
# plt.savefig("age_dist.png", format="png", dpi=300)


# %%  --------------------------------------------------------------------------
a = np.dot(age_F["frac"], age_voters["frac"])
b = np.dot(age_I["frac"], age_voters["frac"])

cf = a / (a + b)
ci = b / (a + b)

print(f"{cf} {ci} {cf+ci}")

# %%  --------------------------------------------------------------------------
comb = combine_platforms(rel_approval_dic=rel_approval, alpha=ci)
comb = smooth_relative_approval(comb, 40)
print(comb)

y_shift = {
        "André Fernandes": -0.5,
        "Evandro Leitão": 0.5,
        "Capitão Wagner": 0,
        "José Sarto": 0
    }

plot_relative_approval(
    comb,
    "vote_intention_ewma",
    custom_palette,
    tit="Aprovação Relativa Acumulada - Instagram e Facebook combinados",
    filename="smooth_40_18",
    ylabel="Aprovação Relativa Acumulada (%)",
    y_shift=y_shift

)

y_shift = {
        "André Fernandes": -1.,
        "Evandro Leitão": 1.,
        "Capitão Wagner": 0,
        "José Sarto": 0
    }


plot_relative_approval(
    comb,
    "vote_intention",
    custom_palette,
    tit="Aprovação Relativa - Instagram e Facebook combinados",
    filename="rel_appr_18",
    y_shift=y_shift
)

# %%
