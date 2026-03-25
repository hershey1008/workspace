import gc
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        col_type = df[col].dtype

        if pd.api.types.is_integer_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()

            if c_min >= 0:
                if c_max <= np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif c_max <= np.iinfo(np.uint16).max:
                    df[col] = df[col].astype(np.uint16)
                elif c_max <= np.iinfo(np.uint32).max:
                    df[col] = df[col].astype(np.uint32)
                else:
                    df[col] = df[col].astype(np.uint64)
            else:
                if np.iinfo(np.int8).min <= c_min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif np.iinfo(np.int16).min <= c_min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif np.iinfo(np.int32).min <= c_min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)

        elif pd.api.types.is_float_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="float")

    return df


def convert_date(date: int):
    date_str = str(date)
    year = date_str[:4]
    month = date_str[4:6]
    day = date_str[6:]
    return datetime(int(year), int(month), int(day))


def compute_last_k_clicked_history(
    df: pd.DataFrame, k: int = 20, pad_value: int = 0
) -> pd.DataFrame:
    """
    计算每个用户在每个日期的“前一天及更早”的最后k个点击视频历史。
    同一个 user_id + date 下，所有样本得到相同的 last_k_clicked_items。
    """
    required_cols = ["user_id", "video_id", "date", "time_ms", "is_click"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(
            "输入的DataFrame必须包含列: 'user_id', 'video_id', 'date', 'time_ms', 'is_click'"
        )

    if k <= 0:
        df = df.copy()
        df["last_k_clicked_items"] = [[] for _ in range(len(df))]
        return df

    df_processed = df.copy()
    df_processed["date"] = df_processed["date"].apply(convert_date)
    df_processed["is_click"] = df_processed["is_click"].astype(bool)
    df_processed = df_processed.sort_values(
        by=["user_id", "date", "time_ms"], ascending=True
    ).reset_index(drop=True)

    clicked_df = df_processed[df_processed["is_click"]].copy()

    if clicked_df.empty:
        print("Warning: No click interactions found in the data.")
        df_processed["last_k_clicked_items"] = [[pad_value] * k for _ in range(len(df_processed))]
        return df_processed

    daily_clicks = (
        clicked_df.groupby(["user_id", "date"], sort=True)["video_id"]
        .apply(list)
        .reset_index()
    )
    daily_clicks = daily_clicks.sort_values(by=["user_id", "date"]).reset_index(drop=True)

    # 每个用户按天累积点击历史
    def cumulative_concat(series):
        acc = []
        out = []
        for x in series:
            acc = acc + x
            out.append(acc.copy())
        return out

    daily_clicks["cumulative_history"] = (
        daily_clicks.groupby("user_id")["video_id"]
        .transform(cumulative_concat)
    )

    # 前一天及以前的历史
    daily_clicks["prev_days_history"] = (
        daily_clicks.groupby("user_id")["cumulative_history"].shift(1)
    )
    daily_clicks["prev_days_history"] = daily_clicks["prev_days_history"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    history_map = daily_clicks[["user_id", "date", "prev_days_history"]]

    df_processed = pd.merge(
        df_processed, history_map, on=["user_id", "date"], how="left"
    )

    df_processed = df_processed.sort_values(
        by=["user_id", "date", "time_ms"], ascending=True
    ).reset_index(drop=True)

    df_processed["propagated_history"] = (
        df_processed.groupby("user_id")["prev_days_history"].ffill()
    )
    df_processed["propagated_history"] = df_processed["propagated_history"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    def pad_and_truncate(history_list: list, target_k: int, pad_val: int) -> list:
        last_k = history_list[-target_k:]
        pad_len = target_k - len(last_k)
        return ([pad_val] * pad_len) + last_k

    df_processed["last_k_clicked_items"] = df_processed["propagated_history"].apply(
        lambda hist: pad_and_truncate(hist, k, pad_value)
    )

    return df_processed.drop(columns=["prev_days_history", "propagated_history"])


def compute_session_id(df: pd.DataFrame, time_threshold_minutes: int = 30) -> pd.Series:
    """
    根据用户活动间隙计算 session_id，返回整数型 session_id，避免字符串高内存占用。
    """
    if not all(col in df.columns for col in ["user_id", "time_ms"]):
        raise ValueError("输入的DataFrame必须包含列: 'user_id', 'time_ms'")

    if df.empty:
        return pd.Series(dtype=np.int64)

    df_processed = df.copy()
    df_processed["time_ms"] = pd.to_numeric(df_processed["time_ms"], downcast="integer")

    df_processed = df_processed.sort_values(
        by=["user_id", "time_ms"], ascending=True
    ).reset_index()

    time_threshold_ms = time_threshold_minutes * 60 * 1000

    prev_time_ms = df_processed.groupby("user_id")["time_ms"].shift(1)
    time_diff_ms = df_processed["time_ms"] - prev_time_ms

    is_new_session = (time_diff_ms > time_threshold_ms) | (prev_time_ms.isna())

    # 每个用户内会话从0开始编号
    session_numeric_id = (
        is_new_session.groupby(df_processed["user_id"]).cumsum().astype(np.int32) - 1
    )

    # 组合成全局唯一整数ID
    session_ids = (
        df_processed["user_id"].astype(np.int64) * 100000
        + session_numeric_id.astype(np.int64)
    )

    session_series = pd.Series(session_ids.values, index=df_processed["index"])
    return session_series.sort_index()


def save_split_pickle(data_df: pd.DataFrame, save_file: Path):
    """
    分 split 保存，避免一次性构造 train/test 总大字典。
    """
    save_dict = {}
    for feat_name in data_df.columns:
        save_dict[feat_name] = data_df[feat_name].to_numpy(copy=False)

    with open(save_file, "wb") as f:
        pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    del save_dict
    gc.collect()


def preprocess(input_path: Path, output_path: Path) -> dict:
    print("加载数据...")

    select_log_columns = [
        "user_id",
        "video_id",
        "date",
        "time_ms",
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_hate",
        "long_view",
        "is_profile_enter",
        "tab",
    ]

    log_dtype = {
        "user_id": np.int32,
        "video_id": np.int32,
        "date": np.int32,
        "time_ms": np.int64,
        "is_click": np.int8,
        "is_like": np.int8,
        "is_follow": np.int8,
        "is_comment": np.int8,
        "is_forward": np.int8,
        "is_hate": np.int8,
        "long_view": np.int8,
        "is_profile_enter": np.int8,
        "tab": np.int16,
    }

    log_df = pd.read_csv(
        input_path / "data" / "log_standard_4_22_to_5_08_1k.csv",
        usecols=select_log_columns,
        dtype=log_dtype,
    )

    user_sparse_feature_columns = [
        "user_id",
        "user_active_degree",
        "is_live_streamer",
        "is_video_author",
        "follow_user_num_range",
        "fans_user_num_range",
        "friend_user_num_range",
        "register_days_range",
        "onehot_feat0",
        "onehot_feat1",
        "onehot_feat2",
        "onehot_feat3",
        "onehot_feat4",
        "onehot_feat5",
        "onehot_feat6",
        "onehot_feat7",
        "onehot_feat8",
        "onehot_feat9",
        "onehot_feat10",
        "onehot_feat11",
        "onehot_feat12",
        "onehot_feat13",
        "onehot_feat14",
        "onehot_feat15",
        "onehot_feat16",
        "onehot_feat17",
    ]

    user_features = pd.read_csv(
        input_path / "data" / "user_features_1k.csv",
        usecols=user_sparse_feature_columns + [
            "follow_user_num",
            "fans_user_num",
            "friend_user_num",
            "register_days",
        ],
    )

    select_video_basic_feature_columns = [
        "video_id",
        "author_id",
        "video_type",
        "upload_type",
        "visible_status",
        "music_id",
        "music_type",
        "tag",
    ]

    video_features_basic = pd.read_csv(
        input_path / "data" / "video_features_basic_1k.csv",
        usecols=select_video_basic_feature_columns,
    )

    video_features_statistics_columns = [
        "video_id",
        "counts",
        "show_cnt",
        "show_user_num",
        "play_cnt",
        "play_user_num",
        "play_duration",
        "complete_play_cnt",
        "complete_play_user_num",
        "valid_play_cnt",
        "valid_play_user_num",
        "long_time_play_cnt",
        "long_time_play_user_num",
        "short_time_play_cnt",
        "short_time_play_user_num",
        "play_progress",
        "comment_stay_duration",
        "like_cnt",
        "like_user_num",
        "click_like_cnt",
        "double_click_cnt",
        "cancel_like_cnt",
        "cancel_like_user_num",
        "comment_cnt",
        "comment_user_num",
        "direct_comment_cnt",
        "reply_comment_cnt",
        "delete_comment_cnt",
        "delete_comment_user_num",
        "comment_like_cnt",
        "comment_like_user_num",
        "follow_cnt",
        "follow_user_num",
        "cancel_follow_cnt",
        "cancel_follow_user_num",
        "share_cnt",
        "share_user_num",
        "download_cnt",
        "download_user_num",
        "report_cnt",
        "report_user_num",
        "reduce_similar_cnt",
        "reduce_similar_user_num",
        "collect_cnt",
        "collect_user_num",
        "cancel_collect_cnt",
        "cancel_collect_user_num",
        "direct_comment_user_num",
        "reply_comment_user_num",
        "share_all_cnt",
        "share_all_user_num",
        "outsite_share_all_cnt",
    ]

    video_features_statistics = pd.read_csv(
        input_path / "data" / "video_features_statistic_1k.csv",
        usecols=video_features_statistics_columns,
    )

    print("处理用户特征...")
    new_user_feature_df = user_features[user_sparse_feature_columns].copy()

    # is_live_streamer中的-124转0
    new_user_feature_df["is_live_streamer"] = new_user_feature_df["is_live_streamer"].apply(
        lambda x: 0 if x == -124 else x
    )

    user_label_encode_feature_columns = [
        "user_id",
        "user_active_degree",
        "follow_user_num_range",
        "fans_user_num_range",
        "friend_user_num_range",
        "register_days_range",
    ]

    for feat_name in user_label_encode_feature_columns:
        label_encoder = LabelEncoder()
        encoded = label_encoder.fit_transform(new_user_feature_df[feat_name].astype(str)) + 1
        if feat_name == "user_id":
            new_user_feature_df["user_id_encode"] = encoded.astype(np.int32)
        else:
            new_user_feature_df[feat_name] = encoded.astype(np.int32)

    onehot_feat_columns = [x for x in user_sparse_feature_columns if "onehot_" in x]
    for feat_name in onehot_feat_columns:
        max_val = new_user_feature_df[feat_name].max()
        fill_val = (0 if pd.isna(max_val) else max_val) + 1
        new_user_feature_df[feat_name] = new_user_feature_df[feat_name].fillna(fill_val).astype(np.int32)

    del user_features
    gc.collect()

    print("处理视频基础特征...")
    new_video_features_basic_df = video_features_basic[select_video_basic_feature_columns].copy()

    for feat_name in ["visible_status", "music_type"]:
        max_val = new_video_features_basic_df[feat_name].max()
        fill_val = (0 if pd.isna(max_val) else max_val) + 1
        new_video_features_basic_df[feat_name] = (
            new_video_features_basic_df[feat_name].fillna(fill_val).astype(np.int32)
        )

    video_sparse_feature_columns = [
        x for x in select_video_basic_feature_columns if x != "tag"
    ]

    for feat_name in video_sparse_feature_columns:
        label_encoder = LabelEncoder()
        encoded = label_encoder.fit_transform(new_video_features_basic_df[feat_name].astype(str)) + 1
        if feat_name == "video_id":
            new_video_features_basic_df["video_id_encode"] = encoded.astype(np.int32)
        else:
            new_video_features_basic_df[feat_name] = encoded.astype(np.int32)

    new_video_features_basic_df["tag"] = new_video_features_basic_df["tag"].fillna("-1")

    tag_set = set()
    for x in new_video_features_basic_df["tag"].astype(str).values:
        for tag in x.split(","):
            tag_set.add(tag)

    tag_map_dict = {tag: i + 1 for i, tag in enumerate(tag_set)}

    new_video_features_basic_df["tag"] = new_video_features_basic_df["tag"].apply(
        lambda x: [tag_map_dict[tag] for tag in str(x).split(",")]
    )

    del video_features_basic
    gc.collect()

    print("处理视频统计特征...")
    video_dense_feature_columns = [
        x for x in video_features_statistics_columns if x not in ("video_id", "counts")
    ]

    new_video_features_statistics_df = video_features_statistics[
        ["video_id"] + video_dense_feature_columns
    ].copy()

    for feat_name in video_dense_feature_columns:
        new_video_features_statistics_df[feat_name] = (
            new_video_features_statistics_df[feat_name].fillna(0.0).astype(np.float32)
        )

    del video_features_statistics
    gc.collect()

    print("合并特征...")
    df_merged = log_df.merge(new_user_feature_df, on="user_id", how="left")
    del log_df, new_user_feature_df
    gc.collect()

    df_merged = df_merged.merge(new_video_features_basic_df, on="video_id", how="left")
    del new_video_features_basic_df
    gc.collect()

    df_merged = df_merged.merge(
        new_video_features_statistics_df, on="video_id", how="left"
    )
    del new_video_features_statistics_df
    gc.collect()

    # 替换id为encode id
    df_merged["user_id"] = df_merged["user_id_encode"].astype(np.int32)
    df_merged["video_id"] = df_merged["video_id_encode"].astype(np.int32)
    del df_merged["user_id_encode"]
    del df_merged["video_id_encode"]

    # 处理tag：空值填0，仅保留第一个tag
    def process_tag(x):
        if isinstance(x, list):
            return x[0] if len(x) > 0 else 0
        return 0

    df_merged["tag"] = df_merged["tag"].apply(process_tag).astype(np.int32)

    df_merged.fillna(value=0, inplace=True)

    print("数值特征分桶...")
    for feat_name in tqdm(video_dense_feature_columns):
        binned = pd.qcut(
            df_merged[feat_name],
            q=10,
            labels=False,
            duplicates="drop",
        )
        df_merged[feat_name] = binned.fillna(0).astype(np.uint8)

    # 主场景过滤
    main_tab_set = {1, 0, 4, 2, 6}
    df_merged = df_merged[df_merged["tab"].isin(main_tab_set)].reset_index(drop=True)

    # tab重编码为连续值
    label_encoder = LabelEncoder()
    df_merged["tab"] = label_encoder.fit_transform(df_merged["tab"]).astype(np.int8)

    print("计算每个用户在每个日期点击的最后k个视频id，基于当前日期之前的点击...")
    history_df = compute_last_k_clicked_history(
        df_merged[["user_id", "video_id", "date", "time_ms", "is_click"]],
        k=20,
        pad_value=0,
    )

    # 校验 user-date 下序列一致
    def check_identical_sequences(group):
        first_sequence = tuple(group.iloc[0])
        return all(tuple(seq) == first_sequence for seq in group)

    group_results = history_df.groupby(["user_id", "date"])["last_k_clicked_items"].apply(
        check_identical_sequences
    )
    mismatched_groups = group_results[~group_results].index.tolist()

    assert len(mismatched_groups) == 0, (
        f"Found mismatched sequences for the following user-date pairs: {mismatched_groups}"
    )

    # 拆成20个固定整型列，避免object/list列占内存
    history_arr = np.vstack(history_df["last_k_clicked_items"].to_numpy()).astype(np.int32)
    for i in range(history_arr.shape[1]):
        df_merged[f"last_click_{i}"] = history_arr[:, i]

    del history_df
    del history_arr
    gc.collect()

    print("计算会话id...")
    df_merged = df_merged.reset_index(drop=True)
    df_merged["session_id"] = compute_session_id(df_merged[["user_id", "time_ms"]]).astype(np.int64)

    print("压缩数据类型...")
    df_merged = reduce_mem_usage(df_merged)

    print("生成特征词典...")
    not_feat_dict_columns = [
        "date",
        "time_ms",
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_hate",
        "long_view",
        "is_profile_enter",
        "session_id",
    ]
    total_columns = [x for x in list(df_merged.columns) if x not in not_feat_dict_columns]

    final_feature_dict = {}
    for feat_name in total_columns:
        final_feature_dict[feat_name] = int(df_merged[feat_name].max()) + 1

    save_path = output_path / "feature_dict" / "kuairand_feature_dict.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(final_feature_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("划分训练和测试集并分别保存...")
    save_dir = output_path / "train_eval_sample_final"
    save_dir.mkdir(parents=True, exist_ok=True)

    is_test = df_merged["date"].eq(20220508)

    train_df = df_merged.loc[~is_test].copy()
    train_df.drop(columns=["date", "time_ms"], inplace=True)
    train_df = reduce_mem_usage(train_df)

    test_df = df_merged.loc[is_test].copy()
    test_df.drop(columns=["date", "time_ms"], inplace=True)
    test_df = reduce_mem_usage(test_df)

    train_dict = {}
    for feat_name in train_df.columns:
        train_dict[feat_name] = train_df[feat_name].to_numpy(copy=False)

    test_dict = {}
    for feat_name in test_df.columns:
        test_dict[feat_name] = test_df[feat_name].to_numpy(copy=False)

    train_eval_dict = {
        "train": train_dict,
        "test": test_dict,
    }

    with open(save_dir / "kuairand_train_eval.pkl", "wb") as f:
        pickle.dump(train_eval_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    del train_df, test_df, train_dict, test_dict, train_eval_dict, df_merged
    gc.collect()

    print("保存完成。")