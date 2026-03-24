from app.app_data import load_track_explorer_data

track_df = load_track_explorer_data()

print("Shape:", track_df.shape)
print("\nColumns:")
print(track_df.columns.tolist())

dup_count = track_df.duplicated(
    subset=["release_group_mbid", "tmdb_id", "track_number"]
).sum()
print("\nDuplicate album-track-number rows:", int(dup_count))

print("\nTrack number summary:")
print(track_df["track_number"].describe())

print("\nObserved track count summary:")
print(track_df["track_count_observed"].describe())

compare_cols = [
    col for col in [
        "film_title",
        "album_title",
        "track_number",
        "track_title",
        "track_count_observed",
        "max_track_number_observed",
        "n_tracks",
        "lfm_track_listeners",
        "spotify_popularity",
    ]
    if col in track_df.columns
]

print("\nSample rows:")
print(track_df[compare_cols].head(20).to_string(index=False))