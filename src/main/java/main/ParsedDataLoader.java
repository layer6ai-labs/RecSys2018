package main;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import common.MLFeatureTransform;
import common.MLMatrixElement;
import common.MLSparseFeature;
import common.MLSparseMatrixAOO;
import common.MLSparseVector;
import common.MLTextTransform;
import common.MLTimer;
import main.ParsedData.PlaylistFeature;
import main.ParsedData.SongExtraInfoFeature;
import main.ParsedData.SongFeature;
import net.minidev.json.JSONArray;
import net.minidev.json.JSONObject;
import net.minidev.json.parser.JSONParser;

public class ParsedDataLoader {

	private Data dataLoaded;
	public ParsedData dataParsed;

	public ParsedDataLoader(final Data dataLoadedP) {
		this.dataLoaded = dataLoadedP;
		this.dataParsed = new ParsedData();
	}

	public ParsedDataLoader(final ParsedData dataParsedP) {
		this.dataParsed = dataParsedP;
	}

	private void loadPlaylists() {
		MLTimer timer = new MLTimer("loadPlaylists");
		timer.tic();

		int nPlaylists = this.dataLoaded.playlists.length;
		int nSongs = this.dataLoaded.songs.length;

		MLSparseVector[] rows = new MLSparseVector[nPlaylists];
		this.dataParsed.interactions = new MLSparseMatrixAOO(rows, nSongs);

		// init playlist feature matrices
		this.dataParsed.playlistFeatures = new HashMap<PlaylistFeature, MLSparseFeature>();
		for (PlaylistFeature featureName : PlaylistFeature.values()) {
			MLFeatureTransform[] featTransforms = new MLFeatureTransform[] {
					new MLFeatureTransform.ColSelectorTransform(1_000) };

			MLTextTransform[] textTransforms;
			switch (featureName) {
				case NAME_TOKENIZED: {
					// tokenize playlist name
					textTransforms = new MLTextTransform[] {
							new MLTextTransform.LuceneAnalyzerTextTransform(
									new MLTextTransform.DefaultAnalyzer()) };
					break;
				}
				default: {
					textTransforms = null;
					break;
				}
			}

			MLSparseFeature feature = new MLSparseFeature(nPlaylists,
					textTransforms, featTransforms, MLSparseMatrixAOO.class);
			this.dataParsed.playlistFeatures.put(featureName, feature);
		}
		timer.toc("init done");

		// load playlists

		AtomicInteger count = new AtomicInteger(0);
		this.dataParsed.testIndexes = this.dataLoaded.testIndexes;
		this.dataParsed.playlistIds = new String[nPlaylists];
		// IntStream.range(0, nPlaylists).parallel()(i -> {
		for (int i = 0; i < nPlaylists; i++) {
			Playlist playlist = this.dataLoaded.playlists[i];
			this.dataParsed.playlistIds[i] = playlist.get_pid();

			Track[] tracks = playlist.getTracks();

			// convert playlist to sparse matrix
			if (tracks != null && tracks.length > 0) {
				Map<Integer, MLMatrixElement> elementMap = new HashMap<Integer, MLMatrixElement>();
				for (int j = 0; j < tracks.length; j++) {
					MLMatrixElement element = elementMap
							.get(tracks[j].getSongIndex());
					if (element == null) {
						// set date to position in the playlist
						element = new MLMatrixElement(i,
								tracks[j].getSongIndex(), 1.0f,
								tracks[j].getSongPos());
						elementMap.put(tracks[j].getSongIndex(), element);
					} else {
						// some playlists have duplicate songs
						element.setValue(element.getValue() + 1.0f);
					}
				}
				MLMatrixElement[] elements = new MLMatrixElement[elementMap
						.size()];
				int curIndex = 0;
				for (MLMatrixElement element : elementMap.values()) {
					elements[curIndex] = element;
					curIndex++;
				}
				Arrays.sort(elements,
						new MLMatrixElement.ColIndexComparator(false));

				int[] indexes = new int[elements.length];
				float[] values = new float[elements.length];
				long[] dates = new long[elements.length];
				for (int j = 0; j < elements.length; j++) {
					indexes[j] = elements[j].getColIndex();
					values[j] = elements[j].getValue();
					dates[j] = elements[j].getDate();
				}
				rows[i] = new MLSparseVector(indexes, values, dates, nSongs);
			}

			// add playlist features
			for (PlaylistFeature featureName : PlaylistFeature.values()) {
				switch (featureName) {
					case NAME_ORIGINAL: {
						if (playlist.get_name() != null) {
							this.dataParsed.playlistFeatures.get(featureName)
									.addRow(i, playlist.get_name());
						}
						break;
					}

					case NAME_REGEXED: {
						if (playlist.get_name() != null) {
							String name = playlist.get_name();
							name = name.toLowerCase();
							name = name.replaceAll("\\p{Punct}", " ");
							name = name.replaceAll("\\s+", " ").trim();
							this.dataParsed.playlistFeatures.get(featureName)
									.addRow(i, name);
						}
						break;
					}

					case NAME_TOKENIZED: {
						if (playlist.get_name() != null) {
							// convert emojis to string
							String name = playlist.get_name();
							this.dataParsed.playlistFeatures.get(featureName)
									.addRow(i, name);
						}
						break;
					}

					case N_TRACKS: {
						if (playlist.get_num_tracks() != null) {
							this.dataParsed.playlistFeatures.get(featureName)
									.addRow(i, new MLSparseVector(
											new int[] { 0 },
											new float[] {
													playlist.get_num_tracks() },
											null, 1));
						}
						break;
					}

					// case IS_COLLABORATIVE: {
					// int collab = 0;
					// if (playlist.get_collaborative() == true) {
					// collab = 1;
					// }
					// this.dataParsed.playlistFeatures.get(featureName)
					// .addRow(i, new MLSparseVector(new int[] { 0 },
					// new float[] { collab }, null, 1));
					// break;
					// }
					//
					// case MODIFIED_AT: {
					// this.dataParsed.playlistFeatures.get(featureName)
					// .addRow(i, new MLSparseVector(new int[] { 0 },
					// new float[] { TimeUnit.MILLISECONDS
					// .toHours(playlist
					// .get_modified_at()) },
					// null, 1));
					// break;
					// }
					//
					// case N_FOLLOWERS: {
					// this.dataParsed.playlistFeatures.get(featureName)
					// .addRow(i, new MLSparseVector(new int[] { 0 },
					// new float[] {
					// playlist.get_num_followers() },
					// null, 1));
					// break;
					// }
					//
					// case N_EDITS: {
					// this.dataParsed.playlistFeatures.get(featureName)
					// .addRow(i, new MLSparseVector(new int[] { 0 },
					// new float[] {
					// playlist.get_num_edits() },
					// null, 1));
					// break;
					// }
				}
			}

			int curCount = count.incrementAndGet();
			if (curCount % 100_000 == 0) {
				timer.tocLoop(curCount);
			}
			// });
		}
		timer.tocLoop(count.get());

		for (PlaylistFeature featureName : PlaylistFeature.values()) {
			// finalize feature, apply transforms but preserve original data
			this.dataParsed.playlistFeatures.get(featureName)
					.finalizeFeature(true);
		}
	}

	public void loadSongs() {
		MLTimer timer = new MLTimer("loadSongs");
		timer.tic();
		int nSongs = this.dataLoaded.songs.length;

		// init song feature matrices
		this.dataParsed.songFeatures = new HashMap<SongFeature, MLSparseFeature>();
		for (SongFeature featureName : SongFeature.values()) {
			MLFeatureTransform[] featTransforms = new MLFeatureTransform[] {
					new MLFeatureTransform.ColSelectorTransform(1_000) };

			MLTextTransform[] textTransforms;
			switch (featureName) {
				case TRACK_NAME: {
					// tokenize song name
					textTransforms = new MLTextTransform[] {
							new MLTextTransform.LuceneAnalyzerTextTransform(
									new MLTextTransform.DefaultAnalyzer()) };
					break;
				}
				default: {
					textTransforms = null;
					break;
				}
			}

			MLSparseFeature feature = new MLSparseFeature(nSongs,
					textTransforms, featTransforms, MLSparseMatrixAOO.class);
			this.dataParsed.songFeatures.put(featureName, feature);
		}

		AtomicInteger count = new AtomicInteger(0);
		this.dataParsed.songIds = new String[nSongs];
		// IntStream.range(0, nSongs).parallel()(i -> {
		for (int i = 0; i < nSongs; i++) {
			Song song = this.dataLoaded.songs[i];
			this.dataParsed.songIds[i] = song.get_track_uri();

			// add song features
			for (SongFeature featureName : SongFeature.values()) {
				switch (featureName) {
					case ARTIST_ID: {
						this.dataParsed.songFeatures.get(featureName).addRow(i,
								new String[] { song.get_artist_uri() });
						break;
					}

					case ALBUM_ID: {
						this.dataParsed.songFeatures.get(featureName).addRow(i,
								new String[] { song.get_album_uri() });
						break;
					}

					case TRACK_NAME: {
						this.dataParsed.songFeatures.get(featureName).addRow(i,
								song.get_track_name());
						break;
					}

					case DURATION: {
						this.dataParsed.songFeatures.get(featureName).addRow(i,
								new MLSparseVector(new int[] { 0 },
										new float[] { TimeUnit.MILLISECONDS
												.toSeconds(song
														.get_duration_ms()) },
										null, 1));
						break;
					}
				}
			}

			int cur = count.incrementAndGet();
			if (cur % 100_000 == 0) {
				timer.tocLoop(cur);
			}
		}
		// });
		timer.tocLoop(count.get());

		for (SongFeature featureName : SongFeature.values()) {
			// finalize feature, apply transforms but preserve original data
			this.dataParsed.songFeatures.get(featureName).finalizeFeature(true);
		}

	}

	public void loadSongExtraInfo(final String inFile) throws Exception {
		MLTimer timer = new MLTimer("loadSongExtraInfo");
		timer.tic();

		Map<String, Integer> songToIndexMap = new HashMap<String, Integer>();
		for (int i = 0; i < this.dataParsed.songIds.length; i++) {
			songToIndexMap.put(this.dataParsed.songIds[i], i);
		}

		this.dataParsed.songExtraInfoFeatures = new HashMap<SongExtraInfoFeature, MLSparseFeature>();
		for (SongExtraInfoFeature featureName : SongExtraInfoFeature.values()) {
			MLFeatureTransform[] featTransforms = new MLFeatureTransform[] {
					new MLFeatureTransform.ColSelectorTransform(1_000) };

			MLSparseFeature feature = new MLSparseFeature(
					this.dataParsed.songIds.length, null, featTransforms,
					MLSparseMatrixAOO.class);
			this.dataParsed.songExtraInfoFeatures.put(featureName, feature);
		}

		JSONParser parser = new JSONParser(JSONParser.USE_INTEGER_STORAGE);
		AtomicInteger count = new AtomicInteger(0);
		try (BufferedReader reader = new BufferedReader(
				new FileReader(inFile))) {
			JSONArray parsed = (JSONArray) parser.parse(reader);
			for (Object element : parsed) {
				if (element == null
						|| ((JSONObject) element).containsKey("uri") == false) {
					continue;
				}

				String songId = ((JSONObject) element).getAsString("uri");
				int songIndex = songToIndexMap.get(songId);

				int cur = count.incrementAndGet();
				if (cur % 100_000 == 0) {
					timer.tocLoop(cur);
				}

				for (SongExtraInfoFeature feature : SongExtraInfoFeature
						.values()) {
					if (((JSONObject) element)
							.containsKey(feature.name()) == false
							|| ((JSONObject) element)
									.get(feature.name()) == null) {
						continue;
					}

					if (feature.equals(SongExtraInfoFeature.key) == true) {
						this.dataParsed.songExtraInfoFeatures.get(feature)
								.addRow(songIndex,
										((JSONObject) element)
												.getAsNumber(feature.name())
												.intValue() + "");

					} else {
						float value = ((JSONObject) element)
								.getAsNumber(feature.name()).floatValue();
						this.dataParsed.songExtraInfoFeatures.get(feature)
								.addRow(songIndex,
										new MLSparseVector(new int[] { 0 },
												new float[] { value }, null,
												1));
					}
				}
			}
		}
		timer.tocLoop(count.get());

		for (SongExtraInfoFeature featureName : SongExtraInfoFeature.values()) {
			// finalize feature, apply transforms but preserve original data
			this.dataParsed.songExtraInfoFeatures.get(featureName)
					.finalizeFeature(true);
		}
	}

}
