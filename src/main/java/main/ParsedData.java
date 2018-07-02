package main;

import java.io.Serializable;
import java.util.Map;

import common.MLSparseFeature;
import common.MLSparseMatrix;

public class ParsedData implements Serializable {

	public static final String INTERACTION_KEY = "INT";

	public enum PlaylistFeature implements Serializable {
		NAME_TOKENIZED,
		NAME_REGEXED,
		NAME_ORIGINAL,
		N_TRACKS,
	}

	public enum SongFeature implements Serializable {
		ARTIST_ID,
		ALBUM_ID,
		TRACK_NAME,
		DURATION;
	}

	public enum SongExtraInfoFeature implements Serializable {
		acousticness,
		danceability,
		energy,
		instrumentalness,
		key, // categorical
		liveness,
		loudness,
		mode,
		speechiness,
		tempo,
		time_signature,
		valence
	}

	private static final long serialVersionUID = 736424464160763130L;

	public String[] songIds;
	public String[] playlistIds;
	public int[] testIndexes;

	public Map<PlaylistFeature, MLSparseFeature> playlistFeatures;
	public Map<SongFeature, MLSparseFeature> songFeatures;
	public Map<SongExtraInfoFeature, MLSparseFeature> songExtraInfoFeatures;
	public MLSparseMatrix interactions;

}
