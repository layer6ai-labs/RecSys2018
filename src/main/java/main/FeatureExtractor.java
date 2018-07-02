package main;

import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import common.MLDenseMatrix;
import common.MLDenseVector;
import common.MLFeatureTransform;
import common.MLRandomUtils;
import common.MLSparseFeature;
import common.MLSparseMatrix;
import common.MLSparseMatrixAOO;
import common.MLSparseVector;
import common.MLTimer;
import main.ParsedData.PlaylistFeature;
import main.ParsedData.SongExtraInfoFeature;
import main.ParsedData.SongFeature;

public class FeatureExtractor {

	public static enum PlaylistDerivedFeature {
		SONG_STATS,
		SONG_ARTISTS,
		SONG_ALBUMS
	}

	public static enum SongDerivedFeature {
		PLAYLIST_ARTISTS,
		PLAYLIST_ALBUMS,
		PLAYLIST_NAMES,
		PLAYLIST_STATS
	}

	public static final PlaylistFeature[] PLAYLIST_FEATS_TO_USE = new PlaylistFeature[] {
			PlaylistFeature.N_TRACKS };
	public static final PlaylistDerivedFeature[] PLAYLIST_DERIVED_FEATS_TO_USE = new PlaylistDerivedFeature[] {
			PlaylistDerivedFeature.SONG_STATS,
			PlaylistDerivedFeature.SONG_ALBUMS };

	public static final SongFeature[] SONG_FEATS_TO_USE = new SongFeature[] {
			SongFeature.DURATION };
	public static final SongDerivedFeature[] SONG_DERIVED_FEATS_TO_USE = new SongDerivedFeature[] {
			SongDerivedFeature.PLAYLIST_STATS };

	public static final SongExtraInfoFeature[] SONG_EXTRA_INFO_FEATS_TO_USE = new SongExtraInfoFeature[] {
			SongExtraInfoFeature.acousticness,
			SongExtraInfoFeature.danceability, SongExtraInfoFeature.energy,
			SongExtraInfoFeature.instrumentalness, SongExtraInfoFeature.key, // categorical
			SongExtraInfoFeature.liveness, SongExtraInfoFeature.loudness,
			SongExtraInfoFeature.mode, SongExtraInfoFeature.speechiness,
			SongExtraInfoFeature.tempo, SongExtraInfoFeature.time_signature,
			SongExtraInfoFeature.valence };

	private Map<PlaylistFeature, MLSparseFeature> playlistFeatMap;
	private Map<PlaylistDerivedFeature, MLSparseFeature> playlistDerivedFeatMap;

	private Map<SongFeature, MLSparseFeature> songFeatsMap;
	private Map<SongExtraInfoFeature, MLSparseFeature> songExtraInfoFeatMap;
	private Map<SongDerivedFeature, MLSparseFeature> songDerivedFeatMap;

	private MLSparseMatrix R;
	private MLSparseMatrix Rt;
	private Latents latents;

	private MLSparseMatrix playlistFeats;
	private MLSparseMatrix playlistFeatsDerived;

	private MLSparseMatrix songFeats;
	private MLSparseMatrix songFeatsDerived;
	private MLSparseMatrix songExtraInfoFeats;

	private float[] songStats;
	private float[] artistStats;
	private float[] albumStats;

	private MLTimer timer;

	public FeatureExtractor(
			final Map<PlaylistFeature, MLSparseFeature> playlistFeatsMapP,
			final Map<SongFeature, MLSparseFeature> songFeatsMapP,
			final Map<SongExtraInfoFeature, MLSparseFeature> songExtraInfoFeatMapP,
			final MLSparseMatrix RP, final MLSparseMatrix RtP,
			final Latents latentsP, final MLTimer timerP) throws Exception {
		this.playlistFeatMap = playlistFeatsMapP;
		this.songFeatsMap = songFeatsMapP;
		this.songExtraInfoFeatMap = songExtraInfoFeatMapP;
		this.R = RP;
		this.Rt = RtP;
		this.latents = latentsP;
		this.timer = timerP;

		// pre-compute all caches
		this.computePopularityCache();
		this.computePlaylistDerivedFeats();
		this.computeSongDerivedFeats();

		// concat playlist features
		if (PLAYLIST_FEATS_TO_USE.length > 0) {
			MLSparseMatrix[] temp = new MLSparseMatrix[PLAYLIST_FEATS_TO_USE.length];
			for (int i = 0; i < temp.length; i++) {
				temp[i] = this.playlistFeatMap.get(PLAYLIST_FEATS_TO_USE[i])
						.getFeatMatrixTransformed();
			}
			this.playlistFeats = MLSparseMatrix.concatHorizontal(temp);
			this.timer.toc("playlist feats " + this.playlistFeats.getNCols());
		}

		if (PLAYLIST_DERIVED_FEATS_TO_USE.length > 0) {
			MLSparseMatrix[] temp = new MLSparseMatrix[PLAYLIST_DERIVED_FEATS_TO_USE.length];
			for (int i = 0; i < PLAYLIST_DERIVED_FEATS_TO_USE.length; i++) {
				temp[i] = this.playlistDerivedFeatMap
						.get(PLAYLIST_DERIVED_FEATS_TO_USE[i])
						.getFeatMatrixTransformed();
			}
			this.playlistFeatsDerived = MLSparseMatrix.concatHorizontal(temp);
			this.timer.toc("playlist derived feats "
					+ this.playlistFeatsDerived.getNCols());
		}

		// concant song features
		if (SONG_FEATS_TO_USE.length > 0) {
			MLSparseMatrix[] temp = new MLSparseMatrix[SONG_FEATS_TO_USE.length];
			for (int i = 0; i < temp.length; i++) {
				temp[i] = this.songFeatsMap.get(SONG_FEATS_TO_USE[i])
						.getFeatMatrixTransformed();
			}
			this.songFeats = MLSparseMatrix.concatHorizontal(temp);
			this.timer.toc("song feats " + this.songFeats.getNCols());
		}

		if (SONG_DERIVED_FEATS_TO_USE.length > 0) {
			MLSparseMatrix[] temp = new MLSparseMatrix[SONG_DERIVED_FEATS_TO_USE.length];
			for (int i = 0; i < temp.length; i++) {
				temp[i] = this.songDerivedFeatMap
						.get(SONG_DERIVED_FEATS_TO_USE[i])
						.getFeatMatrixTransformed();
			}
			this.songFeatsDerived = MLSparseMatrix.concatHorizontal(temp);
			this.timer.toc(
					"song derived feats " + this.songFeatsDerived.getNCols());
		}

		if (this.songExtraInfoFeatMap != null) {
			MLSparseMatrix[] temp = new MLSparseMatrix[SONG_EXTRA_INFO_FEATS_TO_USE.length];
			for (int i = 0; i < SONG_EXTRA_INFO_FEATS_TO_USE.length; i++) {
				temp[i] = this.songExtraInfoFeatMap
						.get(SONG_EXTRA_INFO_FEATS_TO_USE[i])
						.getFeatMatrixTransformed();
			}
			this.songExtraInfoFeats = MLSparseMatrix.concatHorizontal(temp);
			this.timer.toc("song extra info feats "
					+ this.songExtraInfoFeats.getNCols());
		}

	}

	private void computePlaylistDerivedFeats() throws Exception {
		this.playlistDerivedFeatMap = new HashMap<PlaylistDerivedFeature, MLSparseFeature>();
		for (PlaylistDerivedFeature featureName : PlaylistDerivedFeature
				.values()) {
			MLFeatureTransform[] featTransforms = new MLFeatureTransform[] {
					new MLFeatureTransform.ColSelectorTransform(500) };

			MLSparseFeature feature = null;
			if (featureName
					.equals(PlaylistDerivedFeature.SONG_ALBUMS) == true) {
				feature = new MLSparseFeature(this.R.getNRows(), null,
						featTransforms, MLSparseMatrixAOO.class,
						this.songFeatsMap.get(SongFeature.ALBUM_ID));

			} else if (featureName
					.equals(PlaylistDerivedFeature.SONG_ARTISTS) == true) {
				feature = new MLSparseFeature(this.R.getNRows(), null,
						featTransforms, MLSparseMatrixAOO.class,
						this.songFeatsMap.get(SongFeature.ARTIST_ID));
			} else {
				feature = new MLSparseFeature(this.R.getNRows(), null,
						featTransforms, MLSparseMatrixAOO.class);
			}

			this.playlistDerivedFeatMap.put(featureName, feature);
		}

		AtomicInteger counter = new AtomicInteger(0);
		IntStream.range(0, this.R.getNRows()).parallel()
				.forEach(playlistIndex -> {
					MLSparseVector playlist = this.R.getRow(playlistIndex);
					if (playlist == null) {
						return;
					}

					int count = counter.incrementAndGet();
					if (count % 500_000 == 0) {
						timer.tocLoop("computePlaylistDerivedFeats", count);
					}

					Integer playlistName = this.getPlaylistName(playlistIndex);

					int[] songIndexes = playlist.getIndexes();

					float[] playlistStats = new float[9];
					Set<Integer> uniqueArtists = new HashSet<Integer>();
					Set<Integer> uniqueAlbums = new HashSet<Integer>();

					String[] artists = new String[songIndexes.length];
					String[] albums = new String[songIndexes.length];
					for (int i = 0; i < songIndexes.length; i++) {
						int songIndex = songIndexes[i];

						int artistIndex = this.getSongArtist(songIndex);
						int albumIndex = this.getSongAlbum(songIndex);
						float duration = this.getSongDuration(songIndex);

						uniqueArtists.add(artistIndex);
						uniqueAlbums.add(albumIndex);

						playlistStats[0] += this.songStats[songIndex];
						playlistStats[1] += this.artistStats[artistIndex];
						playlistStats[2] += this.albumStats[albumIndex];
						playlistStats[3] += duration;
						playlistStats[4] += this.latents.U.getRow(playlistIndex)
								.mult(this.latents.V.getRow(songIndex));
						if (playlistName != null) {
							playlistStats[5] += this.latents.Uname
									.getRow(playlistIndex)
									.mult(this.latents.Vname.getRow(songIndex));
						}

						artists[i] = this.getSongArtistString(songIndex);
						albums[i] = this.getSongAlbumString(songIndex);
					}
					// unique artist and album counts
					playlistStats[6] = uniqueArtists.size();
					playlistStats[7] = uniqueAlbums.size();
					playlistStats[8] = songIndexes.length;

					if (songIndexes.length > 1) {
						for (int i = 0; i < playlistStats.length - 1; i++) {
							playlistStats[i] = playlistStats[i]
									/ songIndexes.length;
						}
					}

					// strore features
					this.playlistDerivedFeatMap
							.get(PlaylistDerivedFeature.SONG_STATS)
							.addRow(playlistIndex,
									new MLDenseVector(playlistStats));

					this.playlistDerivedFeatMap
							.get(PlaylistDerivedFeature.SONG_ARTISTS)
							.addRow(playlistIndex, artists);

					this.playlistDerivedFeatMap
							.get(PlaylistDerivedFeature.SONG_ALBUMS)
							.addRow(playlistIndex, albums);

				});

		// sanity check
		System.out.println(this.playlistDerivedFeatMap
				.get(PlaylistDerivedFeature.SONG_ARTISTS).getCatToIndex().size()
				+ " " + this.songFeatsMap.get(SongFeature.ARTIST_ID)
						.getCatToIndex().size());
		System.out.println(this.playlistDerivedFeatMap
				.get(PlaylistDerivedFeature.SONG_ALBUMS).getCatToIndex().size()
				+ " " + this.songFeatsMap.get(SongFeature.ALBUM_ID)
						.getCatToIndex().size());

		for (PlaylistDerivedFeature featureName : PlaylistDerivedFeature
				.values()) {
			this.playlistDerivedFeatMap.get(featureName).finalizeFeature(false);
			this.timer.toc("PLAYLIST " + featureName + " "
					+ this.playlistDerivedFeatMap.get(featureName)
							.getFeatMatrixTransformed().getNCols());
		}
	}

	private void computePopularityCache() {

		this.songStats = new float[this.R.getNCols()];

		MLSparseFeature artist = this.songFeatsMap.get(SongFeature.ARTIST_ID);
		MLSparseFeature album = this.songFeatsMap.get(SongFeature.ALBUM_ID);
		this.artistStats = new float[artist.getCatToIndex().size()];
		this.albumStats = new float[album.getCatToIndex().size()];

		IntStream.range(0, this.R.getNRows()).parallel().forEach(i -> {
			MLSparseVector row = this.R.getRow(i);
			if (row == null) {
				return;
			}

			int[] indexes = row.getIndexes();
			long[] dates = row.getDates();

			Set<Integer> seenArtists = new HashSet<Integer>();
			Set<Integer> seenAlbums = new HashSet<Integer>();
			for (int j = 0; j < indexes.length; j++) {
				int index = indexes[j];
				long date = dates[j];

				synchronized (this.songStats) {
					this.songStats[index]++;
				}

				// avoids bias towards playlists that have many songs from same
				// artist/album
				int artistIndex = this.getSongArtist(index);
				if (seenArtists.contains(artistIndex) == false) {
					seenArtists.add(artistIndex);
					synchronized (this.artistStats) {
						this.artistStats[artistIndex]++;
					}
				}

				int albumIndex = this.getSongAlbum(index);
				if (seenAlbums.contains(albumIndex) == false) {
					seenAlbums.add(albumIndex);
					synchronized (this.albumStats) {
						this.albumStats[albumIndex]++;
					}
				}
			}
		});
	}

	private void computeSongDerivedFeats() throws Exception {

		this.songDerivedFeatMap = new HashMap<SongDerivedFeature, MLSparseFeature>();
		for (SongDerivedFeature featureName : SongDerivedFeature.values()) {
			MLSparseFeature feature = new MLSparseFeature(this.Rt.getNRows(),
					null, null, MLSparseMatrixAOO.class);
			this.songDerivedFeatMap.put(featureName, feature);
		}

		final int N_PLAYLISTS_TO_USE = 1_000;
		AtomicInteger counter = new AtomicInteger(0);
		IntStream.range(0, this.Rt.getNRows()).parallel().forEach(songIndex -> {
			MLSparseVector song = this.Rt.getRow(songIndex);
			if (song == null) {
				return;
			}

			int count = counter.incrementAndGet();
			if (count % 500_000 == 0) {
				timer.tocLoop("computeSongDerivedFeats", count);
			}

			int[] playlistIndexes = song.getIndexes();

			if (playlistIndexes.length > N_PLAYLISTS_TO_USE) {
				MLRandomUtils.shuffle(playlistIndexes, new Random(songIndex));
			}

			final int N_PLAYLISTS = Math.min(N_PLAYLISTS_TO_USE,
					playlistIndexes.length);
			float[] simPlaylist = new float[3];
			for (int i = 0; i < N_PLAYLISTS; i++) {
				int playlistIndex = playlistIndexes[i];

				Integer playlistName = this.getPlaylistName(playlistIndex);

				simPlaylist[0] += this.getPlayListNTracks(playlistIndex);

				simPlaylist[1] += this.latents.U.getRow(playlistIndex)
						.mult(this.latents.V.getRow(songIndex));

				if (playlistName != null) {
					simPlaylist[2] += this.latents.Uname.getRow(playlistIndex)
							.mult(this.latents.Vname.getRow(songIndex));
				}
			}

			if (playlistIndexes.length > 1) {
				for (int i = 0; i < simPlaylist.length; i++) {
					simPlaylist[i] = simPlaylist[i] / N_PLAYLISTS;
				}
			}
			this.songDerivedFeatMap.get(SongDerivedFeature.PLAYLIST_STATS)
					.addRow(songIndex,
							new MLDenseVector(simPlaylist).toSparse());
		});

		for (SongDerivedFeature featureName : SongDerivedFeature.values()) {
			this.songDerivedFeatMap.get(featureName).finalizeFeature(false);
			this.timer.toc("SONG " + featureName + " " + this.songDerivedFeatMap
					.get(featureName).getFeatMatrixTransformed().getNCols());
		}
	}

	public MLSparseVector extractFeaturesV1(final int playlistIndex,
			final int songIndex, final float[] extraFeats) {

		List<MLSparseVector> feats = new LinkedList<MLSparseVector>();

		// extra input features
		if (extraFeats != null) {
			feats.add(new MLDenseVector(extraFeats).toSparse());
		}

		// song features
		feats.add(this.getSongFeaturesV1(playlistIndex, songIndex));

		// playlist features
		feats.add(this.getPlaylistFeaturesV1(playlistIndex, songIndex));

		// playlist-song features
		feats.add(this.getPlaylistSongFeatsV1(playlistIndex, songIndex));

		if (this.songExtraInfoFeatMap != null) {
			feats.add(this.getPlaylistSongFeatsCreativeV1(playlistIndex,
					songIndex));
		}

		MLSparseVector[] temp = new MLSparseVector[feats.size()];
		feats.toArray(temp);
		return MLSparseVector.concat(temp);
	}

	private MLSparseVector getPlaylistFeaturesV1(final int targetPlaylistIndex,
			final int targetSongIndex) {
		List<MLSparseVector> feats = new LinkedList<MLSparseVector>();

		if (this.playlistFeats != null) {
			feats.add(this.playlistFeats.getRow(targetPlaylistIndex, true));
		}

		if (this.playlistFeatsDerived != null) {
			feats.add(this.playlistFeatsDerived.getRow(targetPlaylistIndex,
					true));
		}

		Integer name = this.getPlaylistName(targetPlaylistIndex);
		if (name == null) {
			name = 0;
		}
		float[] playlistFeats = new float[] { name };
		feats.add(new MLDenseVector(playlistFeats).toSparse());

		MLSparseVector[] temp = new MLSparseVector[feats.size()];
		feats.toArray(temp);
		return MLSparseVector.concat(temp);
	}

	private int getPlayListNTracks(int playlistIndex) {
		return (int) this.playlistFeatMap.get(PlaylistFeature.N_TRACKS)
				.getFeatMatrix().getRow(playlistIndex).getValues()[0];
	}

	private Integer getPlaylistName(int playlistIndex) {
		MLSparseVector row = this.playlistFeatMap
				.get(PlaylistFeature.NAME_REGEXED).getFeatMatrix()
				.getRow(playlistIndex);
		if (row == null) {
			return null;
		} else {
			return row.getIndexes()[0];
		}
	}

	private MLSparseVector getPlaylistSongFeatsV1(final int targetPlaylistIndex,
			final int targetSongIndex) {

		List<MLSparseVector> feats = new LinkedList<MLSparseVector>();

		// similarity between targetSong and songs in targetPlaylist
		int targetSongArtist = this.getSongArtist(targetSongIndex);
		int targetSongAlbum = this.getSongAlbum(targetSongIndex);
		float targetSongDuration = this.getSongDuration(targetSongIndex);
		Integer targetPlaylistName = this.getPlaylistName(targetPlaylistIndex);
		int targetPlaylistNTracks = this
				.getPlayListNTracks(targetPlaylistIndex);

		MLSparseVector playlist = this.R.getRow(targetPlaylistIndex);
		int[] songIndexes = playlist.getIndexes();

		float[] simSong = new float[10];
		for (int songIndex : songIndexes) {

			int artist = this.getSongArtist(songIndex);
			simSong[0] += equalsIndex(targetSongArtist, artist);

			int album = this.getSongAlbum(songIndex);
			simSong[1] += equalsIndex(targetSongAlbum, album);

			simSong[2] += targetSongDuration - this.getSongDuration(songIndex);

			float score = this.latents.V.getRow(targetSongIndex)
					.mult(this.latents.V.getRow(songIndex));
			simSong[3] += score;

			score = this.latents.Vname.getRow(targetSongIndex)
					.mult(this.latents.Vname.getRow(songIndex));
			simSong[4] += score;

			score = this.latents.artist.getRow(targetSongArtist)
					.mult(this.latents.artist.getRow(artist));
			simSong[5] += score;

			score = this.latents.album.getRow(targetSongAlbum)
					.mult(this.latents.album.getRow(album));
			simSong[6] += score;
		}

		if (targetPlaylistName != null) {
			simSong[7] = this.latents.Uname.getRow(targetPlaylistIndex)
					.mult(this.latents.Vname.getRow(targetSongIndex));
		}

		simSong[8] = this.latents.Ualbum.getRow(targetPlaylistIndex)
				.mult(this.latents.Valbum.getRow(targetSongIndex));

		simSong[9] = this.latents.Uartist.getRow(targetPlaylistIndex)
				.mult(this.latents.Vartist.getRow(targetSongIndex));

		if (songIndexes.length > 1) {
			for (int i = 0; i < simSong.length - 3; i++) {
				simSong[i] = simSong[i] / songIndexes.length;
			}
		}
		feats.add(new MLDenseVector(simSong).toSparse());

		// similarity between targetPlaylist and playlists where targetSong
		// appears
		MLSparseVector song = this.Rt.getRow(targetSongIndex, true);
		int[] playlistIndexes = song.getIndexes().clone();

		final int N_PLAYLISTS_TO_USE = 1_000;

		if (playlistIndexes.length > N_PLAYLISTS_TO_USE) {
			MLRandomUtils.shuffle(playlistIndexes, new Random(targetSongIndex));
		}

		final int N_PLAYLISTS = Math.min(N_PLAYLISTS_TO_USE,
				playlistIndexes.length);

		float[] simPlaylist = new float[4];
		for (int i = 0; i < N_PLAYLISTS; i++) {
			int playlistIndex = playlistIndexes[i];

			Integer playlistName = this.getPlaylistName(playlistIndex);

			simPlaylist[0] += equalsIndex(targetPlaylistName, playlistName);

			simPlaylist[1] += targetPlaylistNTracks
					- this.getPlayListNTracks(playlistIndex);

			float score = this.latents.U.getRow(targetPlaylistIndex)
					.mult(this.latents.U.getRow(playlistIndex));
			simPlaylist[2] += score;

			if (targetPlaylistName != null && playlistName != null) {
				score = this.latents.Uname.getRow(targetPlaylistIndex)
						.mult(this.latents.Uname.getRow(playlistIndex));
				simPlaylist[3] += score;
			}
		}

		if (playlistIndexes.length > 1) {
			for (int i = 0; i < simPlaylist.length; i++) {
				simPlaylist[i] = simPlaylist[i] / N_PLAYLISTS;
			}
		}
		feats.add(new MLDenseVector(simPlaylist).toSparse());

		MLSparseVector[] temp = new MLSparseVector[feats.size()];
		feats.toArray(temp);
		return MLSparseVector.concat(temp);
	}

	private MLSparseVector getPlaylistSongFeatsCreativeV1(
			final int targetPlaylistIndex, final int targetSongIndex) {

		List<MLSparseVector> feats = new LinkedList<MLSparseVector>();

		MLSparseVector playlist = this.R.getRow(targetPlaylistIndex);
		int[] songIndexes = playlist.getIndexes();

		float[] simSong = new float[12];
		for (int songIndex : songIndexes) {

			simSong[0] += diffValue(
					this.songExtraInfoFeatMap
							.get(SongExtraInfoFeature.acousticness)
							.getFeatMatrix().getRow(targetSongIndex),
					this.songExtraInfoFeatMap
							.get(SongExtraInfoFeature.acousticness)
							.getFeatMatrix().getRow(songIndex));

			simSong[1] += diffValue(
					this.songExtraInfoFeatMap
							.get(SongExtraInfoFeature.danceability)
							.getFeatMatrix().getRow(targetSongIndex),
					this.songExtraInfoFeatMap
							.get(SongExtraInfoFeature.danceability)
							.getFeatMatrix().getRow(songIndex));

			simSong[2] += diffValue(
					this.songExtraInfoFeatMap.get(SongExtraInfoFeature.energy)
							.getFeatMatrix().getRow(targetSongIndex),
					this.songExtraInfoFeatMap.get(SongExtraInfoFeature.energy)
							.getFeatMatrix().getRow(songIndex));

			simSong[3] += diffValue(
					this.songExtraInfoFeatMap
							.get(SongExtraInfoFeature.instrumentalness)
							.getFeatMatrix().getRow(targetSongIndex),
					this.songExtraInfoFeatMap
							.get(SongExtraInfoFeature.instrumentalness)
							.getFeatMatrix().getRow(songIndex));

			simSong[4] += equalsIndex(
					this.songExtraInfoFeatMap.get(SongExtraInfoFeature.key)
							.getFeatMatrix().getRow(targetSongIndex),
					this.songExtraInfoFeatMap.get(SongExtraInfoFeature.key)
							.getFeatMatrix().getRow(songIndex));

			simSong[5] += diffValue(
					this.songExtraInfoFeatMap.get(SongExtraInfoFeature.liveness)
							.getFeatMatrix().getRow(targetSongIndex),
					this.songExtraInfoFeatMap.get(SongExtraInfoFeature.liveness)
							.getFeatMatrix().getRow(songIndex));

			simSong[6] += diffValue(
					this.songExtraInfoFeatMap.get(SongExtraInfoFeature.loudness)
							.getFeatMatrix().getRow(targetSongIndex),
					this.songExtraInfoFeatMap.get(SongExtraInfoFeature.loudness)
							.getFeatMatrix().getRow(songIndex));

			simSong[7] += diffValue(
					this.songExtraInfoFeatMap.get(SongExtraInfoFeature.mode)
							.getFeatMatrix().getRow(targetSongIndex),
					this.songExtraInfoFeatMap.get(SongExtraInfoFeature.mode)
							.getFeatMatrix().getRow(songIndex));

			simSong[8] += diffValue(
					this.songExtraInfoFeatMap
							.get(SongExtraInfoFeature.speechiness)
							.getFeatMatrix().getRow(targetSongIndex),
					this.songExtraInfoFeatMap
							.get(SongExtraInfoFeature.speechiness)
							.getFeatMatrix().getRow(songIndex));

			simSong[9] += diffValue(
					this.songExtraInfoFeatMap.get(SongExtraInfoFeature.tempo)
							.getFeatMatrix().getRow(targetSongIndex),
					this.songExtraInfoFeatMap.get(SongExtraInfoFeature.tempo)
							.getFeatMatrix().getRow(songIndex));

			simSong[10] += diffValue(
					this.songExtraInfoFeatMap
							.get(SongExtraInfoFeature.time_signature)
							.getFeatMatrix().getRow(targetSongIndex),
					this.songExtraInfoFeatMap
							.get(SongExtraInfoFeature.time_signature)
							.getFeatMatrix().getRow(songIndex));

			simSong[11] += diffValue(
					this.songExtraInfoFeatMap.get(SongExtraInfoFeature.valence)
							.getFeatMatrix().getRow(targetSongIndex),
					this.songExtraInfoFeatMap.get(SongExtraInfoFeature.valence)
							.getFeatMatrix().getRow(songIndex));
		}

		if (songIndexes.length > 1) {
			for (int i = 0; i < simSong.length; i++) {
				simSong[i] = simSong[i] / songIndexes.length;
			}
		}
		feats.add(new MLDenseVector(simSong).toSparse());

		MLSparseVector[] temp = new MLSparseVector[feats.size()];
		feats.toArray(temp);
		return MLSparseVector.concat(temp);
	}

	private int getSongAlbum(int songIndex) {
		return this.songFeatsMap.get(SongFeature.ALBUM_ID).getFeatMatrix()
				.getRow(songIndex).getIndexes()[0];
	}

	private String getSongAlbumString(int songIndex) {
		int albumIndex = this.getSongAlbum(songIndex);
		return this.songFeatsMap.get(SongFeature.ALBUM_ID).getIndexToCat()
				.get(albumIndex);
	}

	private int getSongArtist(int songIndex) {
		return this.songFeatsMap.get(SongFeature.ARTIST_ID).getFeatMatrix()
				.getRow(songIndex).getIndexes()[0];
	}

	private String getSongArtistString(int songIndex) {
		int artistIndex = this.getSongArtist(songIndex);
		return this.songFeatsMap.get(SongFeature.ARTIST_ID).getIndexToCat()
				.get(artistIndex);
	}

	private float getSongDuration(int songIndex) {
		return this.songFeatsMap.get(SongFeature.DURATION).getFeatMatrix()
				.getRow(songIndex).getValues()[0];
	}

	private MLSparseVector getSongFeaturesV1(final int targetPlaylistIndex,
			final int targetSongIndex) {
		List<MLSparseVector> feats = new LinkedList<MLSparseVector>();

		if (this.songFeats != null) {
			feats.add(this.songFeats.getRow(targetSongIndex, true));
		}

		if (this.songFeatsDerived != null) {
			feats.add(this.songFeatsDerived.getRow(targetSongIndex, true));
		}

		// popularity stats
		int artistIndex = this.getSongArtist(targetSongIndex);
		int albumIndex = this.getSongAlbum(targetSongIndex);
		float[] songFeats = new float[] {

				this.songStats[targetSongIndex],

				this.artistStats[artistIndex],

				this.albumStats[albumIndex],

		};
		feats.add(new MLDenseVector(songFeats).toSparse());

		// extra info, if available
		if (this.songExtraInfoFeats != null) {
			feats.add(this.songExtraInfoFeats.getRow(targetSongIndex, true));
		}

		MLSparseVector[] temp = new MLSparseVector[feats.size()];
		feats.toArray(temp);
		return MLSparseVector.concat(temp);
	}

	private static float equalsIndex(final MLSparseVector vector1,
			final MLSparseVector vector2) {
		if (vector1 == null || vector2 == null) {
			return 0f;
		}

		if (vector1.getIndexes()[0] == vector2.getIndexes()[0]) {
			return 1f;

		} else {
			return 0f;
		}
	}

	private static float diffValue(final MLSparseVector vector1,
			final MLSparseVector vector2) {
		if (vector1 == null || vector2 == null) {
			return 0f;
		}

		return vector1.getValues()[0] - vector2.getValues()[0];
	}

	private static float equalsIndex(final Integer index1,
			final Integer index2) {
		if (index1 == null || index2 == null) {
			return 0f;
		}

		if (index1.equals(index2) == true) {
			return 1f;

		} else {
			return 0f;
		}
	}

	public static MLDenseVector getRowAvg(final MLDenseMatrix V,
			final int[] rowIndices, final boolean normalize) {
		float[] rowAvg = new float[V.getNCols()];
		int count = 0;
		for (int rowIndex : rowIndices) {
			MLDenseVector row = V.getRow(rowIndex);
			// count++;
			if (row == null || row.sum() == 0) {
				continue;
			}
			count++;

			float[] values = row.getValues();
			for (int i = 0; i < values.length; i++) {
				rowAvg[i] += values[i];
			}
		}

		if (normalize == true && count > 1) {
			for (int i = 0; i < rowAvg.length; i++) {
				rowAvg[i] = rowAvg[i] / count;
			}
		}

		return new MLDenseVector(rowAvg);
	}

	public static MLDenseVector getRowAvg(final MLSparseMatrix R,
			final int[] rowIndices, final boolean normalize) {
		float[] rowAvg = new float[R.getNCols()];
		int count = 0;
		for (int colIndex : rowIndices) {
			MLSparseVector row = R.getRow(colIndex);
			// count++;
			if (row == null) {
				continue;
			}
			count++;

			int[] indexes = row.getIndexes();
			float[] values = row.getValues();

			for (int i = 0; i < indexes.length; i++) {
				rowAvg[indexes[i]] += values[i];
			}
		}

		if (normalize == true && count > 1) {
			for (int i = 0; i < rowAvg.length; i++) {
				rowAvg[i] = rowAvg[i] / count;
			}
		}

		return new MLDenseVector(rowAvg);
	}

}
