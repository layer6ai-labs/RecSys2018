package main;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import common.MLDenseVector;
import common.MLSparseMatrix;
import common.MLSparseMatrixAOO;
import common.MLSparseVector;
import common.MLTimer;
import common.SplitterCF;
import main.ParsedData.PlaylistFeature;
import main.ParsedData.SongFeature;
import main.SVD.SVDParams;

public class SVDModel {

	private static MLTimer timer = new MLTimer("SVDModel");

	private ParsedData data;
	private SplitterCF split;
	private Latents latents;

	public SVDModel(final ParsedData dataP, final SplitterCF splitP,
			final Latents latentsP) {
		this.data = dataP;
		this.split = splitP;
		this.latents = latentsP;
	}

	public void factorizeNames(final String scriptPath,
			final String cachePath) throws Exception {
		MLSparseMatrix playlistNames = this.data.playlistFeatures
				.get(PlaylistFeature.NAME_REGEXED).getFeatMatrix();
		timer.toc("nNames " + playlistNames.getNCols());

		// create name matrix
		MLSparseMatrix Rtrain = this.split.getRstrain()
				.get(ParsedData.INTERACTION_KEY);
		MLSparseMatrix RtrainT = Rtrain.transpose();

		MLSparseVector[] rowsNames = new MLSparseVector[Rtrain.getNCols()
				+ Rtrain.getNRows()];
		AtomicInteger counter = new AtomicInteger(0);
		IntStream.range(0, Rtrain.getNCols()).parallel().forEach(songIndex -> {
			int count = counter.incrementAndGet();
			if (count % 200_000 == 0) {
				timer.tocLoop("songs done", count);
			}

			MLSparseVector song = RtrainT.getRow(songIndex);
			if (song == null) {
				return;
			}

			MLDenseVector rowAvg = getRowAvg(playlistNames, song.getIndexes(),
					false);
			MLSparseVector rowAvgSparse = rowAvg.toSparse();
			if (rowAvgSparse.getIndexes() != null) {
				rowsNames[songIndex] = rowAvgSparse;
			}
		});

		counter.set(0);
		IntStream.range(0, Rtrain.getNRows()).parallel()
				.forEach(playlistIndex -> {
					int count = counter.incrementAndGet();
					if (count % 200_000 == 0) {
						timer.tocLoop("playlists done", count);
					}

					MLSparseVector names = playlistNames.getRow(playlistIndex);
					if (names != null) {
						rowsNames[playlistIndex + Rtrain.getNCols()] = names
								.deepCopy();
					}
				});

		MLSparseMatrix nameMatrix = new MLSparseMatrixAOO(rowsNames,
				playlistNames.getNCols());
		timer.toc("name matrix done " + nameMatrix.getNRows() + " "
				+ nameMatrix.getNCols() + " " + nameMatrix.getNNZ());

		SVDParams svdParams = new SVDParams();
		svdParams.svdIter = 4;
		svdParams.rank = 200;
		svdParams.scriptPath = "/media/mvolkovs/external4TB/Data/recsys2018/models/svd/svd_py.py";
		svdParams.cachePath = "/media/mvolkovs/external4TB/Data/recsys2018/models/svd/";
		svdParams.shapeRows = nameMatrix.getNRows();
		svdParams.shapeCols = nameMatrix.getNCols();
		SVD svd = new SVD(svdParams);
		svd.runPythonSVD(nameMatrix);

		this.latents.Vname = svd.P.slice(0, Rtrain.getNCols());
		this.latents.Uname = svd.P.slice(Rtrain.getNCols(),
				Rtrain.getNCols() + Rtrain.getNRows());
		this.latents.name = svd.Q;
	}

	public void factorizeAlbums(final String scriptPath,
			final String cachePath) throws Exception {
		MLSparseMatrix songAlbums = this.data.songFeatures
				.get(SongFeature.ALBUM_ID).getFeatMatrix();
		timer.toc("nAlbums " + songAlbums.getNCols());

		// create album matrix
		MLSparseMatrix Rtrain = this.split.getRstrain()
				.get(ParsedData.INTERACTION_KEY);

		MLSparseVector[] rowsAlbums = new MLSparseVector[Rtrain.getNCols()
				+ Rtrain.getNRows()];
		AtomicInteger counter = new AtomicInteger(0);
		IntStream.range(0, Rtrain.getNCols()).parallel().forEach(songIndex -> {
			int count = counter.incrementAndGet();
			if (count % 500_000 == 0) {
				timer.tocLoop("songs done", count);
			}

			MLSparseVector albums = songAlbums.getRow(songIndex);
			if (albums != null) {
				rowsAlbums[songIndex] = albums.deepCopy();
			}
		});

		counter.set(0);
		IntStream.range(0, Rtrain.getNRows()).parallel()
				.forEach(playlistIndex -> {
					int count = counter.incrementAndGet();
					if (count % 500_000 == 0) {
						timer.tocLoop("playlists done", count);
					}

					MLSparseVector playlist = Rtrain.getRow(playlistIndex);
					if (playlist == null) {
						return;
					}

					MLDenseVector rowAvg = getRowAvg(songAlbums,
							playlist.getIndexes(), false);
					MLSparseVector rowAvgSparse = rowAvg.toSparse();
					if (rowAvgSparse.getIndexes() != null) {
						rowsAlbums[playlistIndex
								+ Rtrain.getNCols()] = rowAvgSparse;
					}
				});

		MLSparseMatrix albumMatrix = new MLSparseMatrixAOO(rowsAlbums,
				songAlbums.getNCols());
		timer.toc("album matrix done " + albumMatrix.getNRows() + " "
				+ albumMatrix.getNCols() + " " + albumMatrix.getNNZ());

		SVDParams svdParams = new SVDParams();
		svdParams.svdIter = 4;
		svdParams.rank = 200;
		svdParams.scriptPath = "/media/mvolkovs/external4TB/Data/recsys2018/models/svd/svd_py.py";
		svdParams.cachePath = "/media/mvolkovs/external4TB/Data/recsys2018/models/svd/";
		svdParams.shapeRows = albumMatrix.getNRows();
		svdParams.shapeCols = albumMatrix.getNCols();
		SVD svd = new SVD(svdParams);
		svd.runPythonSVD(albumMatrix);

		this.latents.Valbum = svd.P.slice(0, Rtrain.getNCols());
		this.latents.Ualbum = svd.P.slice(Rtrain.getNCols(),
				Rtrain.getNCols() + Rtrain.getNRows());
		this.latents.album = svd.Q;
	}

	public void factorizeArtists(final String scriptPath,
			final String cachePath) throws Exception {
		MLSparseMatrix songArtists = this.data.songFeatures
				.get(SongFeature.ARTIST_ID).getFeatMatrix();
		timer.toc("nArtists " + songArtists.getNCols());

		// create artist matrix
		MLSparseMatrix Rtrain = this.split.getRstrain()
				.get(ParsedData.INTERACTION_KEY);

		MLSparseVector[] rowsArtist = new MLSparseVector[Rtrain.getNCols()
				+ Rtrain.getNRows()];
		AtomicInteger counter = new AtomicInteger(0);
		IntStream.range(0, Rtrain.getNCols()).parallel().forEach(songIndex -> {
			int count = counter.incrementAndGet();
			if (count % 500_000 == 0) {
				timer.tocLoop("songs done", count);
			}

			MLSparseVector artists = songArtists.getRow(songIndex);
			if (artists != null) {
				rowsArtist[songIndex] = artists.deepCopy();
			}
		});

		counter.set(0);
		IntStream.range(0, Rtrain.getNRows()).parallel()
				.forEach(playlistIndex -> {
					int count = counter.incrementAndGet();
					if (count % 500_000 == 0) {
						timer.tocLoop("playlists done", count);
					}

					MLSparseVector playlist = Rtrain.getRow(playlistIndex);
					if (playlist == null) {
						return;
					}

					MLDenseVector rowAvg = getRowAvg(songArtists,
							playlist.getIndexes(), false);
					MLSparseVector rowAvgSparse = rowAvg.toSparse();
					if (rowAvgSparse.getIndexes() != null) {
						rowsArtist[playlistIndex
								+ Rtrain.getNCols()] = rowAvgSparse;
					}
				});

		MLSparseMatrix artistMatrix = new MLSparseMatrixAOO(rowsArtist,
				songArtists.getNCols());
		timer.toc("artist matrix done " + artistMatrix.getNRows() + " "
				+ artistMatrix.getNCols() + " " + artistMatrix.getNNZ());

		SVDParams svdParams = new SVDParams();
		svdParams.svdIter = 4;
		svdParams.rank = 200;
		svdParams.scriptPath = "/media/mvolkovs/external4TB/Data/recsys2018/models/svd/svd_py.py";
		svdParams.cachePath = "/media/mvolkovs/external4TB/Data/recsys2018/models/svd/";
		svdParams.shapeRows = artistMatrix.getNRows();
		svdParams.shapeCols = artistMatrix.getNCols();
		SVD svd = new SVD(svdParams);
		svd.runPythonSVD(artistMatrix);

		this.latents.Vartist = svd.P.slice(0, Rtrain.getNCols());
		this.latents.Uartist = svd.P.slice(Rtrain.getNCols(),
				Rtrain.getNCols() + Rtrain.getNRows());
		this.latents.artist = svd.Q;
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
