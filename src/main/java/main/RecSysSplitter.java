package main;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import common.MLMatrixElement;
import common.MLRandomUtils;
import common.MLSparseFeature;
import common.MLSparseMatrix;
import common.MLSparseMatrixAOO;
import common.MLSparseVector;
import common.MLTimer;
import common.SplitterCF;
import main.ParsedData.PlaylistFeature;

public class RecSysSplitter {

	// Predict tracks for a playlist given its title only
	// Predict tracks for a playlist given its title and the first track
	// Predict tracks for a playlist given its title and the first 5 tracks
	// Predict tracks for a playlist given its first 5 tracks (no title)
	// Predict tracks for a playlist given its title and the first 10 tracks
	// Predict tracks for a playlist given its first 10 tracks (no title)
	// Predict tracks for a playlist given its title and the first 25 tracks
	// Predict tracks for a playlist given its title and 25 random tracks
	// Predict tracks for a playlist given its title and the first 100 tracks
	// Predict tracks for a playlist given its title and 100 random tracks

	private static MLTimer timer = new MLTimer("RecSysSplitter");

	public static void backFillSplit(final ParsedData data,
			final SplitterCF split, final boolean testOnly) {

		MLSparseMatrix Rtrain = split.getRstrain()
				.get(ParsedData.INTERACTION_KEY);
		MLSparseMatrix Rvalid = split.getRsvalid()
				.get(ParsedData.INTERACTION_KEY);

		System.out.println("before back fill nnz train: " + Rtrain.getNNZ()
				+ "  nnz valid:" + Rvalid.getNNZ() + "  nValidRows:"
				+ split.getValidRowIndexes().length);
		IntStream.range(0, data.interactions.getNRows()).parallel()
				.forEach(rowIndex -> {
					if (Arrays.binarySearch(split.getValidRowIndexes(),
							rowIndex) >= 0) {
						// don't change validation rows
						return;
					}

					MLSparseVector row = data.interactions.getRow(rowIndex);
					if (row != null) {
						// use full data for the rest
						if (testOnly == true) {
							if (Arrays.binarySearch(data.testIndexes,
									rowIndex) >= 0) {
								// only back fill test rows
								Rtrain.setRow(row.deepCopy(), rowIndex);
								Rvalid.setRow(null, rowIndex);
							}
						} else {
							Rtrain.setRow(row.deepCopy(), rowIndex);
							Rvalid.setRow(null, rowIndex);
						}
					}
				});

		System.out.println("after back fill nnz train: " + Rtrain.getNNZ()
				+ "  nnz valid:" + Rvalid.getNNZ());
	}

	public static SplitterCF getSplit(final ParsedData data) {

		Map<String, MLSparseMatrix> temp = new HashMap<String, MLSparseMatrix>();
		temp.put(ParsedData.INTERACTION_KEY, data.interactions);

		SplitterCF split = new SplitterCF();
		split.splitFrac(temp, 0.1f, 5, null, true, 20_000,
				data.interactions.getNCols());

		MLSparseMatrix Rtrain = split.getRstrain()
				.get(ParsedData.INTERACTION_KEY);
		MLSparseMatrix Rvalid = split.getRsvalid()
				.get(ParsedData.INTERACTION_KEY);

		Set<Integer> validRowIndexes = new HashSet<Integer>();
		AtomicInteger nExact = new AtomicInteger(0);
		AtomicInteger counter = new AtomicInteger(0);
		IntStream.range(0, data.testIndexes.length).parallel()
				.forEach(index -> {
					int count = counter.incrementAndGet();
					if (count % 1000 == 0) {
						System.out.println(count + " done");
					}

					int testIndex = data.testIndexes[index];
					if (data.interactions.getRow(testIndex) == null) {
						// skip cold start
						return;
					}

					int nTracksTotal = (int) data.playlistFeatures
							.get(PlaylistFeature.N_TRACKS)
							.getRow(testIndex, false).getValues()[0];
					int nTracksTrain = data.interactions.getRow(testIndex)
							.getIndexes().length;

					// find training playlists with nTracksTotal
					List<Integer> exact = new LinkedList<Integer>();
					for (int i = 0; i < data.interactions.getNRows(); i++) {
						if (Arrays.binarySearch(data.testIndexes, i) >= 0) {
							// don't split test playlists
							continue;
						}

						MLSparseVector row = data.interactions.getRow(i);
						if (row.getIndexes().length == nTracksTotal) {
							exact.add(i);
						}
					}
					Collections.shuffle(exact, new Random(index));

					int repeat = 0;
					while (repeat < 1) {
						Integer validIndex = null;
						synchronized (validRowIndexes) {
							if (validIndex == null && exact.size() > 0) {
								while (exact.size() > 0) {
									int candIndex = exact.remove(0);
									if (validRowIndexes
											.contains(candIndex) == false) {
										validIndex = candIndex;
										nExact.incrementAndGet();
										break;
									}
								}
							}

							if (validIndex == null) {
								// no rows with nTracksTotal
								break;
							}
							validRowIndexes.add(validIndex);
							repeat++;
						}

						// re-split the validIndex row
						MLSparseVector row = data.interactions
								.getRow(validIndex);
						int[] indexes = row.getIndexes();
						float[] values = row.getValues();
						long[] dates = row.getDates();

						Set<Integer> validIndexes = new HashSet<Integer>();
						MLMatrixElement[] elements = new MLMatrixElement[indexes.length];
						for (int i = 0; i < indexes.length; i++) {
							elements[i] = new MLMatrixElement(validIndex,
									indexes[i], values[i], dates[i]);
						}

						if (isInRandomOrder(testIndex) == false) {
							// split by position
							Arrays.sort(elements,
									new MLMatrixElement.DateComparator(true));
						} else {
							// random order split
							MLRandomUtils.shuffle(elements, new Random(index));
						}

						for (int i = 0; i < nTracksTotal - nTracksTrain; i++) {
							validIndexes.add(elements[i].getColIndex());
						}

						int jtrain = 0;
						int[] indexesTrain = new int[nTracksTrain];
						float[] valuesTrain = new float[nTracksTrain];
						long[] datesTrain = new long[nTracksTrain];

						int jvalid = 0;
						int[] indexesValid = new int[nTracksTotal
								- nTracksTrain];
						float[] valuesValid = new float[nTracksTotal
								- nTracksTrain];
						long[] datesValid = new long[nTracksTotal
								- nTracksTrain];

						for (int i = 0; i < indexes.length; i++) {
							if (validIndexes.contains(indexes[i]) == true) {
								indexesValid[jvalid] = indexes[i];
								valuesValid[jvalid] = values[i];
								datesValid[jvalid] = dates[i];
								jvalid++;

							} else {
								indexesTrain[jtrain] = indexes[i];
								valuesTrain[jtrain] = values[i];
								datesTrain[jtrain] = dates[i];
								jtrain++;
							}
						}

						// update split matrices
						Rtrain.setRow(
								new MLSparseVector(indexesTrain, valuesTrain,
										datesTrain, Rtrain.getNCols()),
								validIndex);
						Rvalid.setRow(
								new MLSparseVector(indexesValid, valuesValid,
										datesValid, Rvalid.getNCols()),
								validIndex);
					}
				});

		System.out.println("nExact: " + nExact);

		// update validation row indexes
		int[] validRowIndexesArr = new int[validRowIndexes.size()];
		int cur = 0;
		for (int index : validRowIndexes) {
			validRowIndexesArr[cur] = index;
			cur++;
		}
		Arrays.sort(validRowIndexesArr);
		split.setValidRowIndexes(validRowIndexesArr);

		return split;

	}

	public static SplitterCF getSplitMatching(final ParsedData data,
			final SplitterCF split) {

		// init with full data
		MLSparseVector[] trainRows = new MLSparseVector[data.interactions
				.getNRows()];
		MLSparseVector[] validRows = new MLSparseVector[data.interactions
				.getNRows()];
		for (int i = 0; i < data.interactions.getNRows(); i++) {
			MLSparseVector row = data.interactions.getRow(i);
			if (row != null) {
				trainRows[i] = row.deepCopy();
			}
		}

		// copy validation from existing split
		int[] validRowIndexes = split.getValidRowIndexes();
		for (int i = 0; i < split.getValidRowIndexes().length; i++) {
			int index = validRowIndexes[i];

			trainRows[index] = split.getRstrain()
					.get(ParsedData.INTERACTION_KEY).getRow(index).deepCopy();
			validRows[index] = split.getRsvalid()
					.get(ParsedData.INTERACTION_KEY).getRow(index).deepCopy();
		}

		Set<Integer> addedIndexes = new HashSet<Integer>();
		AtomicInteger nExact = new AtomicInteger(0);
		AtomicInteger nAtLeast = new AtomicInteger(0);
		AtomicInteger counter = new AtomicInteger(0);
		IntStream.range(0, data.testIndexes.length).parallel()
				.forEach(index -> {
					int count = counter.incrementAndGet();
					if (count % 1000 == 0) {
						timer.tocLoop(count);
					}

					int testIndex = data.testIndexes[index];
					if (data.interactions.getRow(testIndex) == null) {
						// skip cold start
						return;
					}

					int nTracksTotal = (int) data.playlistFeatures
							.get(PlaylistFeature.N_TRACKS)
							.getRow(testIndex, false).getValues()[0];
					int nTracksTrain = data.interactions.getRow(testIndex)
							.getIndexes().length;

					// find training playlists with nTracksTotal
					List<Integer> exact = new LinkedList<Integer>();
					List<Integer> atLeast = new LinkedList<Integer>();
					for (int i = 0; i < data.interactions.getNRows(); i++) {
						if (Arrays.binarySearch(data.testIndexes, i) >= 0) {
							// don't split test playlists
							continue;
						}

						if (Arrays.binarySearch(validRowIndexes, i) >= 0) {
							// don't split target valid playlists
							continue;
						}

						MLSparseVector row = data.interactions.getRow(i);
						if (row.getIndexes().length == nTracksTotal) {
							exact.add(i);

						} else if (row.getIndexes().length > nTracksTotal) {
							atLeast.add(i);
						}
					}
					Collections.shuffle(exact, new Random(index));
					Collections.shuffle(atLeast, new Random(index));

					int repeat = 0;
					while (repeat < 10) {
						Integer validIndex = null;
						synchronized (addedIndexes) {
							if (validIndex == null && exact.size() > 0) {
								while (exact.size() > 0) {
									int candIndex = exact.remove(0);
									if (addedIndexes
											.contains(candIndex) == false) {
										validIndex = candIndex;
										nExact.incrementAndGet();
										break;
									}
								}
							}

							if (validIndex == null && atLeast.size() > 0) {
								while (atLeast.size() > 0) {
									int candIndex = atLeast.remove(0);
									if (addedIndexes
											.contains(candIndex) == false) {
										validIndex = candIndex;
										nAtLeast.incrementAndGet();
										break;
									}
								}
							}

							if (validIndex == null) {
								// no rows with nTracksTotal
								break;
							}
							addedIndexes.add(validIndex);
							repeat++;
						}

						// split this row
						MLSparseVector row = data.interactions
								.getRow(validIndex);
						int[] indexes = row.getIndexes();
						float[] values = row.getValues();
						long[] dates = row.getDates();

						MLMatrixElement[] elements = new MLMatrixElement[indexes.length];
						for (int i = 0; i < indexes.length; i++) {
							elements[i] = new MLMatrixElement(validIndex,
									indexes[i], values[i], dates[i]);
						}

						if (isInRandomOrder(testIndex) == false) {
							// split by position
							Arrays.sort(elements,
									new MLMatrixElement.DateComparator(true));
						} else {
							// random order split
							MLRandomUtils.shuffle(elements, new Random(index));
						}

						Set<Integer> trainIndexes = new HashSet<Integer>();
						for (int i = 0; i < nTracksTrain; i++) {
							trainIndexes.add(elements[i].getColIndex());
						}

						int jtrain = 0;
						int[] indexesTrain = new int[nTracksTrain];
						float[] valuesTrain = new float[nTracksTrain];
						long[] datesTrain = new long[nTracksTrain];

						int jvalid = 0;
						int[] indexesValid = new int[indexes.length
								- nTracksTrain];
						float[] valuesValid = new float[indexes.length
								- nTracksTrain];
						long[] datesValid = new long[indexes.length
								- nTracksTrain];

						for (int i = 0; i < indexes.length; i++) {
							if (trainIndexes.contains(indexes[i]) == true) {
								indexesTrain[jtrain] = indexes[i];
								valuesTrain[jtrain] = values[i];
								datesTrain[jtrain] = dates[i];
								jtrain++;

							} else {
								indexesValid[jvalid] = indexes[i];
								valuesValid[jvalid] = values[i];
								datesValid[jvalid] = dates[i];
								jvalid++;
							}
						}

						// update split rows
						trainRows[validIndex] = new MLSparseVector(indexesTrain,
								valuesTrain, datesTrain,
								data.interactions.getNCols());
						validRows[validIndex] = new MLSparseVector(indexesValid,
								valuesValid, datesValid,
								data.interactions.getNCols());
					}
				});

		MLSparseMatrix Rtrain = new MLSparseMatrixAOO(trainRows,
				data.interactions.getNCols());
		Map<String, MLSparseMatrix> trainMap = new HashMap<String, MLSparseMatrix>();
		trainMap.put(ParsedData.INTERACTION_KEY, Rtrain);

		MLSparseMatrix Rvalid = new MLSparseMatrixAOO(validRows,
				data.interactions.getNCols());
		Map<String, MLSparseMatrix> validMap = new HashMap<String, MLSparseMatrix>();
		validMap.put(ParsedData.INTERACTION_KEY, Rvalid);

		SplitterCF newSplit = new SplitterCF();
		newSplit.setRstrain(trainMap);
		newSplit.setRsvalid(validMap);
		newSplit.setValidRowIndexes(validRowIndexes);
		newSplit.setValidColIndexes(split.getValidColIndexes());

		System.out.println("nExact: " + nExact + "  nAtLeast: " + nAtLeast);
		System.out.println(
				"nnz full:" + data.interactions.getNNZ() + "  nnz train: "
						+ Rtrain.getNNZ() + "  nnz valid:" + Rvalid.getNNZ());

		return newSplit;

	}

	public static void removeName(final ParsedData data,
			final SplitterCF split) {
		// Predict tracks for a playlist given its title and the first 5 tracks
		// Predict tracks for a playlist given its first 5 tracks (no title)
		// Predict tracks for a playlist given its title and the first 10 tracks
		// Predict tracks for a playlist given its first 10 tracks (no title)

		List<Integer> fiveTracksValid = new LinkedList<Integer>();
		List<Integer> fiveTracks = new LinkedList<Integer>();

		List<Integer> tenTracksValid = new LinkedList<Integer>();
		List<Integer> tenTracks = new LinkedList<Integer>();

		int[] validRowsIndexes = split.getValidRowIndexes();
		MLSparseMatrix Rtrain = split.getRstrain()
				.get(ParsedData.INTERACTION_KEY);
		MLSparseMatrix Rvalid = split.getRsvalid()
				.get(ParsedData.INTERACTION_KEY);
		for (int i = 0; i < Rtrain.getNRows(); i++) {
			if (Rvalid.getRow(i) == null) {
				continue;
			}

			MLSparseVector row = Rtrain.getRow(i);
			int[] indexes = row.getIndexes();

			if (indexes.length == 5) {
				if (Arrays.binarySearch(validRowsIndexes, i) >= 0) {
					fiveTracksValid.add(i);
				} else {
					fiveTracks.add(i);
				}

			} else if (indexes.length == 10) {
				if (Arrays.binarySearch(validRowsIndexes, i) >= 0) {
					tenTracksValid.add(i);
				} else {
					tenTracks.add(i);
				}
			}
		}

		Collections.shuffle(fiveTracksValid, new Random(0));
		Collections.shuffle(fiveTracks, new Random(1));
		Collections.shuffle(tenTracksValid, new Random(2));
		Collections.shuffle(tenTracks, new Random(3));

		System.out.println("5valid:" + fiveTracksValid.size() + "  5other:"
				+ fiveTracks.size() + "  10valid:" + tenTracksValid.size()
				+ "  10other:" + tenTracks.size());

		// remove names for half of target playlists
		MLSparseFeature nameFeature = data.playlistFeatures
				.get(PlaylistFeature.NAME_REGEXED);

		for (int i = 0; i < fiveTracksValid.size() / 2; i++) {
			nameFeature.getFeatMatrix().setRow(null, fiveTracksValid.get(i));
			nameFeature.getFeatMatrixTransformed().setRow(null,
					fiveTracksValid.get(i));
		}

		for (int i = 0; i < fiveTracks.size() / 2; i++) {
			nameFeature.getFeatMatrix().setRow(null, fiveTracks.get(i));
			nameFeature.getFeatMatrixTransformed().setRow(null,
					fiveTracks.get(i));
		}

		for (int i = 0; i < tenTracksValid.size() / 2; i++) {
			nameFeature.getFeatMatrix().setRow(null, tenTracksValid.get(i));
			nameFeature.getFeatMatrixTransformed().setRow(null,
					tenTracksValid.get(i));
		}

		for (int i = 0; i < tenTracks.size() / 2; i++) {
			nameFeature.getFeatMatrix().setRow(null, tenTracks.get(i));
			nameFeature.getFeatMatrixTransformed().setRow(null,
					tenTracks.get(i));
		}

		System.out.println("5valid:" + (fiveTracksValid.size() / 2)
				+ "  5other:" + (fiveTracks.size() / 2) + "  10valid:"
				+ (tenTracksValid.size() / 2) + "  10other:"
				+ (tenTracks.size() / 2));
	}

	public static boolean isInRandomOrder(final int playlistIndex) {
		if ((playlistIndex >= 1006000 && playlistIndex <= 1006999)
				|| (playlistIndex >= 1008000 && playlistIndex <= 1008999)) {
			return true;
		} else {
			return false;
		}
	}

}
