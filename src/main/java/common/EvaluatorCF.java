package common;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

public abstract class EvaluatorCF {

	protected int[] threshs;

	public EvaluatorCF(final int[] threshsP) {
		this.threshs = threshsP;
	}

	public abstract ResultCF evaluate(final SplitterCF split,
			final String interactionType, final FloatElement[][] preds);

	public int[] getEvalThreshs() {
		return this.threshs;
	}

	public int getMaxEvalThresh() {
		return this.threshs[this.threshs.length - 1];
	}

	public static FloatElement[][] getRankings(SplitterCF split,
			MLDenseMatrix U, MLDenseMatrix V, final int maxThresh,
			final String interactionType) {
		if (Math.floorDiv(LowLevelRoutines.MAX_ARRAY_SIZE, V.getNRows()) >= V
				.getNCols()) {
			return getRankingsNative(split.getRstrain().get(interactionType),
					split.getValidRowIndexes(), split.getValidColIndexes(), U,
					V, maxThresh, 100);
		} else {
			System.out.printf(
					"[WARNING] using non-native ranking can be very slow");
			return getRankingsNonNative(split, U, V, maxThresh,
					interactionType);
		}
	}

	public static FloatElement[][] getRankingsNative(
			final MLSparseMatrix Rtrain, final int[] rowIndexes,
			int[] colIndexes, final MLDenseMatrix U, final MLDenseMatrix V,
			final int rankingSize, final int rowBatchSize) {

		// convenience function
		float[] Vflat = V.slice(colIndexes).toFlatArray();
		return getRankingsNative(Rtrain, rowIndexes, colIndexes, U, Vflat,
				rankingSize, rowBatchSize);
	}

	public static FloatElement[][] getRankingsNative(
			final MLSparseMatrix Rtrain, final int[] rowIndexes,
			final int[] colIndexes, final MLDenseMatrix U, final float[] V,
			final int rankingSize, final int rowBatchSize) {

		FloatElement[][] rankings = new FloatElement[U.getNRows()][];
		final int nRowsV = colIndexes.length;
		final int nCols = U.getNCols();

		final Map<Integer, Integer> colMap = new HashMap<Integer, Integer>();
		for (int i = 0; i < colIndexes.length; i++) {
			colMap.put(colIndexes[i], i);
		}

		final int uBatchSize = Math.min(rowBatchSize,
				Math.floorDiv(LowLevelRoutines.MAX_ARRAY_SIZE, nRowsV));
		int nBatch = -Math.floorDiv(-rowIndexes.length, uBatchSize);

		for (int batch = 0; batch < nBatch; batch++) {
			final int start = batch * uBatchSize;
			final int end = Math.min(start + uBatchSize, rowIndexes.length);

			final float[] result = new float[(end - start) * nRowsV];
			MLDenseMatrix uBatchRows = U.slice(rowIndexes, start, end);
			LowLevelRoutines.sgemm(uBatchRows.toFlatArray(), V, result,
					(end - start), nRowsV, nCols, true, false, 1, 0);

			IntStream.range(0, end - start).parallel().forEach(i -> {
				int rowIndex = rowIndexes[start + i];
				MLSparseVector trainRow = null;
				if (Rtrain != null) {
					trainRow = Rtrain.getRow(rowIndex);
				}
				// map training index to relative index to match sliced V
				FloatElement[] preds;
				int[] excludes = null;
				if (trainRow != null) {
					excludes = Arrays.stream(trainRow.getIndexes())
							.filter(colMap::containsKey).map(colMap::get)
							.toArray();
					if (excludes.length == 0) {
						excludes = null;
					}
				}
				preds = FloatElement.topNSortOffset(result, rankingSize,
						excludes, i * nRowsV, nRowsV);

				if (preds != null) {
					// map back to full index
					for (int j = 0; j < preds.length; j++) {
						preds[j].setIndex(colIndexes[preds[j].getIndex()]);
					}
				}
				rankings[rowIndex] = preds;
			});
		}
		return rankings;
	}

	private static FloatElement[][] getRankingsNonNative(final SplitterCF split,
			final MLDenseMatrix U, final MLDenseMatrix V, final int maxThresh,
			final String interactionType) {

		MLSparseMatrix R_train = split.getRstrain().get(interactionType);
		FloatElement[][] rankings = new FloatElement[R_train.getNRows()][];
		int[] validRowIndexes = split.getValidRowIndexes();
		int[] validColIndexes = split.getValidColIndexes();
		AtomicInteger count = new AtomicInteger(0);
		MLTimer evalTimer = new MLTimer("ALS Eval", validRowIndexes.length);

		IntStream.range(0, validRowIndexes.length).parallel().forEach(index -> {
			final int countLocal = count.incrementAndGet();
			if (countLocal % 1000 == 0) {
				evalTimer.tocLoop(countLocal);
			}
			int rowIndex = validRowIndexes[index];

			MLDenseVector uRow = U.getRow(rowIndex);
			FloatElement[] rowScores = new FloatElement[validColIndexes.length];
			int cur = 0;
			for (int colIndex : validColIndexes) {
				rowScores[cur] = new FloatElement(colIndex,
						uRow.mult(V.getRow(colIndex)));
				cur++;
			}

			MLSparseVector trainRow = R_train.getRow(rowIndex);
			if (trainRow != null) {
				rankings[rowIndex] = FloatElement.topNSortArr(rowScores,
						maxThresh, R_train.getRow(rowIndex).getIndexes());
			} else {
				rankings[rowIndex] = FloatElement.topNSort(rowScores, maxThresh,
						null);
			}
		});
		return rankings;
	}

}
