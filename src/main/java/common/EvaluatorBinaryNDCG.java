package common;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

public class EvaluatorBinaryNDCG extends EvaluatorCF {

	public EvaluatorBinaryNDCG(int[] evalThreshsP) {
		super(evalThreshsP);
	}

	@Override
	public ResultCF evaluate(final SplitterCF split,
			final String interactionType, final FloatElement[][] preds) {

		double[] ndcg = new double[this.threshs.length];

		int maxThresh = this.getMaxEvalThresh();
		Map<Integer, Integer> threshToIndex = new HashMap<>();
		for (int i = 0; i < this.threshs.length; i++) {
			threshToIndex.put(this.threshs[i] - 1, i);
		}

		int[] validRowIndexes = split.getValidRowIndexes();

		MLSparseMatrix validMatrix = split.getRsvalid().get(interactionType);
		AtomicInteger nTotal = new AtomicInteger(0);
		IntStream.range(0, validRowIndexes.length).parallel().forEach(index -> {

			int rowIndex = validRowIndexes[index];
			MLSparseVector row = validMatrix.getRow(rowIndex);
			FloatElement[] rowPreds = preds[rowIndex];

			if (row == null || rowPreds == null) {
				return;
			}
			nTotal.incrementAndGet();
			int[] targetIndexes = row.getIndexes();

			double dcg = 0;
			double idcg = 0;
			for (int i = 0; i < maxThresh; i++) {
				if (Arrays.binarySearch(targetIndexes,
						rowPreds[i].getIndex()) >= 0) {
					// prediction DCG
					if (i == 0) {
						dcg += 1.0;
					} else {
						dcg += 1.0 / log2(i + 2.0);
					}
				}
				if (i < targetIndexes.length) {
					// ideal DCG
					if (i == 0) {
						idcg += 1.0;
					} else {
						idcg += 1.0 / log2(i + 2.0);
					}
				}

				if (threshToIndex.containsKey(i) == true) {
					synchronized (ndcg) {
						ndcg[threshToIndex.get(i)] += dcg / idcg;
					}
				}
			}
		});

		int nEval = nTotal.get();
		for (int i = 0; i < ndcg.length; i++) {
			ndcg[i] /= nTotal.get();
		}
		return new ResultCF("b-ndcg", ndcg, nEval);
	}

	private static double log2(final double in) {
		return Math.log(in) / Math.log(2.0);
	}

}
