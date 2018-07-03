package common;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

public class EvaluatorRPrecision extends EvaluatorCF {

	public EvaluatorRPrecision(int[] evalThreshsP) {
		super(evalThreshsP);
	}

	@Override
	public ResultCF evaluate(final SplitterCF split,
			final String interactionType, final FloatElement[][] preds) {

		double[] rPrecision = new double[] { 0.0 };
		MLSparseMatrix validMatrix = split.getRsvalid().get(interactionType);
		AtomicInteger nTotal = new AtomicInteger(0);
		IntStream.range(0, validMatrix.getNRows()).parallel()
				.forEach(rowIndex -> {
					MLSparseVector row = validMatrix.getRow(rowIndex);
					FloatElement[] rowPreds = preds[rowIndex];

					if (row == null || rowPreds == null) {
						return;
					}
					nTotal.incrementAndGet();

					double nMatched = 0;
					int[] targetIndexes = row.getIndexes();
					for (int i = 0; i < Math.min(targetIndexes.length,
							rowPreds.length); i++) {
						if (Arrays.binarySearch(targetIndexes,
								rowPreds[i].getIndex()) >= 0) {
							nMatched++;
						}
					}
					synchronized (rPrecision) {
						rPrecision[0] += nMatched / Math
								.min(targetIndexes.length, rowPreds.length);
					}
				});

		rPrecision[0] = rPrecision[0] / nTotal.get();
		return new ResultCF("r-precision", rPrecision, nTotal.get());
	}
}
