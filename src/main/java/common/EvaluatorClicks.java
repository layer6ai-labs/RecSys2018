package common;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

public class EvaluatorClicks extends EvaluatorCF {

	public EvaluatorClicks(int[] evalThreshsP) {
		super(evalThreshsP);
	}

	@Override
	public ResultCF evaluate(final SplitterCF split,
			final String interactionType, final FloatElement[][] preds) {

		double[] clicks = new double[] { 0.0 };
		int maxThresh = this.getMaxEvalThresh();
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
					int[] targetIndexes = row.getIndexes();

					int matchIndex = (int) Math.floor(maxThresh / 10.0) + 1;

					for (int i = 0; i < maxThresh; i++) {
						if (Arrays.binarySearch(targetIndexes,
								rowPreds[i].getIndex()) >= 0) {
							matchIndex = (int) Math.floor(i / 10.0);
							break;
						}
					}

					synchronized (clicks) {
						clicks[0] += matchIndex;
					}
				});

		clicks[0] = clicks[0] / nTotal.get();
		return new ResultCF("clicks", clicks, nTotal.get());
	}
}
