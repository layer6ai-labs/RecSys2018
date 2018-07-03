package common;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

public class EvaluatorRPrecisionArtist extends EvaluatorCF {
	public MLSparseMatrix songArtist;

	public EvaluatorRPrecisionArtist(int[] evalThreshsP,
			MLSparseMatrix songArtist) {
		super(evalThreshsP);
		this.songArtist = songArtist;
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

					// get indexes of all artists of the target songs
					Set<Integer> artistIndexes = new HashSet<Integer>();
					for (int songIndex : targetIndexes) {

						MLSparseVector artist = songArtist.getRow(songIndex);
						if (artist == null) {
							continue;
						}

						for (int artistIndex : artist.getIndexes()) {
							artistIndexes.add(artistIndex);
						}
					}

					// set of artist Indexes that's already matched, since it
					// only counts once
					Set<Integer> artistIndexes_already_matched = new HashSet<Integer>();

					for (int i = 0; i < Math.min(targetIndexes.length,
							rowPreds.length); i++) {
						if (Arrays.binarySearch(targetIndexes,
								rowPreds[i].getIndex()) >= 0) {
							nMatched++;
						} else {
							int artistIndex = songArtist
									.getRow(rowPreds[i].getIndex())
									.getIndexes()[0];

							if (artistIndexes.contains(artistIndex)
									&& (!artistIndexes_already_matched
											.contains(artistIndex))) {
								artistIndexes_already_matched.add(artistIndex);
								nMatched = nMatched + 0.25;
							}
						}
					}
					synchronized (rPrecision) {
						rPrecision[0] += nMatched / Math
								.min(targetIndexes.length, rowPreds.length);
					}
				});

		rPrecision[0] = rPrecision[0] / nTotal.get();
		return new ResultCF("r-precision-artist", rPrecision, nTotal.get());
	}
}
