package common;

import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

public class ALS {

	public static class ALSParams {
		public int maxIter = 10;
		public int rank = 200;
		public float alpha = 10f;
		public float lambda = 0.01f;
		public float init = 0.01f;
		public int seed = 1;
		public boolean evaluate = true;
		public boolean debug = true;
		public int printFrequency = 500_000;

		@Override
		public String toString() {
			StringBuilder result = new StringBuilder();
			String newLine = System.getProperty("line.separator");

			result.append(this.getClass().getName());
			result.append(" {");
			result.append(newLine);

			// determine fields declared in this class only (no fields of
			// superclass)
			Field[] fields = this.getClass().getDeclaredFields();

			// print field names paired with their values
			for (Field field : fields) {
				result.append("  ");
				try {
					result.append(field.getName());
					result.append(": ");
					// requires access to private field:
					result.append(field.get(this));
				} catch (IllegalAccessException ex) {
					System.out.println(ex);
				}
				result.append(newLine);
			}
			result.append("}");

			return result.toString();
		}

	}

	private ALSParams params;
	private MLDenseMatrix U;
	private MLDenseMatrix V;
	private MLTimer timer;

	public ALS(final ALSParams paramsP) {
		this.params = paramsP;
		this.timer = new MLTimer("als", this.params.maxIter);
	}

	public MLDenseMatrix getU() {
		return this.U;
	}

	public MLDenseMatrix getV() {
		return this.V;
	}

	public void optimize(final MLSparseMatrix R_train, final String outPath)
			throws Exception {

		this.timer.tic();

		MLSparseMatrix R_train_t = R_train.transpose();
		this.timer.toc("obtained R Rt");

		// randomly initialize U and V
		if (this.U == null) {
			this.U = MLDenseMatrix.initRandom(R_train.getNRows(),
					this.params.rank, this.params.init, this.params.seed);
			this.timer.toc("initialized U");
		}

		if (this.V == null) {
			this.V = MLDenseMatrix.initRandom(R_train.getNCols(),
					this.params.rank, this.params.init, this.params.seed);
			this.timer.toc("initialized V");
		}

		for (int i = 0; i < R_train.getNRows(); i++) {
			if (R_train.getRow(i) == null) {
				// zero out cold start users
				this.U.setRow(new MLDenseVector(new float[this.params.rank]),
						i);
			}
		}
		for (int i = 0; i < R_train_t.getNRows(); i++) {
			if (R_train_t.getRow(i) == null) {
				// zero out cold start items
				this.V.setRow(new MLDenseVector(new float[this.params.rank]),
						i);
			}
		}

		for (int iter = 0; iter < this.params.maxIter; iter++) {
			this.solve(R_train, this.U, this.V);
			this.solve(R_train_t, this.V, this.U);
			this.timer.toc("solver done");
			this.timer.toc(String.format("[iter %d] done", iter));
		}

		if (outPath != null) {
			String uOutFile = outPath + "U_" + this.params.rank + ".bin";
			String vOutFile = outPath + "V_" + this.params.rank + ".bin";

			this.U.toFile(uOutFile);
			this.timer.toc("written U to " + uOutFile);

			this.V.toFile(vOutFile);
			this.timer.toc("written V to " + vOutFile);
		}
	}

	public void setU(final MLDenseMatrix Up) {
		this.U = Up;
	}

	public void setV(final MLDenseMatrix Vp) {
		this.V = Vp;
	}

	private MLDenseVector solve(final int targetIndex,
			final MLSparseMatrix data, final float[] H, final float[] HH,
			final float[] cache) {
		int[] rowIndexes = data.getRow(targetIndex).getIndexes();
		float[] values = data.getRow(targetIndex).getValues();

		float[] HC_minus_IH = new float[this.params.rank * this.params.rank];
		for (int i = 0; i < this.params.rank; i++) {
			for (int j = i; j < this.params.rank; j++) {
				float total = 0;
				for (int k = 0; k < rowIndexes.length; k++) {
					int offset = rowIndexes[k] * this.params.rank;
					total += H[offset + i] * H[offset + j] * values[k];
				}
				HC_minus_IH[i * this.params.rank + j] = total
						* this.params.alpha;
				HC_minus_IH[j * this.params.rank + i] = total
						* this.params.alpha;
			}
		}
		// create HCp in O(f|S_u|)
		float[] HCp = new float[this.params.rank];
		for (int i = 0; i < this.params.rank; i++) {
			float total = 0;
			for (int k = 0; k < rowIndexes.length; k++) {
				total += H[rowIndexes[k] * this.params.rank + i]
						* (1 + this.params.alpha * values[k]);
			}
			HCp[i] = total;
		}
		// create temp = HH + HC_minus_IH + lambda*I
		// temp is symmetric
		// the inverse temp is symmetric
		float[] temp = new float[this.params.rank * this.params.rank];
		for (int i = 0; i < this.params.rank; i++) {
			final int offset = i * this.params.rank;
			for (int j = i; j < this.params.rank; j++) {
				float total = HH[offset + j] + HC_minus_IH[offset + j];
				if (i == j) {
					total += this.params.lambda;
				}
				temp[offset + j] = total;
			}
		}

		LowLevelRoutines.symmetricSolve(temp, this.params.rank, HCp, cache);

		// return optimal solution
		return new MLDenseVector(HCp);
	}

	private MLDenseVector solve(final int targetIndex,
			final MLSparseMatrix data, final MLDenseMatrix H, final float[] HH,
			final float[] cache) {
		int[] rowIndexes = data.getRow(targetIndex).getIndexes();
		float[] values = data.getRow(targetIndex).getValues();

		float[] HC_minus_IH = new float[this.params.rank * this.params.rank];
		for (int i = 0; i < this.params.rank; i++) {
			for (int j = i; j < this.params.rank; j++) {
				float total = 0;
				for (int k = 0; k < rowIndexes.length; k++) {
					total += H.getValue(rowIndexes[k], i)
							* H.getValue(rowIndexes[k], j) * values[k];
				}
				HC_minus_IH[i * this.params.rank + j] = total
						* this.params.alpha;
				HC_minus_IH[j * this.params.rank + i] = total
						* this.params.alpha;
			}
		}
		// create HCp in O(f|S_u|)
		float[] HCp = new float[this.params.rank];
		for (int i = 0; i < this.params.rank; i++) {
			float total = 0;
			for (int k = 0; k < rowIndexes.length; k++) {
				total += H.getValue(rowIndexes[k], i)
						* (1 + this.params.alpha * values[k]);
			}
			HCp[i] = total;
		}
		// create temp = HH + HC_minus_IH + lambda*I
		// temp is symmetric
		// the inverse temp is symmetric
		float[] temp = new float[this.params.rank * this.params.rank];
		for (int i = 0; i < this.params.rank; i++) {
			final int offset = i * this.params.rank;
			for (int j = i; j < this.params.rank; j++) {
				float total = HH[offset + j] + HC_minus_IH[offset + j];
				if (i == j) {
					total += this.params.lambda;
				}
				temp[offset + j] = total;
			}
		}

		LowLevelRoutines.symmetricSolve(temp, this.params.rank, HCp, cache);

		// return optimal solution
		return new MLDenseVector(Arrays.copyOf(HCp, this.params.rank));
	}

	private void solve(final MLSparseMatrix data, final MLDenseMatrix W,
			final MLDenseMatrix H) {

		int cacheSize = LowLevelRoutines.symmInverseCacheSize(
				new float[this.params.rank * this.params.rank],
				this.params.rank);
		// float[] cache = new float[cacheSize];
		MLConcurrentUtils.Async<float[]> cache = new MLConcurrentUtils.Async<>(
				() -> new float[cacheSize], null);
		MLTimer timer = new MLTimer("als", data.getNRows());
		timer.tic();

		// compute H_t * H
		MLDenseMatrix HH = H.transposeMultNative();
		float[] HHflat = HH.toFlatArray();
		if (this.params.debug) {
			timer.toc("HH done");
		}

		boolean[] useFlat = new boolean[] { false };
		float[][] Hflat = new float[1][];
		if (H.getNRows() < LowLevelRoutines.MAX_ARRAY_SIZE / H.getNCols()) {
			// no overflow so use flat version
			useFlat[0] = true;
			Hflat[0] = H.toFlatArray();
			if (this.params.debug) {
				timer.toc("H to flat done");
			}
		} else {
			System.out.println("WARNING: not using flat H");
		}

		int[] rowIndices = new int[data.getNRows()];
		for (int i = 0; i < data.getNRows(); i++) {
			rowIndices[i] = i;
		}
		MLRandomUtils.shuffle(rowIndices, new Random(1));
		AtomicInteger counter = new AtomicInteger(0);
		IntStream.range(0, rowIndices.length).parallel().forEach(i -> {
			int count = counter.incrementAndGet();
			if (this.params.debug && count % this.params.printFrequency == 0) {
				timer.tocLoop(count);
			}
			int rowIndex = rowIndices[i];
			if (data.getRow(rowIndex) == null) {
				return;
			}

			MLDenseVector solution;
			if (useFlat[0] == true) {
				solution = solve(rowIndex, data, Hflat[0], HHflat, cache.get());
			} else {
				solution = solve(rowIndex, data, H, HHflat, cache.get());
			}

			W.setRow(solution, rowIndex);
		});
		if (this.params.debug) {
			timer.tocLoop(counter.get());
		}
	}

	public static void main(String[] args) {
		MLDenseMatrix V = new MLDenseMatrix(
				new MLDenseVector[] { new MLDenseVector(new float[] { 1, 2 }),
						new MLDenseVector(new float[] { 3, 4 }),
						new MLDenseVector(new float[] { 5, 6 }) });
		MLDenseMatrix U = new MLDenseMatrix(
				new MLDenseVector[] { new MLDenseVector(new float[] { 1, -2 }),
						new MLDenseVector(new float[] { 3, -4 }),
						new MLDenseVector(new float[] { 5, -6 }) });

		MLSparseVector[] test = new MLSparseVector[3];
		test[0] = new MLSparseVector(new int[] { 0, 1 }, new float[] { 1, 1 },
				null, 3);
		test[1] = new MLSparseVector(new int[] { 0, 1, 2 },
				new float[] { 1, 1, 1 }, null, 3);
		test[2] = new MLSparseVector(new int[] { 1, 2 }, new float[] { 1, 1 },
				null, 3);
		MLSparseMatrix R = new MLSparseMatrixAOO(test, 3);
		MLSparseMatrix RT = new MLSparseMatrixAOO(test, 3);

		ALSParams params = new ALSParams();
		params.maxIter = 1;
		params.rank = 2;
		params.lambda = 0f;

		ALS als = new ALS(params);
		als.solve(R, U, V);
		als.solve(RT, V, U);

		System.out.println("U");
		System.out.println(Arrays.toString(U.getRow(0).getValues()));
		System.out.println(Arrays.toString(U.getRow(1).getValues()));
		System.out.println(Arrays.toString(U.getRow(2).getValues()));
		System.out.println("\nV");
		System.out.println(Arrays.toString(V.getRow(0).getValues()));
		System.out.println(Arrays.toString(V.getRow(1).getValues()));
		System.out.println(Arrays.toString(V.getRow(2).getValues()));
	}
}
