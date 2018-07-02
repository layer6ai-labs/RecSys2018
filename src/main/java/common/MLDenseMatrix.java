package common;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;
import java.util.StringJoiner;
import java.util.function.Function;
import java.util.stream.IntStream;

import com.google.common.util.concurrent.AtomicDoubleArray;

public class MLDenseMatrix implements Serializable {

	private static final long serialVersionUID = -8815753536628968271L;
	private MLDenseVector[] rows;

	public MLDenseMatrix(final MLDenseVector[] rowsP) {
		this.rows = rowsP;
	}

	public MLDenseMatrix deepCopy() {
		MLDenseVector[] rowsCopy = new MLDenseVector[this.getNRows()];
		IntStream.range(0, this.getNRows()).parallel().forEach(rowIndex -> {
			MLDenseVector row = this.rows[rowIndex];
			if (row == null) {
				return;
			}
			rowsCopy[rowIndex] = row.deepCopy();
		});

		return new MLDenseMatrix(rowsCopy);
	}

	public MLDenseVector getColMean() {

		AtomicDoubleArray results = new AtomicDoubleArray(this.getNCols());
		float[] colMean = new float[this.getNCols()];

		IntStream.range(0, this.getNRows()).parallel().forEach(rowIndex -> {

			MLDenseVector row = this.getRow(rowIndex);

			if (row == null) {
				return;
			}

			float[] values = row.getValues();

			for (int i = 0; i < this.getNCols(); i++) {
				results.addAndGet(i, (double) values[i]);
			}
		});

		IntStream.range(0, this.getNCols()).parallel().forEach(k -> {
			colMean[k] = (float) results.get(k) / this.getNRows();
		});

		return new MLDenseVector(colMean);
	}

	public MLDenseVector getColStd(final MLDenseVector mean) {
		float[] colStd = new float[this.getNCols()];
		float[] colMean = mean.getValues();
		AtomicDoubleArray results = new AtomicDoubleArray(this.getNCols());

		IntStream.range(0, this.getNRows()).parallel().forEach(rowIndex -> {
			MLDenseVector row = this.getRow(rowIndex);

			if (row == null) {
				return;
			}

			float[] values = row.getValues();

			for (int i = 0; i < this.getNCols(); i++) {
				double diff = (double) values[i] - colMean[i];
				results.addAndGet(i, diff * diff);
			}
		});

		IntStream.range(0, this.getNCols()).parallel().forEach(k -> {
			colStd[k] = (float) Math.sqrt(results.get(k) / this.getNRows());
		});

		return new MLDenseVector(colStd);
	}

	public int getNCols() {
		return this.rows[0].getLength();
	}

	public int getNRows() {
		return this.rows.length;
	}

	public MLDenseVector getRow(final int rowIndex) {
		return this.rows[rowIndex];
	}

	public MLDenseVector[] getRows() {
		return this.rows;
	}

	public float getValue(final int rowIndex, final int colIndex) {
		return this.rows[rowIndex].getValue(colIndex);
	}

	public MLDenseVector multRow(final MLDenseVector vector,
			final boolean parallel) {

		// multiply this matrix with nCols x 1 dense vector
		float[] result = new float[this.getNRows()];
		if (parallel) {
			IntStream.range(0, this.getNRows()).parallel()
					.forEach(i -> result[i] = vector.mult(this.rows[i]));
		} else {
			for (int i = 0; i < this.getNRows(); i++) {
				result[i] = vector.mult(this.rows[i]);
			}
		}

		return new MLDenseVector(result);
	}

	public void setRow(final MLDenseVector row, final int rowIndex) {
		this.rows[rowIndex] = row;
	}

	public MLDenseMatrix slice(int fromInclusive, int toExclusive) {
		return new MLDenseMatrix(
				Arrays.copyOfRange(this.rows, fromInclusive, toExclusive));
	}

	public MLDenseMatrix slice(int[] inds) {
		return slice(inds, 0, inds.length);
	}

	public MLDenseMatrix slice(int[] inds, int from, int to) {
		MLDenseVector[] slice = new MLDenseVector[to - from];
		IntStream.range(from, to)
				.forEach(i -> slice[i - from] = this.rows[inds[i]]);
		return new MLDenseMatrix(slice);
	}

	public void Standardize() {
		// cutOff value for Standardize
		int cutOff = 5;

		MLDenseVector mean = this.getColMean();
		MLDenseVector std = this.getColStd(mean);

		float[] colMean = mean.getValues();
		float[] colStd = std.getValues();

		IntStream.range(0, this.getNRows()).parallel().forEach(rowIndex -> {
			MLDenseVector row = this.getRow(rowIndex);

			if (row == null) {
				return;
			}

			float[] values = row.getValues();

			for (int i = 0; i < this.getNCols(); i++) {
				if (colStd[i] == 0) {
					values[i] = values[i] - colMean[i];
					continue;
				}

				values[i] = (values[i] - colMean[i]) / colStd[i];

				if (values[i] > cutOff) {
					values[i] = cutOff;
				} else if (values[i] < -cutOff) {
					values[i] = -cutOff;
				}

			}

			this.setRow(new MLDenseVector(values), rowIndex);
		});

	}

	public void toFile(final String outFile) throws IOException {
		// in MATLAB
		// fid = fopen('U.bin');
		// U_target = fread(fid, nRows * nCols, 'float32', 'ieee-be');
		// fclose(fid);
		// U_target = reshape(U_target, nCols, nRows)';
		// U_target = U_target(2:end, :);

		try (DataOutputStream writer = new DataOutputStream(
				new BufferedOutputStream(new FileOutputStream(outFile)))) {
			for (MLDenseVector row : this.rows) {
				float[] values = row.getValues();
				for (float value : values) {
					writer.writeFloat(value);
				}
			}
		}

	}

	public float[] toFlatArray() {
		int nCols = this.getNCols();
		int nRows = this.getNRows();
		if (nCols * nRows > LowLevelRoutines.MAX_ARRAY_SIZE) {
			throw new IllegalArgumentException("nCols*nRows > MAX_ARRAY_SIZE");
		}

		float[] flat = new float[nCols * nRows];
		IntStream.range(0, nRows).parallel().forEach(rowIndex -> {
			float[] values = this.rows[rowIndex].getValues();
			int offset = rowIndex * nCols;
			System.arraycopy(values, 0, flat, offset, values.length);
		});
		return flat;
	}

	public float[][] toFlatArrayColSlices(final int maxFlatCols) {
		final int nRows = this.getNRows();
		final int nCols = this.getNCols();
		if (nCols > maxFlatCols) {
			// big
			int numBig = (int) Math.ceil((float) nCols / maxFlatCols);
			float[][] big = new float[numBig][];
			for (int i = 0; i < numBig; i++) {
				int start = i * maxFlatCols;
				int end = Math.min(start + maxFlatCols, nCols);
				final int nFlatCols = end - start;
				float[] flat = new float[nFlatCols * nRows];
				IntStream.range(0, nRows).parallel()
						.forEach(row -> System.arraycopy(
								this.rows[row].getValues(), start, flat,
								row * nFlatCols, nFlatCols));
				big[i] = flat;
			}
			return big;
		} else {
			return new float[][] { toFlatArray() };
		}
	}

	public MLSparseMatrix toSparse() {
		MLSparseVector[] rowsSparse = new MLSparseVector[this.getNRows()];
		IntStream.range(0, this.getNRows()).parallel().forEach(rowIndex -> {
			MLSparseVector rowSparse = this.rows[rowIndex].toSparse();
			if (rowSparse.getIndexes() != null) {
				rowsSparse[rowIndex] = rowSparse;
			}
		});

		return new MLSparseMatrixAOO(rowsSparse, this.getNCols());
	}

	@Override
	public String toString() {
		final int nrows = this.getNRows();
		final int ncols = this.getNCols();
		StringBuilder sb = new StringBuilder();
		String fmt = String.format("%%.%df", 4);
		String fmtLong = String.format("\t%s\t%s\t...\t%s\t%s\n", fmt, fmt, fmt,
				fmt);
		Function<float[], String> fmtRow = row -> {
			final int n = row.length;
			if (n > 4)
				return String.format(fmtLong, row[0], row[1], row[n - 2],
						row[n - 1]);
			else {
				StringJoiner ret = new StringJoiner("\t", "\t", "\n");
				for (float val : row)
					ret.add(String.format(fmt, val));
				return ret.toString();
			}
		};
		sb.append(String.format("[%d x %d]\n", nrows, ncols));
		if (nrows > 3) {
			for (int i = 0; i < 2; i++)
				sb.append(fmtRow.apply(this.getRow(i).getValues()));
			sb.append("\t...\n\t...\n");
			for (int i = nrows - 2; i < nrows; i++)
				sb.append(fmtRow.apply(this.getRow(i).getValues()));
		} else {
			for (int i = 0; i < nrows; i++)
				sb.append(fmtRow.apply(this.getRow(i).getValues()));
		}
		return sb.toString();

	}

	public MLDenseMatrix transposeMult() {
		final int nrows = this.getNRows();
		final int ncols = this.getNCols();

		MLDenseVector[] result = new MLDenseVector[ncols];
		IntStream.range(0, ncols).parallel().forEach(i -> {
			float[] resultRow = new float[ncols];
			for (int j = 0; j < ncols; j++) {
				for (int k = 0; k < nrows; k++) {
					resultRow[j] += this.rows[k].getValue(i)
							* this.rows[k].getValue(j);
				}
			}
			result[i] = new MLDenseVector(resultRow);
		});

		return new MLDenseMatrix(result);
	}

	public MLDenseMatrix transposeMultNative() {
		return transposeMultNative(LowLevelRoutines.MAX_ARRAY_SIZE);
	}

	public MLDenseMatrix transposeMultNative(final int maxArraySize) {
		final int nrows = this.getNRows();
		final int ncols = this.getNCols();
		// ability to fit R (m x N/m)
		int maxFlatCols = Math.floorDiv(maxArraySize, this.getNRows());
		if (maxFlatCols > ncols) {
			maxFlatCols = ncols;
		}
		// ability to fit RtR (N/m x N/m)
		int maxResultCols = Math.floorDiv(maxArraySize, maxFlatCols);
		// inability to fit N/m x N/m and overflowing R leads to extreme
		// inefficient case
		// when overflowing, m is usually pretty large that this won't happen
		if (ncols > maxFlatCols && maxResultCols < maxFlatCols) {
			throw new UnsupportedOperationException(
					"we cannot handle extremely flat and wide matrix");
		}
		MLDenseVector[] result = new MLDenseVector[ncols];
		if (ncols > maxFlatCols) {
			// big
			float[][] bigFlat = toFlatArrayColSlices(maxFlatCols);
			IntStream.range(0, ncols).parallel().forEach(
					row -> result[row] = new MLDenseVector(new float[ncols]));
			// permute multiply
			for (int f1 = 0; f1 < bigFlat.length; f1++) {
				final int colStartA = f1 * maxFlatCols;
				final int colEndA = Math.min(colStartA + maxFlatCols, ncols);
				final int ncolsA = colEndA - colStartA;

				float[] flatA = bigFlat[f1];
				// block multiply upper triangle only
				for (int f2 = f1; f2 < bigFlat.length; f2++) {
					final int colStartB = f2 * maxFlatCols;
					final int colEndB = Math.min(colStartB + maxFlatCols,
							ncols);
					final int ncolsB = colEndB - colStartB;
					float[] flatB = bigFlat[f2];

					float[] raw = new float[ncolsA * ncolsB];
					LowLevelRoutines.sgemm(flatA, flatB, raw, ncolsA, ncolsB,
							nrows, false, true, 1, 0);

					// copy into block
					IntStream.range(0, ncolsA).parallel().forEach(i -> {
						final int offset = i * ncolsB;
						System.arraycopy(raw, offset,
								result[i + colStartA].getValues(), colStartB,
								ncolsB);
					});
					// copy into mirrored block
					IntStream.range(0, ncolsB).parallel().forEach(i -> {
						final float[] mirror = result[i + colStartB]
								.getValues();
						for (int j = colStartA, k = i; j < colStartA
								+ ncolsA; j++, k += ncolsB) {
							mirror[j] = raw[k];
						}
					});
				}
			}
		} else {
			// fit ncol x nrow
			float[] flat = toFlatArray();
			maxResultCols = Math.floorDiv(maxArraySize, ncols);
			if (ncols > maxResultCols) {
				// fit R but not RtR
				float[][] bigFlat = this.toFlatArrayColSlices(maxResultCols);
				float[] raw = new float[maxResultCols * ncols];
				for (int f1 = 0; f1 < bigFlat.length; f1++) {
					final int colStartA = f1 * maxResultCols;
					final int colEndA = Math.min(colStartA + maxResultCols,
							ncols);
					final int ncolsA = colEndA - colStartA;
					LowLevelRoutines.sgemm(bigFlat[f1], flat, raw, ncolsA,
							ncols, nrows, false, true, 1, 0);
					IntStream.range(0, ncolsA).parallel().forEach(i -> {
						final int offset = i * ncols;
						float[] resultRow = Arrays.copyOfRange(raw, offset,
								offset + ncols);
						result[colStartA + i] = new MLDenseVector(resultRow);
					});
				}
			} else {
				// fit R AND RtR
				float[] raw = new float[ncols * ncols];
				LowLevelRoutines.sgemm(flat, flat, raw, ncols, ncols, nrows,
						false, true, 1, 0);
				IntStream.range(0, ncols).parallel().forEach(i -> {
					final int offset = i * ncols;
					float[] resultRow = Arrays.copyOfRange(raw, offset,
							offset + ncols);
					result[i] = new MLDenseVector(resultRow);
				});
			}
		}

		return new MLDenseMatrix(result);
	}

	public static MLDenseMatrix fromFile(final String inFile, final int nRows,
			final int nCols) throws IOException {

		MLDenseVector[] rows = new MLDenseVector[nRows];
		try (DataInputStream reader = new DataInputStream(
				new BufferedInputStream(new FileInputStream(inFile)))) {

			for (int i = 0; i < nRows; i++) {
				float[] values = new float[nCols];
				for (int j = 0; j < nCols; j++) {
					values[j] = reader.readFloat();
				}
				rows[i] = new MLDenseVector(values);
			}

			if (reader.available() != 0) {
				throw new IllegalArgumentException(
						"data left after reading nRows x nCols elements");
			}

		}

		return new MLDenseMatrix(rows);
	}

	public static MLDenseMatrix initRandom(final int nRows, final int nCols,
			final float initStd, final long seed) {

		MLDenseVector[] rows = new MLDenseVector[nRows];
		IntStream.range(0, nRows).parallel().forEach(i -> {

			float[] values = new float[nCols];
			// ensures that random init is repeatable
			Random random = new Random(i + seed);
			for (int j = 0; j < nCols; j++) {
				values[j] = initStd * ((float) random.nextGaussian());
			}
			rows[i] = new MLDenseVector(values);
		});

		return new MLDenseMatrix(rows);
	}
}
