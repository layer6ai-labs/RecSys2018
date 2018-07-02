package common;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

import com.google.common.util.concurrent.AtomicDoubleArray;

public class MLSparseMatrixFlat implements MLSparseMatrix {

	private static final long serialVersionUID = -7708714593085005498L;
	public static final int MISSING_ROW = -1;

	private int[] indexes;
	private float[] values;
	private int nCols;

	public MLSparseMatrixFlat(final int nRowsP, final int nColsP) {
		this.indexes = new int[nRowsP];
		Arrays.fill(this.indexes, MISSING_ROW);
		this.values = new float[nRowsP];
		this.nCols = nColsP;
	}

	public MLSparseMatrixFlat(final int[] indexesP, final float[] valuesP,
			final int nColsP) {
		this.indexes = indexesP;
		this.values = valuesP;
		this.nCols = nColsP;
	}

	@Override
	public void applyColNorm(final MLDenseVector colNorm) {
		IntStream.range(0, this.getNRows()).parallel().forEach(rowIndex -> {
			if (this.indexes[rowIndex] == MISSING_ROW) {
				return;
			}

			float norm = colNorm.getValue(this.indexes[rowIndex]);
			if (norm > 1e-10f) {
				this.values[rowIndex] /= norm;
			}
		});
	}

	@Override
	public void applyColSelector(final Map<Integer, Integer> selectedColMap,
			final int nColsSelected) {
		if (this.nCols == nColsSelected) {
			boolean noChanges = true;
			for (Map.Entry<Integer, Integer> entry : selectedColMap
					.entrySet()) {
				if (entry.getValue() != entry.getKey()) {
					noChanges = false;
					break;
				}
			}

			if (noChanges == true) {
				// nothing to do
				return;
			}
		}

		IntStream.range(0, this.getNRows()).parallel().forEach(rowIndex -> {
			Integer index = this.indexes[rowIndex];
			if (index == MISSING_ROW) {
				return;
			}

			index = selectedColMap.get(index);
			if (index == null) {
				// not in the map so remove this row
				this.removeRow(rowIndex);

			} else {
				this.indexes[rowIndex] = index;
			}
		});

		this.setNCols(nColsSelected);
	}

	@Override
	public void applyRowNorm(final MLDenseVector rowNorm) {
		IntStream.range(0, this.getNRows()).parallel().forEach(rowIndex -> {
			if (this.indexes[rowIndex] == MISSING_ROW) {
				return;
			}

			float norm = rowNorm.getValue(rowIndex);
			if (norm > 1e-5f) {
				this.values[rowIndex] /= norm;
			}
		});

	}

	@Override
	public void binarizeValues() {
		Arrays.fill(this.values, 1f);
	}

	@Override
	public MLSparseMatrix deepCopy() {
		return new MLSparseMatrixFlat(this.indexes.clone(), this.values.clone(),
				this.nCols);
	}

	@Override
	public MLDenseVector getColNNZ() {
		float[] colNNZ = new float[this.getNCols()];
		IntStream.range(0, this.getNRows()).parallel().forEach(rowIndex -> {
			int colIndex = this.indexes[rowIndex];
			if (colIndex == MISSING_ROW) {
				return;
			}
			synchronized (colNNZ) {
				colNNZ[colIndex] += 1;
			}
		});
		return new MLDenseVector(colNNZ);
	}

	@Override
	public MLDenseVector getColNorm(final int p) {
		// compute L^p norm
		final float[] colNorm = new float[this.getNCols()];
		IntStream.range(0, this.getNRows()).parallel().forEach(rowIndex -> {
			int colIndex = this.indexes[rowIndex];
			if (colIndex == MISSING_ROW) {
				return;
			}
			synchronized (colNorm) {
				if (p == 1) {
					colNorm[colIndex] += Math.abs(this.values[rowIndex]);
				} else {
					colNorm[colIndex] += Math.pow(this.values[rowIndex], p);
				}
			}
		});

		if (p != 1) {
			for (int i = 0; i < this.getNCols(); i++) {
				// take p'th root
				colNorm[i] = (float) Math.pow(colNorm[i], 1.0 / p);
			}
		}
		return new MLDenseVector(colNorm);
	}

	@Override
	public MLDenseVector getColSum() {
		float[] colSum = new float[this.getNCols()];
		IntStream.range(0, this.getNRows()).parallel().forEach(rowIndex -> {
			int colIndex = this.indexes[rowIndex];
			if (colIndex == MISSING_ROW) {
				return;
			}
			synchronized (colSum) {
				colSum[colIndex] += this.values[rowIndex];
			}
		});
		return new MLDenseVector(colSum);
	}

	@Override
	public MLDenseVector getRowSum() {
		return new MLDenseVector(this.values);
	}

	@Override
	public int getNCols() {
		return this.nCols;
	}

	@Override
	public long getNNZ() {
		long nnz = 0;
		for (int i = 0; i < this.indexes.length; i++) {
			if (this.indexes[i] != MISSING_ROW) {
				nnz++;
			}
		}
		return nnz;
	}

	@Override
	public int getNRows() {
		return this.indexes.length;
	}

	@Override
	public MLSparseVector getRow(final int rowIndex) {
		int colIndex = this.indexes[rowIndex];
		if (colIndex == MISSING_ROW) {
			return null;
		}

		return new MLSparseVector(new int[] { colIndex },
				new float[] { this.values[rowIndex] }, null, this.nCols);
	}

	@Override
	public MLSparseVector getRow(final int rowIndex, boolean returnEmpty) {
		MLSparseVector row = this.getRow(rowIndex);
		if (row == null && returnEmpty == true) {
			// return empty row instead of null
			row = new MLSparseVector(new int[] {}, new float[] {}, null,
					this.getNCols());
		}
		return row;
	}

	@Override
	public MLDenseVector getRowNNZ() {
		float[] rowNNZ = new float[this.getNRows()];
		IntStream.range(0, this.getNRows()).parallel().forEach(rowIndex -> {
			int colIndex = this.indexes[rowIndex];
			if (colIndex == MISSING_ROW) {
				return;
			}
			rowNNZ[rowIndex] = 1;
		});
		return new MLDenseVector(rowNNZ);
	}

	@Override
	public MLDenseVector getRowNorm(final int p) {
		final float[] rowNorm = new float[this.getNRows()];
		System.arraycopy(values, 0, rowNorm, 0, rowNorm.length);
		return new MLDenseVector(rowNorm);
	}

	@Override
	public boolean hasDates() {
		return false;
	}

	@Override
	public void inferAndSetNCols() {
		// infer number of columns if it wasn't known during constructor
		int nColsNew = 0;

		for (int i = 0; i < this.indexes.length; i++) {
			int colIndex = this.indexes[i];
			if (colIndex == MISSING_ROW) {
				continue;
			}

			if (colIndex + 1 > nColsNew) {
				// nCols is 1 + largest col index
				nColsNew = colIndex + 1;
			}
		}

		this.setNCols(nColsNew);
	}

	@Override
	public MLSparseMatrix mult(final MLSparseMatrix another) {
		if (this.getNCols() != another.getNRows()) {
			throw new IllegalArgumentException(
					"this.getNCols() != another.getNRows()");
		}
		MLSparseVector[] resultRows = new MLSparseVector[this.getNRows()];
		IntStream.range(0, this.getNRows()).parallel().forEach(i -> {
			int index = this.indexes[i];
			if (index == MISSING_ROW) {
				return;
			}
			float value = this.values[i];
			float[] resultRow = new float[another.getNCols()];
			MLSparseVector rowAnother = another.getRow(index);
			if (rowAnother != null) {

				int[] indexesAnother = rowAnother.getIndexes();
				float[] valuesAnother = rowAnother.getValues();
				for (int k = 0; k < indexesAnother.length; k++) {
					resultRow[indexesAnother[k]] += value * valuesAnother[k];
				}
			}
			MLSparseVector resultRowSparse = MLSparseVector
					.fromDense(new MLDenseVector(resultRow));
			if (resultRowSparse.getIndexes() != null) {
				resultRows[i] = resultRowSparse;
			}
		});

		return new MLSparseMatrixAOO(resultRows, another.getNCols());
	}

	@Override
	public MLDenseVector multCol(final MLDenseVector vector) {

		// multiply 1 x nRows dense vector with this matrix
		if (this.getNRows() != vector.getLength()) {
			throw new IllegalArgumentException(
					"this.getNRows() != vector.getLength()");
		}

		AtomicDoubleArray result = new AtomicDoubleArray(this.nCols);
		IntStream.range(0, this.getNRows()).parallel().forEach(rowIndex -> {
			float val = vector.getValue(rowIndex);
			if (val == 0) {
				return;
			}

			int colIndex = this.indexes[rowIndex];
			if (colIndex == MISSING_ROW) {
				return;
			}
			float colValue = this.values[rowIndex];
			result.addAndGet(colIndex, val * colValue);
		});
		float[] temp = new float[this.nCols];
		for (int i = 0; i < temp.length; i++) {
			temp[i] = (float) result.get(i);
		}

		return new MLDenseVector(temp);
	}

	@Override
	public MLDenseVector multCol(final MLSparseVector vector) {

		// multiply 1 x nRows sparse vector with this matrix
		if (this.getNRows() != vector.getLength()) {
			throw new IllegalArgumentException(
					"this.getNRows() != vector.getLength()");
		}

		AtomicDoubleArray result = new AtomicDoubleArray(this.nCols);
		int[] vectorIndexes = vector.getIndexes();
		float[] vectorValues = vector.getValues();
		IntStream.range(0, vectorIndexes.length).parallel()
				.forEach(rowIndex -> {
					int ind = vectorIndexes[rowIndex];
					float val = vectorValues[rowIndex];

					int colIndex = this.indexes[ind];
					if (colIndex == MISSING_ROW) {
						return;
					}
					float colValue = this.values[ind];

					result.addAndGet(colIndex, val * colValue);
				});

		float[] temp = new float[this.nCols];
		for (int i = 0; i < temp.length; i++) {
			temp[i] = (float) result.get(i);
		}

		return new MLDenseVector(temp);
	}

	@Override
	public MLDenseVector multRow(final MLDenseVector vector) {

		// multiply this matrix with nCols x 1 dense vector
		if (this.getNCols() != vector.getLength()) {
			throw new IllegalArgumentException(
					"this.getNCols() != vector.getLength()");
		}

		float[] result = new float[this.getNRows()];
		IntStream.range(0, this.getNRows()).parallel().forEach(rowIndex -> {
			int colIndex = this.indexes[rowIndex];
			if (colIndex == MISSING_ROW) {
				return;
			}
			float colValue = this.values[rowIndex];

			result[rowIndex] = vector.getValue(colIndex) * colValue;
		});

		return new MLDenseVector(result);
	}

	@Override
	public MLDenseVector multRow(final MLSparseVector vector) {

		// multiply this matrix with nCols x 1 sparse vector
		if (this.getNCols() != vector.getLength()) {
			throw new IllegalArgumentException(
					"this.getNCols() != vector.getLength()");
		}

		int[] vecIndexes = vector.getIndexes();
		float[] vecValues = vector.getValues();
		float[] result = new float[this.getNRows()];
		IntStream.range(0, this.getNRows()).parallel().forEach(rowIndex -> {
			int colIndex = this.indexes[rowIndex];
			if (colIndex == MISSING_ROW) {
				return;
			}
			float colValue = this.values[rowIndex];
			int matchIndex = Arrays.binarySearch(vecIndexes, colIndex);
			if (matchIndex >= 0) {
				result[rowIndex] = vecValues[matchIndex] * colValue;
			}
		});

		return new MLDenseVector(result);
	}

	private void removeRow(final int rowIndex) {
		this.indexes[rowIndex] = MISSING_ROW;
	}

	@Override
	public Map<Integer, Integer> selectCols(final int nnzCutOff) {
		Map<Integer, Integer> selectedColMap = new HashMap<Integer, Integer>(
				this.nCols);

		MLDenseVector colNNZ = this.getColNNZ();
		int newIndex = 0;
		for (int colIndex = 0; colIndex < this.nCols; colIndex++) {
			if (colNNZ.getValue(colIndex) > nnzCutOff) {
				selectedColMap.put(colIndex, newIndex);
				newIndex++;
			}
		}

		return selectedColMap;
	}

	@Override
	public void setNCols(final int nColsP) {
		this.nCols = nColsP;

	}

	public void setRow(final int index, final float value, final int rowIndex) {
		this.indexes[rowIndex] = index;
		this.values[rowIndex] = value;
	}

	@Override
	public void setRow(final MLSparseVector row, final int rowIndex) {
		if (row == null || row.getIndexes().length == 0) {
			this.removeRow(rowIndex);
			return;
		}

		int[] rowIndexes = row.getIndexes();
		if (rowIndexes.length != 1) {
			throw new IllegalArgumentException(
					"can't add row with != 1 element");
		}
		float[] rowValues = row.getValues();

		this.indexes[rowIndex] = rowIndexes[0];
		this.values[rowIndex] = rowValues[0];
	}

	@Override
	public void toBinFile(final String outFile) throws Exception {
		throw new UnsupportedOperationException("unsupported function");
	}

	@Override
	public MLSparseMatrix transpose() {
		/**
		 * convert to csr
		 */
		final int nnz = (int) this.getNNZ();
		final int nRows = this.getNRows();
		final int[] jaP = new int[nnz];
		final float[] aP = new float[nnz];
		for (int i = 0, inz = 0; i < nRows; i++) {
			int jaPi = this.indexes[i];
			if (jaPi != MISSING_ROW) {
				jaP[inz] = jaPi;
				aP[inz] = this.values[i];
				inz++;
			}
		}
		/**
		 * perform transpose
		 */
		final int nnzT = nnz;
		final int nRowsT = this.getNCols();
		final int nColsT = this.getNRows();
		final int[] rowIndexT = new int[nRowsT + 1];
		final int[] jaPT = new int[nnzT];
		final float[] aPT = new float[nnzT];

		// count nnz in each row
		for (int i = 0; i < nnzT; i++) {
			int jaPi = jaP[i];
			if (jaPi != MISSING_ROW) {
				rowIndexT[jaPi]++;
			}
		}

		// Fill starting point of the previous row to begin tally
		int r, j;
		rowIndexT[nRowsT] = nnzT - rowIndexT[nRowsT - 1];
		for (r = nRowsT - 1; r > 0; r--) {
			rowIndexT[r] = rowIndexT[r + 1] - rowIndexT[r - 1];
		}
		rowIndexT[0] = 0;

		// assign the new columns and values
		// synchronously tally
		// this is the place to insert extra values like dates
		for (int c = 0, i = 0; c < nColsT; c++) {
			// don't need to walk through row, there's only 0/1 vals per row
			if (this.indexes[c] == MISSING_ROW) {
				continue;
			}
			r = jaP[i];
			j = rowIndexT[r + 1]++;
			jaPT[j] = c;
			aPT[j] = aP[i];
			i++;
		}
		// TODO: CSR3 -> NIST and parsing can probably be more clever
		int[] pntrBT = new int[nRowsT];
		int[] pntrET = new int[nRowsT];
		int lastActivePtrE = 0;
		for (int i = 0; i < nRowsT; i++) {
			if (lastActivePtrE == rowIndexT[i + 1]) {
				continue;
			}
			pntrET[i] = rowIndexT[i + 1];
			pntrBT[i] = lastActivePtrE;
			lastActivePtrE = rowIndexT[i + 1];
		}

		/**
		 * consolidate csr (NIST) back to mlsparse
		 */
		final MLSparseVector[] rows = new MLSparseVector[nRowsT];
		IntStream.range(0, nRowsT).parallel().forEach(i -> {
			int rownnz = pntrET[i] - pntrBT[i];
			if (rownnz == 0) {
				return;
			}
			int[] rowColInds = new int[rownnz];
			float[] rowVals = new float[rownnz];
			for (int jj = 0, k = pntrBT[i]; jj < rownnz; jj++, k++) {
				rowColInds[jj] = jaPT[k];
				rowVals[jj] = aPT[k];
			}
			MLSparseVector rowVec = new MLSparseVector(rowColInds, rowVals,
					null, nColsT);
			rows[i] = rowVec;
		});
		return new MLSparseMatrixAOO(rows, nColsT);
	}
}
