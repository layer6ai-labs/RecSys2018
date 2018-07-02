package common;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.IntStream;

import com.google.common.util.concurrent.AtomicDoubleArray;

public class MLSparseMatrixAOO implements MLSparseMatrix {

	private static final long serialVersionUID = -7521797137964819356L;
	private MLSparseVector[] rows;
	private int nCols;

	public MLSparseMatrixAOO(final int nRowsP, final int nColsP) {
		this.rows = new MLSparseVector[nRowsP];
		this.nCols = nColsP;
	}

	public MLSparseMatrixAOO(final MLSparseVector[] rowsP, final int nColsP) {
		this.rows = rowsP;
		this.nCols = nColsP;
	}

	@Override
	public void applyColNorm(final MLDenseVector colNorm) {
		IntStream.range(0, this.getNRows()).parallel().forEach(rowIndex -> {
			MLSparseVector row = this.rows[rowIndex];
			if (row == null) {
				return;
			}
			row.applyNorm(colNorm);
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

		// apply column selector in place to this matrix
		IntStream.range(0, this.getNRows()).parallel().forEach(rowIndex -> {
			MLSparseVector row = this.rows[rowIndex];
			if (row == null) {
				return;
			}

			if (nColsSelected == 0) {
				this.rows[rowIndex] = null;
				return;
			}

			int[] indexes = row.getIndexes();
			float[] values = row.getValues();
			long[] dates = row.getDates();

			// apply column selector in place to this vector
			List<MLMatrixElement> reindexElms = new ArrayList<MLMatrixElement>(
					row.getIndexes().length);
			for (int i = 0; i < indexes.length; i++) {
				Integer newIndex = selectedColMap.get(indexes[i]);
				if (newIndex != null) {
					if (dates != null) {
						reindexElms.add(new MLMatrixElement(-1, newIndex,
								values[i], dates[i]));
					} else {
						reindexElms.add(new MLMatrixElement(-1, newIndex,
								values[i], -1));
					}
				}
			}

			if (reindexElms.size() == 0) {
				this.rows[rowIndex] = null;
				return;
			}

			Collections.sort(reindexElms,
					new MLMatrixElement.ColIndexComparator(false));
			int[] prunedIndexes = new int[reindexElms.size()];
			float[] prunedValues = new float[reindexElms.size()];
			long[] prunedDates = null;
			if (dates != null) {
				prunedDates = new long[reindexElms.size()];
			}

			int cur = 0;
			for (MLMatrixElement element : reindexElms) {

				prunedIndexes[cur] = element.getColIndex();
				prunedValues[cur] = element.getValue();
				if (dates != null) {
					prunedDates[cur] = element.getDate();
				}
				cur++;
			}

			this.rows[rowIndex] = new MLSparseVector(prunedIndexes,
					prunedValues, prunedDates, nColsSelected);

		});

		// update matrix nCols
		this.setNCols(nColsSelected);
	}

	public void applyDateThresh(final long dateThresh, final boolean greater) {

		IntStream.range(0, this.getNRows()).parallel().forEach(rowIndex -> {
			MLSparseVector row = this.rows[rowIndex];
			if (row == null) {
				return;
			}

			row.applyDateThresh(dateThresh, greater);
			if (row.getIndexes() == null) {
				// empty rows must be set to null
				this.rows[rowIndex] = null;
			}
		});
	}

	@Override
	public void applyRowNorm(final MLDenseVector rowNorm) {
		float[] normValues = rowNorm.getValues();
		IntStream.range(0, this.getNRows()).parallel().forEach(rowIndex -> {
			MLSparseVector row = this.rows[rowIndex];
			if (row == null) {
				return;
			}

			if (normValues[rowIndex] > 1e-5f) {
				row.divide(normValues[rowIndex]);
			}
		});
	}

	@Override
	public void binarizeValues() {
		IntStream.range(0, this.getNRows()).parallel().forEach(rowIndex -> {
			MLSparseVector row = this.rows[rowIndex];
			if (row == null) {
				return;
			}

			float[] values = row.getValues();
			for (int i = 0; i < values.length; i++) {
				values[i] = 1f;
			}
		});
	}

	@Override
	public MLSparseMatrix deepCopy() {
		MLSparseVector[] rowsCopy = new MLSparseVector[this.getNRows()];
		IntStream.range(0, this.getNRows()).parallel().forEach(rowIndex -> {
			MLSparseVector row = this.rows[rowIndex];
			if (row == null) {
				return;
			}
			rowsCopy[rowIndex] = row.deepCopy();
		});

		return new MLSparseMatrixAOO(rowsCopy, this.getNCols());
	}

	@Override
	public MLDenseVector getColNNZ() {
		float[] colNNZ = new float[this.getNCols()];
		IntStream.range(0, this.getNRows()).parallel().forEach(rowIndex -> {
			MLSparseVector row = this.rows[rowIndex];
			if (row == null) {
				return;
			}
			int[] indexes = row.getIndexes();
			for (int j = 0; j < indexes.length; j++) {
				synchronized (colNNZ) {
					colNNZ[indexes[j]] += 1;
				}
			}
		});
		return new MLDenseVector(colNNZ);
	}

	@Override
	public MLDenseVector getColNorm(final int p) {
		// compute L^p norm
		final int nCol = this.getNCols();
		final float[] colNorm = new float[nCol];
		IntStream.range(0, this.getNRows()).parallel().forEach(rowIndex -> {
			MLSparseVector row = this.rows[rowIndex];
			if (row == null) {
				return;
			}

			float[] values = row.getValues();
			int[] indexes = row.getIndexes();
			synchronized (colNorm) {
				for (int i = 0; i < values.length; i++) {
					if (p == 1) {
						colNorm[indexes[i]] += Math.abs(values[i]);
					} else {
						colNorm[indexes[i]] += Math.pow(values[i], p);
					}
				}
			}
		});

		if (p != 1) {
			for (int i = 0; i < nCol; i++) {
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
			MLSparseVector row = this.rows[rowIndex];
			if (row == null) {
				return;
			}
			int[] indexes = row.getIndexes();
			float[] values = row.getValues();
			for (int i = 0; i < indexes.length; i++) {
				synchronized (colSum) {
					colSum[indexes[i]] += values[i];
				}
			}
		});
		return new MLDenseVector(colSum);
	}

	@Override
	public int getNCols() {
		return this.nCols;
	}

	@Override
	public long getNNZ() {

		long nnz = 0;
		for (MLSparseVector row : this.rows) {
			if (row == null) {
				continue;
			}
			nnz += row.getIndexes().length;
		}

		return nnz;
	}

	@Override
	public int getNRows() {
		return this.rows.length;
	}

	@Override
	public MLSparseVector getRow(final int rowIndex) {
		return this.rows[rowIndex];
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
		IntStream.range(0, this.getNRows()).parallel().forEach(i -> {
			MLSparseVector row = this.rows[i];
			if (row == null) {
				return;
			}
			rowNNZ[i] = row.getIndexes().length;
		});
		return new MLDenseVector(rowNNZ);
	}

	@Override
	public MLDenseVector getRowNorm(final int p) {
		final float[] rowNorm = new float[this.getNRows()];
		IntStream.range(0, this.getNRows()).parallel().forEach(rowIndex -> {
			MLSparseVector row = this.rows[rowIndex];
			if (row == null) {
				return;
			}
			rowNorm[rowIndex] = row.getNorm(p);
		});
		return new MLDenseVector(rowNorm);
	}

	@Override
	public MLDenseVector getRowSum() {
		float[] rowSum = new float[this.getNRows()];
		IntStream.range(0, this.getNRows()).parallel().forEach(rowIndex -> {
			MLSparseVector row = this.rows[rowIndex];
			if (row == null) {
				return;
			}
			float[] values = row.getValues();
			for (int i = 0; i < values.length; i++) {
				rowSum[rowIndex] += values[i];
			}
		});
		return new MLDenseVector(rowSum);
	}

	@Override
	public boolean hasDates() {
		for (MLSparseVector row : this.rows) {
			if (row != null) {
				if (row.getDates() != null && row.getDates().length > 0) {
					return true;

				} else {
					return false;
				}
			}
		}
		return false;
	}

	@Override
	public void inferAndSetNCols() {
		// infer number of columns if it wasn't known during constructor
		int nColsNew = 0;

		for (MLSparseVector row : this.rows) {
			if (row == null) {
				continue;
			}

			int[] indexes = row.getIndexes();
			if (indexes[indexes.length - 1] + 1 > nColsNew) {
				// nCols is 1 + largest col index
				nColsNew = indexes[indexes.length - 1] + 1;
			}
		}

		this.setNCols(nColsNew);
	}

	public void merge(final MLSparseMatrixAOO matToMerge) {

		if (this.getNRows() != matToMerge.getNRows()) {
			throw new IllegalArgumentException(
					"nRows must be the same to merge");
		}

		if (this.getNCols() != matToMerge.getNCols()) {
			throw new IllegalArgumentException(
					"nCols must be the same to merge");
		}

		if (this.hasDates() != matToMerge.hasDates()) {
			throw new IllegalArgumentException(
					"hasDates() must be the same to merge");
		}

		boolean hasDates = this.hasDates();
		IntStream.range(0, this.getNRows()).parallel().forEach(rowIndex -> {

			MLSparseVector curRow = this.rows[rowIndex];
			MLSparseVector mergeRow = matToMerge.getRow(rowIndex);
			if (mergeRow == null) {
				return;
			}

			if (curRow == null) {
				if (hasDates == true) {
					this.rows[rowIndex] = new MLSparseVector(
							mergeRow.getIndexes().clone(),
							mergeRow.getValues().clone(),
							mergeRow.getDates().clone(), this.getNCols());
				} else {
					this.rows[rowIndex] = new MLSparseVector(
							mergeRow.getIndexes().clone(),
							mergeRow.getValues().clone(), null,
							this.getNCols());
				}

				return;
			}

			curRow.merge(mergeRow);
		});
	}

	@Override
	public MLSparseMatrix mult(final MLSparseMatrix another) {
		if (this.getNCols() != another.getNRows()) {
			throw new IllegalArgumentException(
					"this.getNCols() != another.getNRows()");
		}
		MLSparseVector[] resultRows = new MLSparseVector[this.getNRows()];
		IntStream.range(0, this.getNRows()).parallel().forEach(i -> {
			MLSparseVector row = this.rows[i];
			if (row == null) {
				return;
			}

			float[] resultRow = new float[another.getNCols()];
			int[] indexes = row.getIndexes();
			float[] values = row.getValues();
			for (int j = 0; j < indexes.length; j++) {
				int index = indexes[j];
				float value = values[j];

				MLSparseVector rowAnother = another.getRow(index);
				if (rowAnother == null) {
					continue;
				}

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

			MLSparseVector row = this.rows[rowIndex];
			if (row == null) {
				return;
			}

			int[] indexes = row.getIndexes();
			float[] values = row.getValues();

			for (int i = 0; i < indexes.length; i++) {
				result.addAndGet(indexes[i], val * values[i]);
			}
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
					float val = vectorValues[rowIndex];
					if (val == 0) {
						return;
					}

					MLSparseVector row = this.rows[vectorIndexes[rowIndex]];
					if (row == null) {
						return;
					}

					int[] indexes = row.getIndexes();
					float[] values = row.getValues();

					for (int i = 0; i < indexes.length; i++) {
						result.addAndGet(indexes[i], val * values[i]);
					}
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
		IntStream.range(0, this.getNRows()).parallel().forEach(i -> {
			MLSparseVector row = this.rows[i];
			if (row == null) {
				return;
			}
			result[i] = vector.mult(row);
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

		float[] result = new float[this.getNRows()];
		IntStream.range(0, this.getNRows()).parallel().forEach(i -> {
			MLSparseVector row = this.rows[i];
			if (row == null) {
				return;
			}
			result[i] = row.multiply(vector);
		});

		return new MLDenseVector(result);
	}

	@Override
	public Map<Integer, Integer> selectCols(int nnzCutOff) {
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
	public void setNCols(int nCols) {
		this.nCols = nCols;

		// set nCols for all rows
		for (MLSparseVector row : this.rows) {
			if (row == null) {
				continue;
			}
			row.setLength(this.nCols);
		}
	}

	@Override
	public void setRow(final MLSparseVector row, final int rowIndex) {
		if (row == null || row.getIndexes() == null) {
			this.rows[rowIndex] = null;
		} else {
			this.rows[rowIndex] = row;
		}
	}

	public void tfIDF() {
		// apply tf-idf in place
		float[] idf = this.getColNNZ().getValues();
		for (int i = 0; i < idf.length; i++) {
			if (idf[i] != 0) {
				idf[i] = (float) Math.log(this.getNRows() / idf[i]);
			}
		}

		IntStream.range(0, this.getNRows()).parallel().forEach(rowIndex -> {
			MLSparseVector row = this.rows[rowIndex];
			if (row == null) {
				return;
			}
			int[] indexes = row.getIndexes();
			float[] values = row.getValues();
			// float max = row.max();
			for (int i = 0; i < indexes.length; i++) {
				values[i] = (float) Math.log(1 + values[i]) * idf[indexes[i]];

				// values[i] = (float) (0.5f + 0.5f * values[i] / max)
				// * idf[indexes[i]];
			}
		});
	}

	@Override
	public void toBinFile(final String outFile) throws Exception {

		try (DataOutputStream writer = new DataOutputStream(
				new BufferedOutputStream(new FileOutputStream(outFile)))) {
			for (int i = 0; i < this.rows.length; i++) {
				MLSparseVector row = this.rows[i];
				if (row == null) {
					continue;
				}

				int[] indexes = row.getIndexes();
				float[] values = row.getValues();
				long[] dates = row.getDates();

				for (int j = 0; j < indexes.length; j++) {
					writer.writeInt(i);
					writer.writeInt(indexes[j]);
					writer.writeFloat(values[j]);

					if (dates != null) {
						writer.writeLong(dates[j]);
					}
				}
			}
		}
	}

	@Override
	public MLSparseMatrix transpose() {
		/**
		 * convert to csr
		 */
		final int nnz = (int) this.getNNZ();
		final int nRows = this.getNRows();
		final int[] pntrB = new int[nRows];
		final int[] pntrE = new int[nRows];
		final int[] jaP = new int[nnz];
		final float[] aP = new float[nnz];
		final long[] datesP;
		final boolean hasDates = this.hasDates();
		if (hasDates) {
			datesP = new long[nnz];
		} else {
			datesP = null;
		}
		{
			// namespace just for csr creation
			int cur = 0;
			int rowNNZ, rowi;
			MLSparseVector raw;
			int[] ind;
			float[] val;
			long[] dates;

			for (rowi = 0; rowi < this.getNRows(); rowi++) {
				raw = this.getRow(rowi);
				if (raw == null) {
					continue;
				}
				ind = raw.getIndexes();
				rowNNZ = ind.length;
				val = raw.getValues();
				dates = raw.getDates();
				pntrB[rowi] = cur;
				System.arraycopy(ind, 0, jaP, cur, rowNNZ);
				System.arraycopy(val, 0, aP, cur, rowNNZ);
				if (hasDates) {
					System.arraycopy(dates, 0, datesP, cur, rowNNZ);
				}
				cur += rowNNZ;
				pntrE[rowi] = cur;
			}
		}
		/**
		 * perform transpose
		 */
		final int nnzT = nnz;
		final int nRowsT = nCols;
		final int nColsT = nRows;
		final int[] rowIndexT = new int[nRowsT + 1];
		final int[] jaPT = new int[nnzT];
		final float[] aPT = new float[nnzT];

		final long[] datesPT;
		if (hasDates) {
			datesPT = new long[nnzT];
		} else {
			datesPT = null;
		}

		// count nnz in each row
		for (int i = 0; i < nnzT; i++) {
			rowIndexT[jaP[i]]++;
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
			for (; i < pntrE[c]; i++) {
				r = jaP[i];
				j = rowIndexT[r + 1]++;
				jaPT[j] = c;
				aPT[j] = aP[i];
				if (hasDates) {
					datesPT[j] = datesP[i];
				}
			}
		}
		// CSR3 -> NIST
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
			long[] rowDates = null;
			if (datesPT != null) {
				rowDates = new long[rownnz];
			}
			for (int jj = 0, k = pntrBT[i]; jj < rownnz; jj++, k++) {
				rowColInds[jj] = jaPT[k];
				rowVals[jj] = aPT[k];
				if (datesPT != null) {
					rowDates[jj] = datesPT[k];
				}
			}
			MLSparseVector rowVec = new MLSparseVector(rowColInds, rowVals,
					rowDates, nColsT);
			rows[i] = rowVec;
		});
		return new MLSparseMatrixAOO(rows, nColsT);
	}

	public static MLSparseMatrix loadFromCSV(final String inFile,
			final int nUsers, final int nItems, final Integer[] csvIndexes)
			throws FileNotFoundException, IOException {
		final int USER_INDEX = 0;
		final int ITEM_INDEX = 1;
		final int INTERACTION_INDEX = 2;
		final int DATE_INDEX = 3;

		MLTimer timer = new MLTimer("loadFromCSV");
		try (BufferedReader reader = new BufferedReader(
				new FileReader(inFile))) {
			Map<Integer, MLMatrixElement>[] rowMaps = new Map[nUsers];

			String line = null;
			timer.tic();
			int count = 0;
			while ((line = reader.readLine()) != null) {
				String[] split = line.split(",");
				int userIndex = Integer.parseInt(split[csvIndexes[USER_INDEX]]);
				int itemIndex = Integer.parseInt(split[csvIndexes[ITEM_INDEX]]);
				float interaction = 1f;
				if (csvIndexes[INTERACTION_INDEX] != null) {
					interaction = Float
							.parseFloat(split[csvIndexes[INTERACTION_INDEX]]);
				}

				long date = 0;
				if (csvIndexes[DATE_INDEX] != null) {
					date = Long.parseLong(split[csvIndexes[DATE_INDEX]]);
				}

				if (rowMaps[userIndex] == null) {
					rowMaps[userIndex] = new TreeMap<Integer, MLMatrixElement>();
				}

				Map<Integer, MLMatrixElement> userItemMap = rowMaps[userIndex];
				MLMatrixElement cur = userItemMap.get(itemIndex);
				if (cur == null) {
					userItemMap.put(itemIndex, new MLMatrixElement(userIndex,
							itemIndex, interaction, date));
				} else {
					if (csvIndexes[DATE_INDEX] != null
							&& date > cur.getDate()) {
						// store latest date
						cur.setDate(date);
					}
					// increment interaction count
					float value = cur.getValue() + interaction;
					cur.setValue(value);
				}

				count++;
				if (count % 5_000_000 == 0) {
					timer.tocLoop(count);
				}
			}

			MLSparseVector[] rows = new MLSparseVector[nUsers];
			IntStream.range(0, nUsers).parallel().forEach(userIndex -> {
				Map<Integer, MLMatrixElement> userItemMap = rowMaps[userIndex];
				if (userItemMap == null) {
					return;
				}

				int[] indexes = new int[userItemMap.size()];
				float[] values = new float[userItemMap.size()];
				long[] dates = null;
				if (csvIndexes[DATE_INDEX] != null) {
					dates = new long[userItemMap.size()];
				}
				int index = 0;
				for (Map.Entry<Integer, MLMatrixElement> entry : userItemMap
						.entrySet()) {
					indexes[index] = entry.getValue().getColIndex();
					values[index] = entry.getValue().getValue();
					if (csvIndexes[DATE_INDEX] != null) {
						dates[index] = entry.getValue().getDate();
					}
					index++;
				}
				rows[userIndex] = new MLSparseVector(indexes, values, dates,
						nItems);
			});

			return new MLSparseMatrixAOO(rows, nItems);
		}
	}

	public static MLSparseMatrix loadFromLIBSVM(final String inFile,
			final int nRows) throws FileNotFoundException, IOException {

		MLTimer timer = new MLTimer("loadFromLIBSVM", nRows);
		MLSparseVector[] rows = new MLSparseVector[nRows];
		try (BufferedReader reader = new BufferedReader(
				new FileReader(inFile))) {
			timer.tic();
			String line = null;
			int count = 0;
			while ((line = reader.readLine()) != null) {
				String[] split = line.split("\\s+");
				count++;
				if (count % 100_000 == 0) {
					timer.tocLoop(count);
				}

				if (split.length < 2) {
					continue;
				}

				int rowIndex = Integer.parseInt(split[0]);
				int[] indexes = new int[split.length - 1];
				float[] values = new float[split.length - 1];
				for (int i = 1; i < split.length; i++) {
					String[] splitFeat = split[i].split(":");
					indexes[i - 1] = Integer.parseInt(splitFeat[0]);
					values[i - 1] = Float.parseFloat(splitFeat[1]);
				}
				rows[rowIndex] = new MLSparseVector(indexes, values, null, 0);
			}
		}
		timer.tocLoop(nRows);

		MLSparseMatrix matrix = new MLSparseMatrixAOO(rows, 0);
		matrix.inferAndSetNCols();
		timer.toc("loaded " + matrix.getNRows() + "x" + matrix.getNCols());

		return matrix;
	}
}
