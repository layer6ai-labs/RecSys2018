package common;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

public class SplitterCF implements Serializable {

	private static final long serialVersionUID = -3182298371988867241L;
	private Map<String, MLSparseMatrix> Rstrain;
	private Map<String, MLSparseMatrix> Rsvalid;
	private int[] validRowIndexes;
	private int[] validColIndexes;

	public SplitterCF() {

	}

	public Map<String, MLSparseMatrix> getRstrain() {
		return Rstrain;
	}

	public Map<String, MLSparseMatrix> getRsvalid() {
		return Rsvalid;
	}

	public int[] getValidColIndexes() {
		return validColIndexes;
	}

	public int[] getValidRowIndexes() {
		return validRowIndexes;
	}

	public void setRstrain(Map<String, MLSparseMatrix> rstrain) {
		this.Rstrain = rstrain;
	}

	public void setRsvalid(Map<String, MLSparseMatrix> rsvalid) {
		this.Rsvalid = rsvalid;
	}

	public void setValidColIndexes(int[] validColIndexes) {
		this.validColIndexes = validColIndexes;
	}

	public void setValidRowIndexes(int[] validRowIndexes) {
		this.validRowIndexes = validRowIndexes;
	}

	private void split(final Map<String, MLSparseMatrix> Rs,
			final long dateCutOff) {
		this.Rstrain = new HashMap<String, MLSparseMatrix>();
		this.Rsvalid = new HashMap<String, MLSparseMatrix>();

		for (Map.Entry<String, MLSparseMatrix> entry : Rs.entrySet()) {
			MLSparseMatrix R = entry.getValue();

			MLSparseVector[] trainRows = new MLSparseVector[R.getNRows()];
			MLSparseVector[] validRows = new MLSparseVector[R.getNRows()];

			AtomicInteger nnzTrain = new AtomicInteger(0);
			AtomicInteger nnzValid = new AtomicInteger(0);
			IntStream.range(0, R.getNRows()).parallel().forEach(rowIndex -> {
				MLSparseVector row = R.getRow(rowIndex);
				if (row == null) {
					return;
				}
				long[] dates = row.getDates();

				int nGreater = 0;
				for (int i = 0; i < dates.length; i++) {
					if (dates[i] > dateCutOff) {
						nGreater++;
					}
				}
				if (nGreater == dates.length) {
					// no training data
					return;
				}

				// split forward in time
				int jtrain = 0;
				int[] indexesTrain = new int[dates.length - nGreater];
				float[] valuesTrain = new float[dates.length - nGreater];
				long[] datesTrain = new long[dates.length - nGreater];

				int jvalid = 0;
				int[] indexesValid = new int[nGreater];
				float[] valuesValid = new float[nGreater];
				long[] datesValid = new long[nGreater];

				int[] indexes = row.getIndexes();
				float[] values = row.getValues();
				for (int j = 0; j < dates.length; j++) {
					if (dates[j] > dateCutOff) {
						// interactions after dateCutOff
						indexesValid[jvalid] = indexes[j];
						valuesValid[jvalid] = values[j];
						datesValid[jvalid] = dates[j];
						jvalid++;

					} else {
						// interactions before dateCutOff
						indexesTrain[jtrain] = indexes[j];
						valuesTrain[jtrain] = values[j];
						datesTrain[jtrain] = dates[j];
						jtrain++;
					}
				}

				trainRows[rowIndex] = new MLSparseVector(indexesTrain,
						valuesTrain, datesTrain, R.getNCols());
				nnzTrain.addAndGet(indexesTrain.length);

				if (indexesValid.length > 0) {
					// avoid empty rows
					validRows[rowIndex] = new MLSparseVector(indexesValid,
							valuesValid, datesValid, R.getNCols());
					nnzValid.addAndGet(indexesValid.length);
				}
			});

			this.Rstrain.put(entry.getKey(),
					new MLSparseMatrixAOO(trainRows, R.getNCols()));
			this.Rsvalid.put(entry.getKey(),
					new MLSparseMatrixAOO(validRows, R.getNCols()));
			System.out.println("split() valid interaction " + entry.getKey()
					+ " nnz train:" + nnzTrain.get() + " nnz valid:"
					+ nnzValid.get());
		}
	}

	public void splitByDate(final Map<String, MLSparseMatrix> Rs,
			final long dateCutOff) {

		// use all rows and all cols for validation
		int nRows = Rs.entrySet().iterator().next().getValue().getNRows();
		int nCols = Rs.entrySet().iterator().next().getValue().getNCols();

		splitByDate(Rs, dateCutOff, null, nRows, nCols, false);
	}

	public void splitByDate(final Map<String, MLSparseMatrix> Rs,
			final long dateCutOff, final Set<String> interToSkip,
			final int nValidRows, final int nValidCols,
			final boolean coldStart) {

		// generate forward in time split
		split(Rs, dateCutOff);

		// get target row and column indices
		this.validRowIndexes = getRowIndexes(interToSkip, nValidRows,
				this.Rsvalid);
		this.validColIndexes = getColIndexes(interToSkip, nValidCols,
				this.validRowIndexes, this.Rsvalid);

		if (coldStart == true) {
			// remove selected training rows to simulate cold start
			for (Map.Entry<String, MLSparseMatrix> entry : this.Rstrain
					.entrySet()) {
				MLSparseMatrix R = entry.getValue();
				for (int index : this.validRowIndexes) {
					R.setRow(null, index);
				}
			}
		}
	}

	public void splitFrac(final Map<String, MLSparseMatrix> Rs,
			final float frac, final int minToSplit,
			final Set<String> interToSkip, final boolean useDate,
			final int nValidRows, final int nValidCols) {
		this.Rstrain = new HashMap<String, MLSparseMatrix>();
		this.Rsvalid = new HashMap<String, MLSparseMatrix>();

		for (Map.Entry<String, MLSparseMatrix> entry : Rs.entrySet()) {
			MLSparseMatrix R = entry.getValue();

			MLSparseVector[] trainRows = new MLSparseVector[R.getNRows()];
			MLSparseVector[] validRows = new MLSparseVector[R.getNRows()];

			AtomicInteger nnzTrain = new AtomicInteger(0);
			AtomicInteger nnzValid = new AtomicInteger(0);
			IntStream.range(0, R.getNRows()).parallel().forEach(rowIndex -> {
				MLSparseVector row = R.getRow(rowIndex);
				if (row == null) {
					return;
				}
				int[] indexes = row.getIndexes();
				float[] values = row.getValues();
				long[] dates = row.getDates();

				int nTotal = indexes.length;
				int nInValid = 0;
				if (nTotal < minToSplit) {
					// not enough to split
					trainRows[rowIndex] = row.deepCopy();
					return;
				}

				nInValid = (int) Math.ceil(frac * nTotal);
				Set<Integer> validIndexes = new HashSet<Integer>();
				if (useDate == false) {
					// randomly generate valid indexes
					// TODO: make this deterministic
					Random random = new Random(rowIndex);
					while (validIndexes.size() < nInValid) {
						int i = random.nextInt(nTotal);
						if (validIndexes.contains(indexes[i]) == false) {
							validIndexes.add(indexes[i]);
						}
					}
				} else {
					// sort by date and take *last* frac indexes for validation
					MLMatrixElement[] elements = new MLMatrixElement[indexes.length];
					for (int i = 0; i < indexes.length; i++) {
						elements[i] = new MLMatrixElement(rowIndex, indexes[i],
								values[i], dates[i]);
					}
					Arrays.sort(elements,
							new MLMatrixElement.DateComparator(true));
					for (int i = 0; i < nInValid; i++) {
						validIndexes.add(elements[i].getColIndex());
					}
				}

				// split using validIndexes
				int jtrain = 0;
				int[] indexesTrain = new int[nTotal - nInValid];
				float[] valuesTrain = new float[nTotal - nInValid];
				long[] datesTrain = null;
				if (dates != null) {
					datesTrain = new long[nTotal - nInValid];
				}

				int jvalid = 0;
				int[] indexesValid = new int[nInValid];
				float[] valuesValid = new float[nInValid];
				long[] datesValid = null;
				if (dates != null) {
					datesValid = new long[nInValid];
				}

				for (int i = 0; i < dates.length; i++) {
					if (validIndexes.contains(indexes[i]) == true) {
						indexesValid[jvalid] = indexes[i];
						valuesValid[jvalid] = values[i];
						if (dates != null) {
							datesValid[jvalid] = dates[i];
						}
						jvalid++;

					} else {
						indexesTrain[jtrain] = indexes[i];
						valuesTrain[jtrain] = values[i];
						if (dates != null) {
							datesTrain[jtrain] = dates[i];
						}
						jtrain++;
					}
				}

				trainRows[rowIndex] = new MLSparseVector(indexesTrain,
						valuesTrain, datesTrain, R.getNCols());
				nnzTrain.addAndGet(indexesTrain.length);

				if (indexesValid.length > 0) {
					// avoid empty rows
					validRows[rowIndex] = new MLSparseVector(indexesValid,
							valuesValid, datesValid, R.getNCols());
					nnzValid.addAndGet(indexesValid.length);
				}
			});

			this.Rstrain.put(entry.getKey(),
					new MLSparseMatrixAOO(trainRows, R.getNCols()));
			this.Rsvalid.put(entry.getKey(),
					new MLSparseMatrixAOO(validRows, R.getNCols()));

			// get target row and column indices
			this.validRowIndexes = getRowIndexes(interToSkip, nValidRows,
					this.Rsvalid);
			this.validColIndexes = getColIndexes(interToSkip, nValidCols,
					this.validRowIndexes, this.Rsvalid);

			System.out.println("split() valid interaction " + entry.getKey()
					+ " nnz train:" + nnzTrain.get() + " nnz valid:"
					+ nnzValid.get());
		}
	}

	private static int[] getColIndexes(final Set<String> interToSkip,
			final int nValidCols, final int[] validRowIndexes,
			final Map<String, MLSparseMatrix> Rs) {

		int nCols = Rs.entrySet().iterator().next().getValue().getNCols();
		if (nValidCols > nCols) {
			throw new IllegalArgumentException(
					"nValidCols=" + nValidCols + "  nCols=" + nCols);
		}

		if (nValidCols == nCols) {
			// use all columns
			int[] validColIndexes = new int[nCols];
			for (int i = 0; i < nCols; i++) {
				validColIndexes[i] = i;
			}
			return validColIndexes;
		}

		// find all candidate column ids that appear in the valid set
		Set<Integer> validCols = null;
		for (Map.Entry<String, MLSparseMatrix> entry : Rs.entrySet()) {
			if (interToSkip != null
					&& interToSkip.contains(entry.getKey()) == true) {
				// skip these interaction types
				continue;
			}
			MLSparseMatrix R = entry.getValue();
			if (validCols == null) {
				validCols = new HashSet<Integer>(R.getNCols());
			}

			for (int rowIndex : validRowIndexes) {
				MLSparseVector row = R.getRow(rowIndex);
				if (row == null) {
					continue;
				}

				for (int colIndex : row.getIndexes()) {
					validCols.add(colIndex);
				}
			}
		}

		if (validCols.size() > nValidCols) {
			// randomly select nValidCols
			List<Integer> validIndexesPerm = new ArrayList<Integer>(validCols);
			Collections.shuffle(validIndexesPerm, new Random(1));

			validCols = new HashSet<Integer>();
			validCols.addAll(validIndexesPerm.subList(0, nValidCols));

		} else {
			// backfill with random sampling
			int[] colIndexesRemain = new int[nCols - validCols.size()];
			int cur = 0;
			for (int i = 0; i < nCols; i++) {
				if (validCols.contains(i) == false) {
					colIndexesRemain[cur] = i;
					cur++;
				}
			}
			MLRandomUtils.shuffle(colIndexesRemain, new Random(1));
			for (int i = 0; i < nValidCols - validCols.size(); i++) {
				validCols.add(colIndexesRemain[i]);
			}
		}

		int[] validColIndexes = new int[validCols.size()];
		int cur = 0;
		for (int index : validCols) {
			validColIndexes[cur] = index;
			cur++;
		}
		Arrays.sort(validColIndexes);
		return validColIndexes;
	}

	private static int[] getRowIndexes(final Set<String> interToSkip,
			final int nValidRows, final Map<String, MLSparseMatrix> Rs) {

		int nRows = Rs.entrySet().iterator().next().getValue().getNRows();
		if (nValidRows > nRows) {
			throw new IllegalArgumentException(
					"nValidRows=" + nValidRows + "  nRows=" + nRows);
		}

		if (nValidRows == nRows) {
			// use all rows
			int[] validRowIndexes = new int[nRows];
			for (int i = 0; i < nRows; i++) {
				validRowIndexes[i] = i;
			}
			return validRowIndexes;
		}

		// get indexes of all validation rows
		Set<Integer> validRows = null;
		for (Map.Entry<String, MLSparseMatrix> entry : Rs.entrySet()) {
			if (interToSkip != null
					&& interToSkip.contains(entry.getKey()) == true) {
				// skip these interaction types
				continue;
			}

			MLSparseMatrix R = entry.getValue();
			if (validRows == null) {
				validRows = new HashSet<Integer>(R.getNRows());
			}

			for (int i = 0; i < R.getNRows(); i++) {
				if (R.getRow(i) != null) {
					validRows.add(i);
				}
			}
		}

		// shuffle all validation row indexes and select nValidRows
		if (validRows.size() > nValidRows) {
			List<Integer> validIndexesPerm = new ArrayList<Integer>(validRows);
			Collections.shuffle(validIndexesPerm, new Random(1));

			validRows = new HashSet<Integer>();
			validRows.addAll(validIndexesPerm.subList(0, nValidRows));
		}
		int[] validRowIndexes = new int[validRows.size()];
		int cur = 0;
		for (int index : validRows) {
			validRowIndexes[cur] = index;
			cur++;
		}
		Arrays.sort(validRowIndexes);
		return validRowIndexes;
	}
}
