package main;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.Field;

import common.MLDenseMatrix;
import common.MLDenseVector;
import common.MLSparseMatrix;
import common.MLSparseVector;
import common.MLTimer;

public class SVD {

	public static class SVDParams {
		public int svdIter = 4;
		public int rank = 200;
		public String scriptPath;
		public String cachePath;
		public String cacheName = "matrix.csv";
		public String cachePName = "U.nd";
		public String cacheQName = "V.nd";
		public String cacheSName = "S.nd";
		public int shapeRows;
		public int shapeCols;

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

	public SVDParams params;
	public MLDenseMatrix P;
	public MLDenseMatrix Q;
	public MLDenseVector s;

	public SVD(final SVDParams paramsP) {
		this.params = paramsP;
	}

	public void runPythonSVD(final MLSparseMatrix matrix) {

		MLTimer timer = new MLTimer("runPythonSVD");
		timer.tic();
		Process process = null;
		try {
			String command = String.format(
					"python %s -r %s -i %s -d %s -f %s --shape %s %s",
					this.params.scriptPath, this.params.rank,
					this.params.svdIter, this.params.cachePath,
					this.params.cacheName, this.params.shapeRows,
					this.params.shapeCols);
			System.out.println(command);

			toCSV(matrix, this.params.cachePath + this.params.cacheName);
			process = Runtime.getRuntime()
					.exec(new String[] { "bash", "-c", command });
			process.waitFor();

			this.P = fromCSV(this.params.cachePath + this.params.cachePName,
					this.params.shapeRows);
			this.Q = fromCSV(this.params.cachePath + this.params.cacheQName,
					this.params.shapeCols);
		} catch (Exception e) {
			e.printStackTrace();

		} finally {
			if (process != null) {
				process.destroy();
			}
		}
	}

	public static MLDenseMatrix fromCSV(final String inFile, final int nRows) {

		MLTimer timer = new MLTimer("fromCSV");
		timer.tic();

		MLDenseVector[] rows = new MLDenseVector[nRows];
		int curRow = 0;
		try (BufferedReader reader = new BufferedReader(
				new FileReader(inFile))) {
			String line;
			while ((line = reader.readLine()) != null) {
				if (curRow % 500_000 == 0) {
					timer.tocLoop(curRow);
				}

				String[] split = line.split(",");
				float[] values = new float[split.length];
				for (int i = 0; i < split.length; i++) {
					values[i] = Float.parseFloat(split[i]);
				}
				rows[curRow] = new MLDenseVector(values);
				curRow++;
			}

			if (curRow != nRows) {
				throw new Exception("urRow != nRows");
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

		return new MLDenseMatrix(rows);
	}

	public static void toCSV(final MLSparseMatrix matrix, final String outFile)
			throws IOException {
		MLTimer timer = new MLTimer("toCSV");
		timer.tic();

		try (BufferedWriter writer = new BufferedWriter(
				new PrintWriter(outFile, "UTF-8"))) {
			int numRows = matrix.getNRows();
			for (int i = 0; i < numRows; i++) {
				if (i % 500_000 == 0) {
					timer.tocLoop(i);
				}

				MLSparseVector row = matrix.getRow(i);
				if (row != null) {
					int[] indexes = row.getIndexes();
					float[] values = row.getValues();
					for (int j = 0; j < indexes.length; j++) {
						writer.write(
								i + "," + indexes[j] + "," + values[j] + "\n");
					}
				}
			}
		}
	}
}
