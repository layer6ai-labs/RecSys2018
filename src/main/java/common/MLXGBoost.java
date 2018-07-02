package common;

import ml.dmlc.xgboost4j.LabeledPoint;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import common.MLConcurrentUtils.Async;

public class MLXGBoost {

	public static class MLXGBoostFeature {

		public static class ScoreComparator
				implements Comparator<MLXGBoostFeature> {

			private boolean decreasing;

			public ScoreComparator(final boolean decreasingP) {
				this.decreasing = decreasingP;
			}

			@Override
			public int compare(final MLXGBoostFeature e1,
					final MLXGBoostFeature e2) {
				if (this.decreasing == true) {
					return Double.compare(e2.score, e1.score);
				} else {
					return Double.compare(e1.score, e2.score);
				}
			}
		}

		private String name;
		private double score;

		public MLXGBoostFeature(final String nameP, final double scoreP) {
			this.name = nameP;
			this.score = scoreP;
		}

		public String getName() {
			return this.name;
		}

		public double getScore() {
			return this.score;
		}
	}

	public static MLXGBoostFeature[] analyzeFeatures(final String modelFile,
			final String featureFile) throws Exception {

		Booster model = XGBoost.loadModel(modelFile);

		List<String> temp = new LinkedList<String>();
		try (BufferedReader reader = new BufferedReader(
				new FileReader(featureFile))) {
			String line;
			while ((line = reader.readLine()) != null) {
				temp.add(line);
			}
		}

		// get feature importance scores
		String[] featureNames = new String[temp.size()];
		temp.toArray(featureNames);
		int[] importances = MLXGBoost.getFeatureImportance(model, featureNames);

		// sort features by their importance
		MLXGBoostFeature[] sortedFeatures = new MLXGBoostFeature[featureNames.length];
		for (int i = 0; i < featureNames.length; i++) {
			sortedFeatures[i] = new MLXGBoostFeature(featureNames[i],
					importances[i]);
		}
		Arrays.sort(sortedFeatures, new MLXGBoostFeature.ScoreComparator(true));

		return sortedFeatures;
	}

	public static Async<Booster> asyncModel(final String modelFile) {
		return asyncModel(modelFile, 0);
	}

	public static Async<Booster> asyncModel(final String modelFile,
			final int nthread) {
		// load xgboost model
		final Async<Booster> modelAsync = new Async<Booster>(() -> {
			try {
				Booster bst = XGBoost.loadModel(modelFile);
				if (nthread > 0) {
					bst.setParam("nthread", nthread);
				}
				return bst;
			} catch (XGBoostError e) {
				e.printStackTrace();
				return null;
			}
		}, Booster::dispose);
		return modelAsync;
	}

	public static int[] getFeatureImportance(final Booster model,
			final String[] featNames) throws XGBoostError {

		int[] importances = new int[featNames.length];
		// NOTE: not used feature are dropped here
		Map<String, Integer> importanceMap = model.getFeatureScore(null);

		for (Map.Entry<String, Integer> entry : importanceMap.entrySet()) {
			// get index from f0, f1 feature name output from xgboost
			int index = Integer.parseInt(entry.getKey().substring(1));
			importances[index] = entry.getValue();
		}

		return importances;
	}

	public static DMatrix toDMatrix(final MLSparseMatrix matrix)
			throws XGBoostError {

		final int nnz = (int) matrix.getNNZ();
		final int nRows = matrix.getNRows();
		final int nCols = matrix.getNCols();

		long[] rowIndex = new long[nRows + 1];
		int[] indexesFlat = new int[nnz];
		float[] valuesFlat = new float[nnz];

		int cur = 0;
		for (int i = 0; i < nRows; i++) {
			MLSparseVector row = matrix.getRow(i);
			if (row == null) {
				rowIndex[i] = cur;
				continue;
			}
			int[] indexes = row.getIndexes();
			int rowNNZ = indexes.length;
			if (rowNNZ == 0) {
				rowIndex[i] = cur;
				continue;
			}
			float[] values = row.getValues();
			rowIndex[i] = cur;

			for (int j = 0; j < rowNNZ; j++, cur++) {
				indexesFlat[cur] = indexes[j];
				valuesFlat[cur] = values[j];
			}
		}
		rowIndex[nRows] = cur;
		return new DMatrix(rowIndex, indexesFlat, valuesFlat,
				DMatrix.SparseType.CSR, nCols);
	}

	public static String toLIBSVMString(final LabeledPoint vec) {
		float target = vec.label();
		StringBuilder builder = new StringBuilder();
		if (target == (int) target) {
			builder.append((int) target);
		} else {
			builder.append(String.format("%.5f", target));
		}
		for (int i = 0; i < vec.indices().length; i++) {
			float val = vec.values()[i];
			if (val == Math.round(val)) {
				builder.append(" " + (vec.indices()[i]) + ":" + ((int) val));
			} else {
				builder.append(" " + (vec.indices()[i]) + ":"
						+ String.format("%.5f", val));
			}
		}
		return builder.toString();
	}

}
