package common;

import java.io.Serializable;
import java.util.Map;
import java.util.stream.IntStream;

public abstract class MLFeatureTransform implements Serializable {

	public static class ColNormTransform extends MLFeatureTransform {

		private static final long serialVersionUID = -1777920290446866227L;
		private int normType;
		private MLDenseVector colNorm;

		public ColNormTransform(final int normTypeP) {
			this.normType = normTypeP;
		}

		@Override
		public void apply(final MLSparseFeature feature) {
			MLSparseMatrix matrix = feature.getFeatMatrixTransformed();
			this.colNorm = matrix.getColNorm(this.normType);
			this.applyInference(feature);
		}

		@Override
		public String[] applyFeatureName(final String[] featureNames) {
			// do nothing
			return featureNames;
		}

		@Override
		public void applyInference(final MLSparseFeature feature) {
			MLSparseMatrix matrix = feature.getFeatMatrixTransformed();
			matrix.applyColNorm(this.colNorm);
		}

		@Override
		public void applyInference(final MLSparseVector vector) {
			if (vector == null || vector.getIndexes() == null) {
				return;
			}
			vector.applyNorm(this.colNorm);
		}
	}

	public static class ColSelectorTransform extends MLFeatureTransform {

		private static final long serialVersionUID = -4544414355983339509L;
		private Map<Integer, Integer> selectedColMap;
		private int nColsSelected;
		private int nnzCutOff;

		public ColSelectorTransform(final int nnzCutOffP) {
			this.nnzCutOff = nnzCutOffP;
		}

		@Override
		public void apply(final MLSparseFeature feature) {

			// select columns that pass nnz cutoff
			MLSparseMatrix matrix = feature.getFeatMatrixTransformed();
			this.selectedColMap = matrix.selectCols(this.nnzCutOff);

			// calculate new ncols
			this.nColsSelected = 0;
			for (Integer index : selectedColMap.values()) {
				if (this.nColsSelected < (index + 1)) {
					this.nColsSelected = index + 1;
				}
			}

			// apply column selector to feature matrix
			this.applyInference(feature);
		}

		@Override
		public String[] applyFeatureName(final String[] featureNames) {
			String[] selectedFeatNames = new String[this.selectedColMap.size()];
			for (int i = 0; i < featureNames.length; i++) {
				Integer newIndex = this.selectedColMap.get(i);
				if (newIndex != null) {
					selectedFeatNames[newIndex] = featureNames[i];
				}
			}
			return selectedFeatNames;
		}

		@Override
		public void applyInference(final MLSparseFeature feature) {
			MLSparseMatrix matrix = feature.getFeatMatrixTransformed();
			matrix.applyColSelector(this.selectedColMap, this.nColsSelected);
		}

		@Override
		public void applyInference(final MLSparseVector vector) {
			vector.applyIndexSelector(this.selectedColMap, this.nColsSelected);
		}

	}

	public static class RowNormTransform extends MLFeatureTransform {

		private static final long serialVersionUID = -1777920290446866227L;
		private int normType;

		public RowNormTransform(final int normTypeP) {
			this.normType = normTypeP;
		}

		@Override
		public void apply(final MLSparseFeature feature) {
			this.applyInference(feature);
		}

		@Override
		public String[] applyFeatureName(final String[] featureNames) {
			// do nothing
			return featureNames;
		}

		@Override
		public void applyInference(final MLSparseFeature feature) {
			MLSparseMatrix matrix = feature.getFeatMatrixTransformed();
			MLDenseVector norm = matrix.getRowNorm(this.normType);
			matrix.applyRowNorm(norm);
		}

		@Override
		public void applyInference(final MLSparseVector vector) {
			if (vector == null || vector.getIndexes() == null) {
				return;
			}
			vector.applyNorm(this.normType);
		}
	}

	public static class StandardizeTransform extends MLFeatureTransform {

		private static final long serialVersionUID = -2289862537575019481L;
		private float[] mean;
		private float[] std;
		private float cutOff;

		public StandardizeTransform(final float cutOffP) {
			this.cutOff = cutOffP;
		}

		@Override
		public void apply(final MLSparseFeature feature) {
			// compute mean
			MLSparseMatrix matrix = feature.getFeatMatrixTransformed();
			this.mean = matrix.getColSum().getValues();
			float[] colNNZ = matrix.getColNNZ().getValues();
			for (int i = 0; i < this.mean.length; i++) {
				if (colNNZ[i] > 0) {
					this.mean[i] /= colNNZ[i];
				}
			}

			// compute std
			this.std = new float[this.mean.length];
			IntStream.range(0, matrix.getNRows()).parallel()
					.forEach(rowIndex -> {
						MLSparseVector row = matrix.getRow(rowIndex);
						if (row == null) {
							return;
						}
						int[] indexes = row.getIndexes();
						float[] values = row.getValues();
						for (int i = 0; i < indexes.length; i++) {
							float diff = values[i] - this.mean[indexes[i]];
							synchronized (this.std) {
								this.std[indexes[i]] += diff * diff;
							}
						}

					});
			for (int i = 0; i < this.std.length; i++) {
				if (colNNZ[i] > 0) {
					this.std[i] = (float) Math.sqrt(this.std[i] / colNNZ[i]);
				}
			}

			// apply transform to this feature
			this.applyInference(feature);
		}

		@Override
		public String[] applyFeatureName(final String[] featureNames) {
			// do nothing
			return featureNames;
		}

		@Override
		public void applyInference(final MLSparseFeature feature) {
			MLSparseMatrix matrix = feature.getFeatMatrixTransformed();
			IntStream.range(0, matrix.getNRows()).parallel()
					.forEach(rowIndex -> {
						MLSparseVector row = matrix.getRow(rowIndex);
						if (row == null) {
							return;
						}
						this.applyInference(row);
						matrix.setRow(row, rowIndex);
					});
		}

		@Override
		public void applyInference(final MLSparseVector vector) {
			if (vector == null || vector.getIndexes() == null) {
				return;
			}

			int[] indexes = vector.getIndexes();
			float[] values = vector.getValues();
			int nnz = 0;
			for (int i = 0; i < indexes.length; i++) {
				int index = indexes[i];
				if (Math.abs(values[i] - this.mean[index]) < 1e-5) {
					values[i] = 0;
					continue;
				}
				nnz++;

				values[i] = (values[i] - this.mean[index]) / this.std[index];

				// clip standardized values
				if (values[i] > this.cutOff) {
					values[i] = this.cutOff;

				} else if (values[i] < -this.cutOff) {
					values[i] = -this.cutOff;
				}
			}
			if (nnz != values.length) {
				// remove zeros
				int[] newIndexes = new int[nnz];
				float[] newValues = new float[nnz];
				int cur = 0;
				for (int i = 0; i < indexes.length; i++) {
					if (values[i] != 0) {
						newIndexes[cur] = indexes[i];
						newValues[cur] = values[i];
						cur++;
					}
				}
				vector.setIndexes(newIndexes);
				vector.setValues(newValues);
			}
		}

	}

	private static final long serialVersionUID = 3186575390529411219L;

	public abstract void apply(final MLSparseFeature feature);

	public abstract String[] applyFeatureName(final String[] featureNames);

	public abstract void applyInference(final MLSparseFeature feature);

	public abstract void applyInference(final MLSparseVector vector);
}
