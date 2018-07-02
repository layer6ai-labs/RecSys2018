package common;

import java.io.Serializable;
import java.util.stream.IntStream;

public class MLDenseVector implements Serializable {

	private static final long serialVersionUID = 5061781213113137196L;
	private float[] values;

	public MLDenseVector(final float[] valuesP) {
		this.values = valuesP;
	}

	public MLDenseVector add(final MLDenseVector vector) {
		float[] sum = new float[this.getLength()];
		float[] vectorValues = vector.getValues();

		IntStream.range(0, this.getLength()).parallel()
				.forEach(i -> sum[i] = this.values[i] + vectorValues[i]);

		return new MLDenseVector(sum);
	}

	public MLDenseVector deepCopy() {
		return new MLDenseVector(this.values.clone());
	}

	public int getLength() {
		return this.values.length;
	}

	public float getValue(final int index) {
		return this.values[index];
	}

	public float[] getValues() {
		return values;
	}

	public float mult(final MLDenseVector vector) {
		// multiply two dense vectors
		if (this.getLength() != vector.getLength()) {
			throw new IllegalArgumentException("vectors must be same length");
		}

		float[] vectorValues = vector.getValues();
		float product = 0f;
		for (int i = 0; i < this.values.length; i++) {
			product += this.values[i] * vectorValues[i];
		}

		return product;
	}

	public float mult(final MLSparseVector vector) {
		// multiply sparse and dense vectors
		if (this.getLength() != vector.getLength()) {
			throw new IllegalArgumentException("vectors must be same length");
		}

		int[] indexesSparse = vector.getIndexes();
		float[] valuesSparse = vector.getValues();

		float product = 0f;
		for (int i = 0; i < indexesSparse.length; i++) {
			if (this.values[indexesSparse[i]] != 0) {
				product += valuesSparse[i] * this.values[indexesSparse[i]];
			}
		}

		return product;
	}

	public void scalarDivide(final float f) {
		// divide by a scalar
		for (int i = 0; i < this.values.length; i++) {
			this.values[i] = this.values[i] / f;
		}
	}

	public void scalarMult(final float f) {
		// multiply by a scalar
		for (int i = 0; i < this.values.length; i++) {
			this.values[i] = this.values[i] * f;
		}
	}

	public void setValues(final float[] values) {
		this.values = values;
	}

	public float sum() {
		float sum = 0;
		for (float value : values) {
			sum += value;
		}

		return sum;
	}

	public MLSparseVector toSparse() {
		int nnz = 0;
		for (float value : this.values) {
			if (value != 0) {
				nnz++;
			}
		}
		if (nnz == 0) {
			return new MLSparseVector(null, null, null, this.getLength());
		}

		int[] indexes = new int[nnz];
		float[] values = new float[nnz];
		int cur = 0;
		for (int i = 0; i < this.values.length; i++) {
			if (this.values[i] != 0) {
				indexes[cur] = i;
				values[cur] = this.values[i];
				cur++;
			}
		}

		return new MLSparseVector(indexes, values, null, this.getLength());
	}

}
