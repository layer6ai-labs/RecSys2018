package common;

import java.io.Serializable;

public class ResultCF implements Comparable<ResultCF>, Serializable {
	private static final long serialVersionUID = 22998468127105885L;
	private String objective;
	private double[] result;
	private int nEval;

	public ResultCF(final String objectiveP, final double[] resultP,
			final int nEvalP) {
		this.objective = objectiveP;
		this.result = resultP;
		this.nEval = nEvalP;
	}

	@Override public int compareTo(final ResultCF o) {
		return Double.compare(this.last(), o.last());
	}

	public double[] get() {
		return this.result;
	}

	public double last() {
		return this.result[this.result.length - 1];
	}

	@Override public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append("nEval: " + this.nEval + ", ");
		builder.append(this.objective + ":");
		for (int i = 0; i < this.result.length; i++) {
			builder.append(String.format(" %.4f", this.result[i]));
		}

		return builder.toString();
	}
}
