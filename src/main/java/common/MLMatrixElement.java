package common;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Set;

public class MLMatrixElement implements Serializable {

	public static class ColIndexComparator
			implements Comparator<MLMatrixElement> {

		private boolean decreasing;

		public ColIndexComparator(final boolean decreasingP) {
			this.decreasing = decreasingP;
		}

		@Override
		public int compare(final MLMatrixElement e1, final MLMatrixElement e2) {
			if (this.decreasing == true) {
				return Integer.compare(e2.colIndex, e1.colIndex);
			} else {
				return Integer.compare(e1.colIndex, e2.colIndex);
			}
		}
	}

	public static class DateComparator implements Comparator<MLMatrixElement> {

		private boolean decreasing;

		public DateComparator(final boolean decreasingP) {
			this.decreasing = decreasingP;
		}

		@Override
		public int compare(final MLMatrixElement e1, final MLMatrixElement e2) {
			if (this.decreasing == true) {
				return Float.compare(e2.date, e1.date);
			} else {
				return Float.compare(e1.date, e2.date);
			}
		}
	}

	public static class RowIndexComparator
			implements Comparator<MLMatrixElement> {

		private boolean decreasing;

		public RowIndexComparator(final boolean decreasingP) {
			this.decreasing = decreasingP;
		}

		@Override
		public int compare(final MLMatrixElement e1, final MLMatrixElement e2) {
			if (this.decreasing == true) {
				return Integer.compare(e2.rowIndex, e1.rowIndex);
			} else {
				return Integer.compare(e1.rowIndex, e2.rowIndex);
			}
		}
	}

	public static class ValueComparator implements Comparator<MLMatrixElement> {

		private boolean decreasing;

		public ValueComparator(final boolean decreasingP) {
			this.decreasing = decreasingP;
		}

		@Override
		public int compare(final MLMatrixElement e1, final MLMatrixElement e2) {
			if (this.decreasing == true) {
				return Float.compare(e2.value, e1.value);
			} else {
				return Float.compare(e1.value, e2.value);
			}
		}
	}

	private static final long serialVersionUID = 1078736772506670L;
	private int rowIndex;
	private int colIndex;
	private float value;
	private long date;

	public MLMatrixElement(int rowIndexP, int colIndexP, float valueP,
			long dateP) {
		this.rowIndex = rowIndexP;
		this.colIndex = colIndexP;
		this.value = valueP;
		this.date = dateP;
	}

	public int getColIndex() {
		return this.colIndex;
	}

	public long getDate() {
		return this.date;
	}

	public int getRowIndex() {
		return this.rowIndex;
	}

	public float getValue() {
		return this.value;
	}

	public void setColIndex(final int colIndexP) {
		this.colIndex = colIndexP;
	}

	public void setDate(final long dateP) {
		this.date = dateP;
	}

	public void setRowIndex(final int rowIndexP) {
		this.rowIndex = rowIndexP;
	}

	public void setValue(final float valueP) {
		this.value = valueP;
	}

	public static MLMatrixElement[] topNSort(final int rowIndex,
			final float[] vec, final int topN, final Set<Integer> exclusions) {

		final Comparator<MLMatrixElement> valAscending = new MLMatrixElement.ValueComparator(
				false);
		final Comparator<MLMatrixElement> valDescending = new MLMatrixElement.ValueComparator(
				true);

		PriorityQueue<MLMatrixElement> heap = new PriorityQueue<MLMatrixElement>(
				topN, valAscending);

		for (int i = 0; i < vec.length; i++) {
			if (exclusions != null && exclusions.contains(i) == true) {
				continue;
			}
			float val = vec[i];
			if (heap.size() < topN) {
				heap.add(new MLMatrixElement(rowIndex, i, val, 0));

			} else {
				if (heap.peek().value < val) {
					heap.poll();
					heap.add(new MLMatrixElement(rowIndex, i, val, 0));
				}
			}
		}

		MLMatrixElement[] heapArray = new MLMatrixElement[heap.size()];
		heap.toArray(heapArray);
		Arrays.sort(heapArray, valDescending);

		return heapArray;
	}

	public static MLMatrixElement[] topNSort(final MLMatrixElement[] elements,
			final int topN, final Set<Integer> exclusions) {

		final Comparator<MLMatrixElement> valAscending = new MLMatrixElement.ValueComparator(
				false);
		final Comparator<MLMatrixElement> valDescending = new MLMatrixElement.ValueComparator(
				true);

		PriorityQueue<MLMatrixElement> heap = new PriorityQueue<MLMatrixElement>(
				topN, valAscending);

		for (int i = 0; i < elements.length; i++) {
			if (exclusions != null && exclusions.contains(i) == true) {
				continue;
			}
			MLMatrixElement element = elements[i];
			if (heap.size() < topN) {
				heap.add(element);

			} else {
				if (heap.peek().value < element.value) {
					heap.poll();
					heap.add(element);
				}
			}
		}

		MLMatrixElement[] heapArray = new MLMatrixElement[heap.size()];
		heap.toArray(heapArray);
		Arrays.sort(heapArray, valDescending);

		return heapArray;
	}

	public static MLMatrixElement[] topNSortArr(final int rowIndex,
			final float[] vec, final int topN, final int[] exclusions) {
		Set<Integer> exclusionSet = new HashSet<Integer>(exclusions.length);
		for (int exclusion : exclusions) {
			exclusionSet.add(exclusion);
		}
		return topNSort(rowIndex, vec, topN, exclusionSet);
	}

}
