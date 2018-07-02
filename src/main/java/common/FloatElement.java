package common;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.stream.IntStream;

public class FloatElement implements Serializable {

	public static class IndexComparator implements Comparator<FloatElement> {

		private boolean decreasing;

		public IndexComparator(final boolean decreasingP) {
			this.decreasing = decreasingP;
		}

		@Override
		public int compare(final FloatElement e1, final FloatElement e2) {
			if (this.decreasing == true) {
				return Integer.compare(e2.index, e1.index);
			} else {
				return Integer.compare(e1.index, e2.index);
			}
		}
	}

	public static class ValueComparator implements Comparator<FloatElement> {

		private boolean decreasing;

		public ValueComparator(final boolean decreasingP) {
			this.decreasing = decreasingP;
		}

		@Override
		public int compare(final FloatElement e1, final FloatElement e2) {
			if (this.decreasing == true) {
				return Float.compare(e2.value, e1.value);
			} else {
				return Float.compare(e1.value, e2.value);
			}
		}
	}

	private static final long serialVersionUID = -4838379190571020403L;
	private int index;
	private float value;
	private Object other;

	public FloatElement(final int indexP, final float valueP) {
		this.index = indexP;
		this.value = valueP;
	}

	public FloatElement(final int indexP, final float valueP,
			final Object otherP) {
		this.index = indexP;
		this.value = valueP;
		this.other = otherP;
	}

	public int getIndex() {
		return this.index;
	}

	public Object getOther() {
		return this.other;
	}

	public float getValue() {
		return this.value;
	}

	public void setIndex(final int indexP) {
		this.index = indexP;
	}

	public void setOther(final Object otherP) {
		this.other = otherP;
	}

	public void setValue(final float valueP) {
		this.value = valueP;
	}

	public static void standardize(final FloatElement[][] scores) {
		// in place row-based score standarization
		IntStream.range(0, scores.length).parallel().forEach(i -> {
			FloatElement[] row = scores[i];
			if (row == null) {
				return;
			}

			standardize(row);
		});
	}

	public static void standardize(final FloatElement[] scores) {
		float mean = 0f;
		for (FloatElement element : scores) {
			mean += element.value;
		}
		mean = mean / scores.length;

		float std = 0f;
		for (FloatElement element : scores) {
			std += (element.value - mean) * (element.value - mean);
		}
		std = (float) Math.sqrt(std / scores.length);

		if (std > 1e-5) {
			for (FloatElement element : scores) {
				element.value = (element.value - mean) / std;
			}
		}
	}

	public static FloatElement[] topNSort(final float[] vec, final int topN,
			final int[] exclusions) {
		Set<Integer> exclusionSet = new HashSet<Integer>(exclusions.length);
		for (int exclusion : exclusions) {
			exclusionSet.add(exclusion);
		}
		return topNSort(vec, topN, exclusionSet);
	}

	public static FloatElement[] topNSort(final float[] vec, final int topN,
			final Set<Integer> exclusions) {

		final Comparator<FloatElement> valAscending = new FloatElement.ValueComparator(
				false);
		final Comparator<FloatElement> valDescending = new FloatElement.ValueComparator(
				true);

		PriorityQueue<FloatElement> heap = new PriorityQueue<FloatElement>(topN,
				valAscending);

		for (int i = 0; i < vec.length; i++) {
			if (exclusions != null && exclusions.contains(i) == true) {
				continue;
			}
			float val = vec[i];
			if (heap.size() < topN) {
				heap.add(new FloatElement(i, val));

			} else {
				if (heap.peek().value < val) {
					heap.poll();
					heap.add(new FloatElement(i, val));
				}
			}
		}

		FloatElement[] heapArray = new FloatElement[heap.size()];
		heap.toArray(heapArray);
		Arrays.sort(heapArray, valDescending);

		return heapArray;
	}

	public static FloatElement[] topNSort(final FloatElement[] vec,
			final int topN, final Set<Integer> exclusions) {

		final Comparator<FloatElement> valAscending = new FloatElement.ValueComparator(
				false);
		final Comparator<FloatElement> valDescending = new FloatElement.ValueComparator(
				true);

		PriorityQueue<FloatElement> heap = new PriorityQueue<FloatElement>(topN,
				valAscending);

		for (int i = 0; i < vec.length; i++) {
			if (exclusions != null && exclusions.contains(i) == true) {
				continue;
			}
			FloatElement element = vec[i];
			if (heap.size() < topN) {
				heap.add(element);

			} else {
				if (heap.peek().value < element.getValue()) {
					heap.poll();
					heap.add(element);
				}
			}
		}

		FloatElement[] heapArray = new FloatElement[heap.size()];
		heap.toArray(heapArray);
		Arrays.sort(heapArray, valDescending);

		return heapArray;
	}

	public static FloatElement[] topNSortArr(final FloatElement[] vec,
			final int topN, final int[] exclusions) {
		Set<Integer> exclusionSet = new HashSet<Integer>(exclusions.length);
		for (int exclusion : exclusions) {
			exclusionSet.add(exclusion);
		}
		return topNSort(vec, topN, exclusionSet);
	}

	public static FloatElement[] topNSortOffset(final float[] vec, int topN,
			final int offset, final int length, Set<Integer> exclusions) {

		final Comparator<FloatElement> valAscending = new FloatElement.ValueComparator(
				false);
		final Comparator<FloatElement> valDescending = new FloatElement.ValueComparator(
				true);
		PriorityQueue<FloatElement> heap = new PriorityQueue<>(topN,
				valAscending);

		for (int i = 0; i < length; i++) {
			if (exclusions != null && exclusions.contains(i) == true) {
				continue;
			}
			float val = vec[i + offset];
			if (heap.size() < topN) {
				heap.add(new FloatElement(i, val));

			} else {
				if (heap.peek().getValue() < val) {
					heap.poll();
					heap.add(new FloatElement(i, val));
				}
			}
		}

		FloatElement[] heapArray = new FloatElement[heap.size()];
		heap.toArray(heapArray);
		Arrays.sort(heapArray, valDescending);

		return heapArray;
	}

	public static FloatElement[] topNSortOffset(final float[] vec, int topN,
			int[] exclusionSorted, final int offset, final int length) {

		final Comparator<FloatElement> valAscending = new FloatElement.ValueComparator(
				false);
		final Comparator<FloatElement> valDescending = new FloatElement.ValueComparator(
				true);
		PriorityQueue<FloatElement> heap = new PriorityQueue<>(topN,
				valAscending);

		int skipping = exclusionSorted == null ? -1 : exclusionSorted[0];
		int skippingCur = 0;
		final int exclusionEnd = exclusionSorted == null ? 0
				: exclusionSorted.length;
		for (int i = 0; i < length; i++) {
			if (i == skipping) {
				skippingCur++;
				if (skippingCur < exclusionEnd) {
					skipping = exclusionSorted[skippingCur];
				} else {
					skipping = -1;
				}
				continue;
			}
			float val = vec[i + offset];
			if (heap.size() < topN) {
				heap.add(new FloatElement(i, val));

			} else {
				if (heap.peek().getValue() < val) {
					heap.poll();
					heap.add(new FloatElement(i, val));
				}
			}
		}

		FloatElement[] heapArray = new FloatElement[heap.size()];
		heap.toArray(heapArray);
		Arrays.sort(heapArray, valDescending);

		return heapArray;
	}

}
