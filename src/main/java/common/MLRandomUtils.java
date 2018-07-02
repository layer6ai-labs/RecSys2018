package common;

import java.util.Random;

public class MLRandomUtils {

	public static float nextFloat(final float min, final float max,
			final Random rng) {
		return min + rng.nextFloat() * (max - min);
	}

	public static void shuffle(int[] array, final Random rng) {
		for (int i = array.length - 1; i > 0; i--) {
			int index = rng.nextInt(i + 1);
			// swap
			int element = array[index];
			array[index] = array[i];
			array[i] = element;
		}
	}

	public static void shuffle(Object[] array, int startInclusive,
			int endExclusive, final Random rng) {
		final int len = endExclusive - startInclusive;

		for (int j = len - 1; j > 0; j--) {
			int index = rng.nextInt(j + 1) + startInclusive;
			int i = j + startInclusive;
			// swap
			Object element = array[index];
			array[index] = array[i];
			array[i] = element;
		}
	}

	public static void shuffle(Object[] array, final Random rng) {
		shuffle(array, 0, array.length, rng);
	}

	public static int[] shuffleCopy(int[] array, final Random rng) {

		int[] copy = array.clone();
		shuffle(copy, rng);
		return copy;
	}
}
