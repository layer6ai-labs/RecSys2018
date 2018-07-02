package common;

import org.netlib.util.intW;

import com.github.fommil.netlib.BLAS;
import com.github.fommil.netlib.LAPACK;

public class LowLevelRoutines {

	public static final int MAX_ARRAY_SIZE = Integer.MAX_VALUE - 50;

	public static void sgemm(float[] A, float[] B, float[] C, int nRowsA,
			int nColsB, int nColsA, boolean rowStoreA, boolean rowStoreB,
			float alpha, float beta) {
		// blas uses fortran out, so we do CT=(BT)(AT) instead of C=AB

		final int m = nColsB;
		final int n = nRowsA;
		final int k = nColsA;
		final int lda;
		final int ldb;
		final int ldc;
		final String transA;
		final String transB;
		if (rowStoreB) {
			transA = "N";
			lda = m;
			ldc = m;
			if (rowStoreA) {
				ldb = k;
				transB = "N";
			} else {
				ldb = n;
				transB = "T";
			}
		} else {
			transA = "T";
			lda = k;
			ldc = m;
			if (rowStoreA) {
				ldb = k;
				transB = "N";
			} else {
				ldb = n;
				transB = "T";
			}
		}
		BLAS.getInstance().sgemm(transA, transB, m, n, k, alpha, B, lda, A, ldb,
				beta, C, ldc);
	}

	public static void symmetricSolve(final float[] data, final int nRows,
			float[] b, float[] cache) {
		int[] ipiv = new int[nRows];
		intW info = new intW(0);
		LAPACK.getInstance().ssysv("L", nRows, 1, data, nRows, ipiv, b,
				b.length, cache, cache.length, info);
	}

	public static int symmInverseCacheSize(final float[] data,
			final int nRows) {
		int[] ipiv = new int[nRows];
		intW info = new intW(0);
		float[] cacheSize = new float[1];
		LAPACK.getInstance().ssytrf("L", nRows, data, nRows, ipiv, cacheSize,
				-1, info);
		return (int) cacheSize[0];
	}

}
