package main;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import common.EvaluatorBinaryNDCG;
import common.EvaluatorCF;
import common.EvaluatorClicks;
import common.EvaluatorRPrecision;
import common.EvaluatorRPrecisionArtist;
import common.FloatElement;
import common.MLConcurrentUtils.Async;
import common.MLDenseMatrix;
import common.MLDenseVector;
import common.MLSparseFeature;
import common.MLSparseMatrix;
import common.MLSparseMatrixAOO;
import common.MLSparseVector;
import common.MLTimer;
import common.MLXGBoost;
import common.ResultCF;
import common.SplitterCF;
import main.ParsedData.PlaylistFeature;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;

public class XGBModel {

	public static class XGBModelParams {

		public int nPosExamples = 20;
		public int nNegExamples = 20;
		public int candidateOffset2StageTR = 0;
		public int nCandidates2StageTR = 1000;
		public int nCandidates2StageVL = 1000;
		public int nativeBatchSize = 500;
		public int targetRankingSize = 500;

		public float validFrac = 0.1f;
		public String xgbModel;
		public String rankingFile;

		public boolean doCreative = false;
	}

	private static MLTimer timer = new MLTimer("XGBModel");
	static {
		timer.toc();
	}

	private ParsedData data;
	private XGBModelParams params;
	private Latents latents;
	private SplitterCF split;
	private EvaluatorCF[] evaluators;
	private FeatureExtractor featureExtractor;
	private MLSparseMatrix R;
	private MLSparseMatrix Rt;
	private FloatElement[][] rankingsBlend;

	public XGBModel(final ParsedData dataP, final XGBModelParams paramsP,
			final Latents latentsP, final SplitterCF splitP) throws Exception {
		this.data = dataP;
		this.params = paramsP;
		this.latents = latentsP;
		this.split = splitP;

		this.evaluators = new EvaluatorCF[] {
				new EvaluatorBinaryNDCG(new int[] { 100, 200, 300, 400, 500 }),
				new EvaluatorRPrecisionArtist(new int[] { 500 },
						this.data.songFeatures
								.get(ParsedData.SongFeature.ARTIST_ID)
								.getFeatMatrix()),
				new EvaluatorRPrecision(new int[] { 500 }),
				new EvaluatorClicks(new int[] { 500 }) };

		this.R = this.split.getRstrain().get(ParsedData.INTERACTION_KEY);
		this.Rt = this.R.transpose();
		timer.toc("Rt done");

		// feature extractor
		if (this.params.doCreative == true) {
			this.featureExtractor = new FeatureExtractor(
					this.data.playlistFeatures, this.data.songFeatures,
					this.data.songExtraInfoFeatures, this.R, this.Rt,
					this.latents, timer);
		} else {
			this.featureExtractor = new FeatureExtractor(
					this.data.playlistFeatures, this.data.songFeatures, null,
					this.R, this.Rt, this.latents, timer);
		}
		timer.toc("featureExtractor init done");

		this.rankingsBlend = this.getRankingsBlendUU();
		timer.toc("ranking blend done");
	}

	public void extractFeatures2Stage(final String outPath) throws Exception {
		final MLSparseMatrix Rvalid = this.split.getRsvalid()
				.get(ParsedData.INTERACTION_KEY);
		this.params.nCandidates2StageTR = 1_000;

		// get all rows that 2nd stage can train on
		List<Integer> temp = new LinkedList<Integer>();
		for (int i = 0; i < Rvalid.getNRows(); i++) {
			if (Rvalid.getRow(i) != null) {
				temp.add(i);
			}
		}
		int[] validRowIndexes = new int[temp.size()];
		int arrIndex = 0;
		for (int index : temp) {
			validRowIndexes[arrIndex] = index;
			arrIndex++;
		}
		timer.toc("2nd stage n training rows: " + validRowIndexes.length);

		AtomicInteger rowCounter = new AtomicInteger(0);
		AtomicInteger totalPos = new AtomicInteger(0);
		AtomicInteger totalPosSampled = new AtomicInteger(0);
		AtomicInteger totalNegSampled = new AtomicInteger(0);
		try (BufferedWriter writerTrain = new BufferedWriter(
				new FileWriter(outPath + "/trainXGB"));
				BufferedWriter writerValid = new BufferedWriter(
						new FileWriter(outPath + "/validXGB"))) {

			int batchSize = 20_000;
			int nBatches = (int) Math
					.ceil(((double) validRowIndexes.length / batchSize));
			for (int batch = 0; batch < nBatches; batch++) {
				int batchStart = batch * batchSize;
				int batchEnd = Math.min((batch + 1) * batchSize,
						validRowIndexes.length);

				int[] batchIndexes = new int[batchEnd - batchStart];
				for (int i = batchStart; i < batchEnd; i++) {
					batchIndexes[i - batchStart] = validRowIndexes[i];
				}

				IntStream.range(0, batchIndexes.length).parallel()
						.forEach(index -> {
							int cur = rowCounter.incrementAndGet();
							if (cur % 10_000 == 0) {
								timer.tocLoop("extractFeatures", cur);
							}

							int rowIndex = batchIndexes[index];
							MLSparseVector validRow = Rvalid.getRow(rowIndex);
							if (validRow == null) {
								return;
							}
							int[] validIndices = validRow.getIndexes();
							FloatElement[] rankingBlend = this.rankingsBlend[rowIndex];

							// select positive and negative indices
							List<Integer> posIndexes = new LinkedList<Integer>();
							List<Integer> negIndexes = new LinkedList<Integer>();
							for (int i = 0; i < Math.min(rankingBlend.length,
									this.params.nCandidates2StageTR); i++) {
								FloatElement element = rankingBlend[i];
								if (Arrays.binarySearch(validIndices,
										element.getIndex()) >= 0) {
									posIndexes.add(i);

								} else {
									negIndexes.add(i);
								}
							}

							boolean isValid = Arrays.binarySearch(
									this.split.getValidRowIndexes(),
									rowIndex) >= 0;
							totalPos.addAndGet(posIndexes.size());
							if (isValid == false && posIndexes.size() == 0) {
								return;
							}

							// randomly sample positive indexes
							StringBuilder builder = new StringBuilder();
							Collections.shuffle(posIndexes,
									new Random(rowIndex));

							cur = 0;
							for (int posIndex : posIndexes) {
								if (isValid == true) {
									if (cur >= 20) {
										break;
									}
								} else {
									if (cur >= this.params.nPosExamples) {
										break;
									}
								}

								MLSparseVector feats = this.featureExtractor
										.extractFeaturesV1(rowIndex,
												rankingBlend[posIndex]
														.getIndex(),
												(float[]) rankingBlend[posIndex]
														.getOther());
								builder.append(
										"1" + feats.toLIBSVMString(0) + "\n");
								totalPosSampled.incrementAndGet();
								cur++;
							}

							// randomly sample negative indexes
							Collections.shuffle(negIndexes,
									new Random(rowIndex));
							cur = 0;
							for (int negIndex : negIndexes) {
								if (isValid == true) {
									if (cur >= 20) {
										break;
									}
								} else {
									if (cur >= this.params.nNegExamples) {
										break;
									}
								}

								MLSparseVector feats = this.featureExtractor
										.extractFeaturesV1(rowIndex,
												rankingBlend[negIndex]
														.getIndex(),
												(float[]) rankingBlend[negIndex]
														.getOther());
								builder.append(
										"0" + feats.toLIBSVMString(0) + "\n");
								totalNegSampled.incrementAndGet();
								cur++;
							}

							if (isValid == true) {
								synchronized (writerValid) {
									try {
										writerValid.write(builder.toString());
									} catch (Exception e) {
										e.printStackTrace();
									}
								}
							} else {
								synchronized (writerTrain) {
									try {
										writerTrain.write(builder.toString());
									} catch (Exception e) {
										e.printStackTrace();
									}
								}
							}
						});
			}
		}

		System.out.println("total pos:" + Rvalid.getNNZ() + "  pos retrieved:"
				+ totalPos.get() + "  pos sampled:" + totalPosSampled.get()
				+ "  neg sampled:" + totalNegSampled.get());

		MLSparseMatrix Rsplit = this.split.getRsvalid()
				.get(ParsedData.INTERACTION_KEY);
		for (int i = 0; i < Rsplit.getNRows(); i++) {
			if (Arrays.binarySearch(this.split.getValidRowIndexes(), i) < 0) {
				Rsplit.setRow(null, i);
			}
		}

		// sanity check
		System.out.println("BLEND");
		for (EvaluatorCF evaluator : this.evaluators) {
			ResultCF result = evaluator.evaluate(this.split,
					ParsedData.INTERACTION_KEY, this.rankingsBlend);
			System.out.println(result.toString());
		}
	}

	private FloatElement[][] getRankingItemItem(final int[] playlistIndexes,
			final int nItems, final float beta) {
		// MLSparseMatrix Rtconcat = MLSparseMatrix.concatVertical(this.R,
		// this.data.songFeatures.get(SongFeature.ARTIST_ID)
		// .getFeatMatrix().transpose(),
		// this.data.songFeatures.get(SongFeature.ALBUM_ID).getFeatMatrix()
		// .transpose())
		// .transpose();

		MLSparseMatrix Rtconcat = this.R.transpose();

		MLSparseMatrix RtconcatNorm = Rtconcat.deepCopy();
		RtconcatNorm.applyColNorm(RtconcatNorm.getColNorm(2));
		RtconcatNorm.applyRowNorm(RtconcatNorm.getRowNorm(2));

		MLDenseVector popularVec = Rtconcat.getRowSum();
		popularVec.scalarDivide(popularVec.sum());
		float[] popularity = popularVec.getValues();

		FloatElement[][] rankingsLatent = EvaluatorCF.getRankingsNative(
				this.split.getRstrain().get(ParsedData.INTERACTION_KEY),
				playlistIndexes, this.split.getValidColIndexes(),
				this.latents.U, this.latents.V, nItems, 500);

		FloatElement[][] rankings = new FloatElement[this.R.getNRows()][];
		AtomicInteger counter = new AtomicInteger(0);
		AtomicInteger counterBackfilled = new AtomicInteger(0);
		IntStream.range(0, playlistIndexes.length).parallel().forEach(index -> {
			int count = counter.incrementAndGet();
			if (count % 5_000 == 0) {
				timer.tocLoop("inferenceItemItem", count);
			}

			int rowIndex = playlistIndexes[index];
			MLSparseVector trainRow = this.R.getRow(rowIndex);

			MLDenseVector colAvg = FeatureExtractor.getRowAvg(RtconcatNorm,
					trainRow.getIndexes(), true);
			float[] scores = new float[this.R.getNCols()];
			for (int i = 0; i < scores.length; i++) {

				// note: use unnormalized row here
				MLSparseVector col = Rtconcat.getRow(i);
				if (col != null) {
					scores[i] = colAvg.mult(col)
							* (float) Math.pow(popularity[i], -(1 - beta));
				}
			}
			rankings[rowIndex] = FloatElement.topNSort(scores, nItems,
					trainRow.getIndexes());

			// back fill with latent, if necessary
			FloatElement[] rankingLatent = rankingsLatent[rowIndex];
			Set<Integer> indexes = new HashSet<Integer>();
			int cur = 0;
			for (int i = 0; i < rankings[rowIndex].length; i++) {
				FloatElement element = rankings[rowIndex][i];
				if (element.getValue() > 0f) {
					indexes.add(element.getIndex());
				} else {
					while (true) {
						FloatElement elementLatent = rankingLatent[cur];
						cur++;
						if (indexes
								.contains(elementLatent.getIndex()) == false) {
							rankings[rowIndex][i] = new FloatElement(
									elementLatent.getIndex(), 0);
							indexes.add(elementLatent.getIndex());
							counterBackfilled.incrementAndGet();
							break;
						}
					}
				}
			}
		});
		timer.tocLoop("inferenceItemItem", counter.get());
		timer.toc("n back filled " + counterBackfilled.get());

		return rankings;
	}

	public FloatElement[][] getRankingsBlendUU() throws Exception {
		final MLSparseMatrix Rvalid = this.split.getRsvalid()
				.get(ParsedData.INTERACTION_KEY);

		MLSparseMatrix Rtconcat = this.R.transpose();
		MLSparseMatrix RtconcatNorm = Rtconcat.deepCopy();
		RtconcatNorm.applyColNorm(RtconcatNorm.getColNorm(2));
		RtconcatNorm.applyRowNorm(RtconcatNorm.getRowNorm(2));

		MLDenseVector popularVec = Rtconcat.getRowSum();
		popularVec.scalarDivide(popularVec.sum());
		float[] popularity = popularVec.getValues();

		// get target rows to generate ranking for
		List<Integer> temp = new LinkedList<Integer>();
		for (int i = 0; i < Rvalid.getNRows(); i++) {
			if (Rvalid.getRow(i) != null
					|| (Arrays.binarySearch(this.data.testIndexes, i) >= 0
							&& this.R.getRow(i) != null)) {
				temp.add(i);
			}
		}
		int[] targetRowIndexes = new int[temp.size()];
		int arrIndex = 0;
		for (int index : temp) {
			targetRowIndexes[arrIndex] = index;
			arrIndex++;
		}
		timer.toc("nrakings " + targetRowIndexes.length);

		final int BLEND_NCANDS = 20_000;
		final int BLEND_N_TO_STORE = 5_000;
		final float ITEM_ITEM_BETA = 0.6f;

		int batchSize = 20_000;
		int nBatches = (int) Math
				.ceil(((double) targetRowIndexes.length / batchSize));

		final float[] WEIGHTS = new float[] { 0.3f, 0.3f, 0.1f, 0.4f };
		AtomicInteger counter = new AtomicInteger(0);
		FloatElement[][] rankingsBlend = new FloatElement[this.R.getNRows()][];
		for (int batch = 0; batch < nBatches; batch++) {
			int batchStart = batch * batchSize;
			int batchEnd = Math.min((batch + 1) * batchSize,
					targetRowIndexes.length);

			int[] batchIndexes = new int[batchEnd - batchStart];
			for (int i = batchStart; i < batchEnd; i++) {
				batchIndexes[i - batchStart] = targetRowIndexes[i];
			}

			FloatElement[][] rankingsUserUser = this.getRankingUserUser(
					batchIndexes, BLEND_NCANDS, 17_000, 0.9f);

			IntStream.range(0, batchIndexes.length).parallel()
					.forEach(index -> {
						int count = counter.incrementAndGet();
						if (count % 1_000 == 0) {
							timer.tocLoop("blend", count);
						}

						int playlistIndex = batchIndexes[index];
						MLSparseVector row = this.R.getRow(playlistIndex);

						// compute all the scores
						MLDenseVector colAvg = FeatureExtractor.getRowAvg(
								RtconcatNorm, row.getIndexes(), true);
						FloatElement[] rankingUserUser = rankingsUserUser[playlistIndex];
						FloatElement[] rankingLatent = new FloatElement[rankingUserUser.length];
						FloatElement[] rankingItemItem = new FloatElement[rankingUserUser.length];
						FloatElement[] rankingLatentCNN = new FloatElement[rankingUserUser.length];
						for (int i = 0; i < rankingUserUser.length; i++) {
							int songIndex = rankingUserUser[i].getIndex();
							float score = 0;

							// user-user
							rankingUserUser[i]
									.setOther(rankingUserUser[i].getValue());

							// latent
							score = this.latents.U.getRow(playlistIndex)
									.mult(this.latents.V.getRow(songIndex));
							rankingLatent[i] = new FloatElement(songIndex,
									score);
							rankingLatent[i].setOther(score);

							// item-item
							score = 0;
							MLSparseVector col = Rtconcat.getRow(songIndex);
							if (col != null) {
								score = colAvg.mult(col) * (float) Math.pow(
										popularity[songIndex],
										-(1 - ITEM_ITEM_BETA));
							}
							rankingItemItem[i] = new FloatElement(songIndex,
									score);
							rankingItemItem[i].setOther(score);

							// latent CNN
							score = this.latents.Ucnn.getRow(playlistIndex)
									.mult(this.latents.Vcnn.getRow(songIndex));
							rankingLatentCNN[i] = new FloatElement(songIndex,
									score);
							rankingLatentCNN[i].setOther(score);

						}

						FloatElement.standardize(rankingUserUser);
						FloatElement.standardize(rankingLatent);
						FloatElement.standardize(rankingItemItem);
						FloatElement.standardize(rankingLatentCNN);

						FloatElement[] rankingBlend = new FloatElement[BLEND_NCANDS];
						for (int i = 0; i < BLEND_NCANDS; i++) {
							float blendScore = WEIGHTS[0]
									* rankingUserUser[i].getValue()
									+ WEIGHTS[1] * rankingLatent[i].getValue()
									+ WEIGHTS[2] * rankingItemItem[i].getValue()
									+ WEIGHTS[3]
											* rankingLatentCNN[i].getValue();

							rankingBlend[i] = new FloatElement(
									rankingLatent[i].getIndex(), blendScore);
							rankingBlend[i].setOther(new float[] {

									(float) rankingUserUser[i].getOther(),

									(float) rankingLatent[i].getOther(),

									(float) rankingItemItem[i].getOther(),

									(float) rankingLatentCNN[i].getOther(),

									blendScore });
						}

						Arrays.sort(rankingBlend,
								new FloatElement.ValueComparator(true));
						rankingsBlend[playlistIndex] = Arrays
								.copyOfRange(rankingBlend, 0, BLEND_N_TO_STORE);

					});
		}

		MLSparseMatrix Rsplit = this.split.getRsvalid()
				.get(ParsedData.INTERACTION_KEY);
		for (int i = 0; i < Rsplit.getNRows(); i++) {
			if (Arrays.binarySearch(this.split.getValidRowIndexes(), i) < 0) {
				Rsplit.setRow(null, i);
			}
		}

		// sanity check
		System.out.println("BLENDUU");
		for (EvaluatorCF evaluator : this.evaluators) {
			ResultCF result = evaluator.evaluate(this.split,
					ParsedData.INTERACTION_KEY, rankingsBlend);
			System.out.println(result.toString());
		}

		return rankingsBlend;
	}

	private FloatElement[][] getRankingUserUser(final int[] playlistIndexes,
			final int nItems, final int topK, final float beta) {

		MLSparseMatrix Rnorm = MLSparseMatrix.concatHorizontal(this.R,
				this.data.playlistFeatures.get(PlaylistFeature.NAME_REGEXED)
						.getFeatMatrix());
		Rnorm.applyRowNorm(Rnorm.getRowNorm(2));
		Rnorm.applyColNorm(Rnorm.getColNorm(2));

		FloatElement[][] rankingsLatent = EvaluatorCF.getRankingsNative(
				this.split.getRstrain().get(ParsedData.INTERACTION_KEY),
				playlistIndexes, this.split.getValidColIndexes(),
				this.latents.U, this.latents.V, nItems, 500);

		// NOTE: Rnorm can be different from R here
		MLSparseMatrix Rnormt = Rnorm.transpose();

		MLDenseVector popularVec = this.R.getColSum();
		popularVec.scalarDivide(popularVec.sum());
		float[] popularity = popularVec.getValues();

		FloatElement[][] rankings = new FloatElement[this.R.getNRows()][];
		AtomicInteger counter = new AtomicInteger(0);
		AtomicInteger counterBackfilled = new AtomicInteger(0);
		IntStream.range(0, playlistIndexes.length).parallel().forEach(index -> {
			int count = counter.incrementAndGet();
			if (count % 5_000 == 0) {
				timer.tocLoop("inferenceUserUser", count);
			}

			int rowIndex = playlistIndexes[index];
			MLSparseVector trainRowNorm = Rnorm.getRow(rowIndex);

			// get indexes of all playlists that intersect with current
			Set<Integer> playlistIntersect = new HashSet<Integer>();
			for (int songIndex : trainRowNorm.getIndexes()) {
				MLSparseVector row = Rnormt.getRow(songIndex);
				if (row == null) {
					continue;
				}
				for (int playlistIndex : row.getIndexes()) {
					playlistIntersect.add(playlistIndex);
				}
			}

			// compute user-user score
			FloatElement[] elements = new FloatElement[playlistIntersect
					.size()];
			int cur = 0;
			for (int playlistIndex : playlistIntersect) {

				float weight = trainRowNorm
						.multiply(Rnorm.getRow(playlistIndex));
				elements[cur] = new FloatElement(playlistIndex, weight);
				cur++;
			}
			Arrays.sort(elements, new FloatElement.ValueComparator(true));

			float[] scores = new float[this.R.getNCols()];
			for (int i = 0; i < Math.min(topK, elements.length); ++i) {
				float weight = elements[i].getValue();
				MLSparseVector row = this.R.getRow(elements[i].getIndex());
				if (row == null) {
					continue;
				}

				int[] indexes = row.getIndexes();
				for (int j = 0; j < indexes.length; j++) {
					scores[indexes[j]] += weight
							* Math.pow(popularity[indexes[j]], -(1 - beta));
				}
			}
			rankings[rowIndex] = FloatElement.topNSort(scores, nItems,
					this.R.getRow(rowIndex).getIndexes());

			// back fill with latent, if necessary
			FloatElement[] rankingLatent = rankingsLatent[rowIndex];
			Set<Integer> indexes = new HashSet<Integer>();
			cur = 0;

			for (int i = 0; i < rankings[rowIndex].length; i++) {
				FloatElement element = rankings[rowIndex][i];
				if (element.getValue() > 0f) {
					indexes.add(element.getIndex());
				} else {
					while (true) {
						FloatElement elementLatent = rankingLatent[cur];
						cur++;
						if (indexes
								.contains(elementLatent.getIndex()) == false) {
							rankings[rowIndex][i] = new FloatElement(
									elementLatent.getIndex(), 0);
							indexes.add(elementLatent.getIndex());
							counterBackfilled.incrementAndGet();
							break;
						}
					}
				}
			}
		});
		timer.toc("n back filled " + counterBackfilled.get());

		return rankings;
	}

	public void inference2Stage() throws Exception {

		final Async<Booster> xgbModelFactory = MLXGBoost
				.asyncModel(this.params.xgbModel);

		final int[] targetRowIndices = this.split.getValidRowIndexes();
		FloatElement[][] rankings2Stage = new FloatElement[this.R.getNRows()][];
		AtomicInteger rowCounter = new AtomicInteger(0);

		this.params.nCandidates2StageVL = 5_000;

		IntStream.range(0, targetRowIndices.length).parallel()
				.forEach(index -> {
					int cur = rowCounter.incrementAndGet();
					if (cur % 1_000 == 0) {
						timer.tocLoop("inference2Stage", cur);
					}

					int playlistIndex = targetRowIndices[index];
					FloatElement[] rankingBlend = rankingsBlend[playlistIndex];

					// re-rank first stage with xgb
					MLSparseVector[] feats = new MLSparseVector[this.params.nCandidates2StageVL];
					for (int i = 0; i < this.params.nCandidates2StageVL; i++) {
						feats[i] = this.featureExtractor.extractFeaturesV1(
								playlistIndex, rankingBlend[i].getIndex(),
								(float[]) rankingBlend[i].getOther());

					}

					FloatElement[] ranking2Stage = new FloatElement[this.params.nCandidates2StageVL];
					synchronized (this) {
						DMatrix xgbMat = null;
						try {
							xgbMat = MLXGBoost.toDMatrix(new MLSparseMatrixAOO(
									feats, feats[0].getLength()));

							float[][] xgbPreds = xgbModelFactory.get()
									.predict(xgbMat);
							for (int i = 0; i < this.params.nCandidates2StageVL; i++) {
								ranking2Stage[i] = new FloatElement(
										rankingBlend[i].getIndex(),
										xgbPreds[i][0]);

							}
						} catch (Exception e) {
							e.printStackTrace();
							throw new RuntimeException("xgb failed");

						} finally {
							xgbMat.dispose();
						}

					}

					// sort and store for sanity check
					if (this.params.rankingFile == null) {
						Arrays.sort(rankingBlend,
								new FloatElement.ValueComparator(true));
						rankingsBlend[playlistIndex] = rankingBlend;
					}

					Arrays.sort(ranking2Stage,
							new FloatElement.ValueComparator(true));
					rankings2Stage[playlistIndex] = ranking2Stage;
				});
		timer.tocLoop("inference2Stage", rowCounter.get());

		if (this.params.rankingFile != null) {
			MLSparseMatrix Rsplit = this.split.getRsvalid()
					.get(ParsedData.INTERACTION_KEY);
			for (int i = 0; i < Rsplit.getNRows(); i++) {
				if (Arrays.binarySearch(this.split.getValidRowIndexes(),
						i) < 0) {
					Rsplit.setRow(null, i);
				}
			}
		}

		// blend
		for (EvaluatorCF evaluator : this.evaluators) {
			ResultCF result = evaluator.evaluate(this.split,
					ParsedData.INTERACTION_KEY, rankingsBlend);
			System.out.println("BLEND " + result.toString());
		}

		// 2-stage
		for (EvaluatorCF evaluator : this.evaluators) {
			ResultCF result = evaluator.evaluate(this.split,
					ParsedData.INTERACTION_KEY, rankings2Stage);
			System.out.println("2STAGE " + result.toString());
		}
	}

	public void inferenceBlendUU() {

		MLSparseMatrix Rtconcat = this.R.transpose();

		MLSparseMatrix RtconcatNorm = Rtconcat.deepCopy();
		RtconcatNorm.applyColNorm(RtconcatNorm.getColNorm(2));
		RtconcatNorm.applyRowNorm(RtconcatNorm.getRowNorm(2));

		MLDenseVector popularVec = Rtconcat.getRowSum();
		popularVec.scalarDivide(popularVec.sum());
		float[] popularity = popularVec.getValues();

		final int N_SONGS_TO_RANK = 20_000;
		final float ITEM_ITEM_BETA = 0.6f;

		int[] validRowIndexes = this.split.getValidRowIndexes();
		FloatElement[][] rankingsUserUser = this.getRankingUserUser(
				validRowIndexes, N_SONGS_TO_RANK, 17_000, 0.9f);

		timer.toc("rankingsUserUser done");

		System.out.println("USER-USER");
		for (EvaluatorCF evaluator : this.evaluators) {
			ResultCF result = evaluator.evaluate(this.split,
					ParsedData.INTERACTION_KEY, rankingsUserUser);
			System.out.println(result.toString());
		}
		System.out.println();

		FloatElement[][] rankingsLatent = new FloatElement[this.R.getNRows()][];
		FloatElement[][] rankingsLatentCNN = new FloatElement[this.R
				.getNRows()][];
		FloatElement[][] rankingsItemItem = new FloatElement[this.R
				.getNRows()][];
		AtomicInteger counter = new AtomicInteger(0);
		IntStream.range(0, validRowIndexes.length).parallel().forEach(index -> {
			int count = counter.incrementAndGet();
			if (count % 1_000 == 0) {
				timer.tocLoop("inference", count);
			}

			int playlistIndex = validRowIndexes[index];
			MLSparseVector row = this.R.getRow(playlistIndex);

			// compute all the scores
			MLDenseVector colAvg = FeatureExtractor.getRowAvg(RtconcatNorm,
					row.getIndexes(), true);
			FloatElement[] rankingUserUser = rankingsUserUser[playlistIndex];
			FloatElement[] rankingLatent = new FloatElement[N_SONGS_TO_RANK];
			FloatElement[] rankingLatentCNN = new FloatElement[N_SONGS_TO_RANK];
			FloatElement[] rankingItemItem = new FloatElement[N_SONGS_TO_RANK];
			for (int i = 0; i < N_SONGS_TO_RANK; i++) {
				int songIndex = rankingUserUser[i].getIndex();
				float score = 0;

				// latent
				score = this.latents.U.getRow(playlistIndex)
						.mult(this.latents.V.getRow(songIndex));
				rankingLatent[i] = new FloatElement(songIndex, score);

				// item-item
				score = 0;
				MLSparseVector col = Rtconcat.getRow(songIndex);
				if (col != null) {
					score = colAvg.mult(col) * (float) Math
							.pow(popularity[songIndex], -(1 - ITEM_ITEM_BETA));
				}
				rankingItemItem[i] = new FloatElement(songIndex, score);

				// latent CNN
				score = this.latents.Ucnn.getRow(playlistIndex)
						.mult(this.latents.Vcnn.getRow(songIndex));
				rankingLatentCNN[i] = new FloatElement(songIndex, score);
			}

			rankingsLatent[playlistIndex] = rankingLatent;
			rankingsItemItem[playlistIndex] = rankingItemItem;
			rankingsLatentCNN[playlistIndex] = rankingLatentCNN;
		});

		FloatElement.standardize(rankingsUserUser);
		FloatElement.standardize(rankingsLatent);
		FloatElement.standardize(rankingsItemItem);
		FloatElement.standardize(rankingsLatentCNN);

		double bestScore = 0;
		float[] weights = new float[] { 0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f };
		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights.length; j++) {
				for (int k = 0; k < weights.length; k++) {
					for (int m = 0; m < weights.length; m++) {

						float[] curWeights = new float[] { weights[i],
								weights[j], weights[k], weights[m] };
						// float[] curWeights = new float[] { 0.4f, 0.5f, 0.2f,
						// 0.5f };

						FloatElement[][] blend = new FloatElement[this.R
								.getNRows()][];
						IntStream.range(0, validRowIndexes.length).parallel()
								.forEach(index -> {
									int rowIndex = validRowIndexes[index];
									FloatElement[] rankingBlend = new FloatElement[N_SONGS_TO_RANK];
									for (int s = 0; s < N_SONGS_TO_RANK; s++) {
										float blendScore = 0;
										blendScore = curWeights[0]
												* rankingsUserUser[rowIndex][s]
														.getValue()
												+ curWeights[1]
														* rankingsLatent[rowIndex][s]
																.getValue()
												+ curWeights[2]
														* rankingsItemItem[rowIndex][s]
																.getValue()
												+ curWeights[3]
														* rankingsLatentCNN[rowIndex][s]
																.getValue();

										rankingBlend[s] = new FloatElement(
												rankingsUserUser[rowIndex][s]
														.getIndex(),
												blendScore);
									}
									Arrays.sort(rankingBlend,
											new FloatElement.ValueComparator(
													true));
									blend[rowIndex] = rankingBlend;
								});
						ResultCF result = this.evaluators[0].evaluate(
								this.split, ParsedData.INTERACTION_KEY, blend);
						double[] resultVals = result.get();
						if (resultVals[resultVals.length - 1] > bestScore) {
							bestScore = resultVals[resultVals.length - 1];

							System.out.println(
									"WEIGHTS " + Arrays.toString(curWeights));
							for (EvaluatorCF evaluator : this.evaluators) {
								result = evaluator.evaluate(this.split,
										ParsedData.INTERACTION_KEY, blend);
								System.out.println(result.toString());
							}
							System.out.println();
						}
					}
				}
			}
		}
	}

	public void submission2Stage(final String outFile) throws Exception {

		int[] testColIndexes = new int[this.data.interactions.getNCols()];
		for (int i = 0; i < this.data.interactions.getNCols(); i++) {
			testColIndexes[i] = i;
		}

		this.params.nCandidates2StageVL = 5_000;

		final Async<Booster> xgbModelFactory = MLXGBoost
				.asyncModel(this.params.xgbModel);

		AtomicInteger count2Stage = new AtomicInteger(0);
		try (BufferedWriter writer = new BufferedWriter(
				new FileWriter(outFile))) {
			if (this.params.doCreative == true) {
				writer.write(getTeamInfoCreative() + "\n");
			} else {
				writer.write(getTeamInfo() + "\n");
			}

			AtomicInteger rowCounter = new AtomicInteger(0);
			IntStream.range(0, this.data.testIndexes.length).parallel()
					.forEach(index -> {
						int cur = rowCounter.incrementAndGet();
						if (cur % 1_000 == 0) {
							timer.tocLoop("submission2Stage", cur);
						}

						int playlistIndex = this.data.testIndexes[index];
						MLSparseVector playlistRow = this.R
								.getRow(playlistIndex);
						if (playlistRow == null) {
							// cold start
							return;
						}
						count2Stage.incrementAndGet();

						// re-rank first stage with xgb
						FloatElement[] rankingBlend = rankingsBlend[playlistIndex];
						MLSparseVector[] feats = new MLSparseVector[this.params.nCandidates2StageVL];
						for (int i = 0; i < this.params.nCandidates2StageVL; i++) {
							feats[i] = this.featureExtractor.extractFeaturesV1(
									playlistIndex, rankingBlend[i].getIndex(),
									(float[]) rankingBlend[i].getOther());

						}

						FloatElement[] ranking2Stage = new FloatElement[this.params.nCandidates2StageVL];
						synchronized (this) {
							DMatrix xgbMat = null;
							try {
								xgbMat = MLXGBoost
										.toDMatrix(new MLSparseMatrixAOO(feats,
												feats[0].getLength()));

								float[][] xgbPreds = xgbModelFactory.get()
										.predict(xgbMat);
								for (int i = 0; i < this.params.nCandidates2StageVL; i++) {
									ranking2Stage[i] = new FloatElement(
											rankingBlend[i].getIndex(),
											xgbPreds[i][0]);

								}
							} catch (Exception e) {
								e.printStackTrace();
								throw new RuntimeException("xgb failed");

							} finally {
								xgbMat.dispose();
							}
						}
						Arrays.sort(ranking2Stage,
								new FloatElement.ValueComparator(true));

						this.writeRanking(writer, playlistIndex, ranking2Stage);

					});
		}

		this.submissionColdLatent(outFile, this.latents.Uname,
				this.latents.Vname);
		timer.toc("submission done " + outFile);
	}

	public void submissionColdLatent(final String outFile,
			final MLDenseMatrix Ucold, final MLDenseMatrix Vcold)
			throws Exception {
		int[] testColIndexes = new int[this.data.interactions.getNCols()];
		for (int i = 0; i < this.data.interactions.getNCols(); i++) {
			testColIndexes[i] = i;
		}

		// get cold start indexes
		List<Integer> temp = new ArrayList<Integer>();
		for (int i = 0; i < this.data.testIndexes.length; i++) {
			int index = this.data.testIndexes[i];
			if (this.data.interactions.getRow(index) == null) {
				temp.add(index);
			}
		}
		int[] testRowIndexes = new int[temp.size()];
		for (int i = 0; i < temp.size(); i++) {
			testRowIndexes[i] = temp.get(i);
		}
		System.out.println("nTestColdStart:" + testRowIndexes.length);

		MLSparseFeature nameFeature = this.data.playlistFeatures
				.get(PlaylistFeature.NAME_REGEXED);

		FloatElement[][] rankingsLatent = EvaluatorCF.getRankingsNative(
				this.data.interactions, testRowIndexes, testColIndexes, Ucold,
				Vcold, 500, 500);
		timer.toc("latent done");

		float[] colSum = this.data.interactions.getColSum().getValues();
		FloatElement[] rankingPopular = FloatElement.topNSort(colSum, 500,
				new HashSet<Integer>());

		try (BufferedWriter writer = new BufferedWriter(
				new FileWriter(outFile, true))) {
			IntStream.range(0, testRowIndexes.length).parallel()
					.forEach(index -> {
						int playlistIndex = testRowIndexes[index];
						if (nameFeature.getRow(playlistIndex, false) != null) {
							this.writeRanking(writer, playlistIndex,
									rankingsLatent[playlistIndex]);

						} else {
							this.writeRanking(writer, playlistIndex,
									rankingPopular);

						}
					});
		}
	}

	private void writeRanking(final BufferedWriter writer,
			final int playlistIndex, final FloatElement[] playlistRanking) {
		StringBuilder builder = new StringBuilder();
		builder.append(this.data.playlistIds[playlistIndex]);
		for (int i = 0; i < this.params.targetRankingSize; i++) {
			builder.append(
					", " + this.data.songIds[playlistRanking[i].getIndex()]);
		}
		synchronized (writer) {
			try {
				writer.write(builder.toString() + "\n");
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	public static String getTeamInfo() {
		return "team_info,main,vl6,guangwei.yu@mail.utoronto.ca";
	}

	public static String getTeamInfoCreative() {
		return "team_info,creative,vl6,guangwei.yu@mail.utoronto.ca";
	}

}
