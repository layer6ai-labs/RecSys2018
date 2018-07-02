package main;

import common.ALS;
import common.ALS.ALSParams;
import common.MLTimer;
import common.SplitterCF;

public class Executor {

	public static void main(final String[] args) {
		try {
			String trainPath = "/media/mvolkovs/external4TB/Data/recsys2018/data/train";
			String testFile = "/media/mvolkovs/external4TB/Data/recsys2018/data/test/challenge_set.json";
			String extraInfoPath = "/media/mvolkovs/external4TB/Data/recsys2018/data/song_audio_features.txt";
			String pythonScriptPath = "/home/mvolkovs/projects/vl6_recsys2018/script/svd_py.py";
			String cachePath = "/media/mvolkovs/external4TB/Data/recsys2018/models/svd/";

			MLTimer timer = new MLTimer("main");
			timer.tic();

			// load data
			Data data = DataLoader.load(trainPath, testFile);
			timer.toc("data loaded");

			ParsedDataLoader loader = new ParsedDataLoader(data);
			loader.loadPlaylists();
			loader.loadSongs();
			loader.loadSongExtraInfo(extraInfoPath);
			ParsedData dataParsed = loader.dataParsed;
			timer.toc("data parsed");

			// generate split
			SplitterCF split = RecSysSplitter.getSplit(dataParsed);
			split = RecSysSplitter.getSplitMatching(dataParsed, split);
			RecSysSplitter.removeName(dataParsed, split);
			timer.toc("split done");

			// get all latents
			Latents latents = new Latents();

			// WMF
			ALSParams alsParams = new ALSParams();
			alsParams.alpha = 100;
			alsParams.rank = 200;
			alsParams.lambda = 0.001f;
			alsParams.maxIter = 10;
			ALS als = new ALS(alsParams);
			als.optimize(split.getRstrain().get(ParsedData.INTERACTION_KEY),
					null);
			latents.U = als.getU();
			latents.V = als.getV();

			// SVD
			SVDModel svdModel = new SVDModel(dataParsed, split, latents);
			svdModel.factorizeAlbums(pythonScriptPath, cachePath);
			svdModel.factorizeArtists(pythonScriptPath, cachePath);
			svdModel.factorizeNames(pythonScriptPath, cachePath);
			timer.toc("latents computed");

			// create training data
			

		} catch (Exception e) {
			e.printStackTrace();
		}

	}
}
