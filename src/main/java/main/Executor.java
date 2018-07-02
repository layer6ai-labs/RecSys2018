package main;

import common.MLTimer;
import common.SplitterCF;

public class Executor {

	public static void main(final String[] args) {
		try {
			String trainPath = "/media/mvolkovs/external4TB/Data/recsys2018/data/train";
			String testFile = "/media/mvolkovs/external4TB/Data/recsys2018/data/test/challenge_set.json";
			String extraInfoPath = "/media/mvolkovs/external4TB/Data/recsys2018/data/song_audio_features.txt";

			MLTimer timer = new MLTimer("main");

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

		} catch (Exception e) {
			e.printStackTrace();
		}

	}
}
