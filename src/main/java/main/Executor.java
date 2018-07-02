package main;

import common.MLTimer;

public class Executor {

	public static void main(final String[] args) {
		try {
			String trainPath = "/media/mvolkovs/external4TB/Data/recsys2018/data/train";
			String testFile = "/media/mvolkovs/external4TB/Data/recsys2018/data/test/challenge_set.json";
			String extraInfoPath = "/media/mvolkovs/external4TB/Data/recsys2018/data/song_audio_features.txt";

			MLTimer timer = new MLTimer("main");

			Data data = DataLoader.load(trainPath, testFile);
			timer.toc("data loaded");

			ParsedDataLoader loader = new ParsedDataLoader(data);
			loader.loadSongExtraInfo(extraInfoPath);
			timer.toc("data parsed");
			
			

		} catch (Exception e) {
			e.printStackTrace();
		}

	}
}
