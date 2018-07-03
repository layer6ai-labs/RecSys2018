package main;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import common.ALS;
import common.ALS.ALSParams;
import common.MLTimer;
import common.SplitterCF;
import main.XGBModel.XGBModelParams;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;

public class Executor {

	private static void downloadCreativeData(Data dataLoaded, String outFile)
			throws IOException {
		// Please provide your own key here in the format shown below
		final String AUTH_TOKEN = "Bearer BQAuDtl8KFXIsv02Uhm7AtYxNg8qMu72mhXgf8mQK61YDZ0jUb8RvGFpeo2PijBYM8PJZngBjUAWInrVhkcC0elkWvrFx3vsUJo3rU_8HjftS6jcH7yGQCAjOWTjM7_DnBGa2gqYf0Xgiq00_JqQ-Izj7UD98Nk";

		try (BufferedWriter bw = new BufferedWriter(new FileWriter(outFile))) {

			int nSongs = dataLoaded.songs.length;
			int batchSize = Math.floorDiv(nSongs, 100);
			OkHttpClient client = new OkHttpClient();

			for (int batch = 0; batch < batchSize; batch++) {

				// uncomment and provide batch number from where to begin in
				// case the operation was terminated due to auth expiration
				/*
				 * if(batch <33207) continue;
				 */

				System.out.println("Doing batch " + batch);

				int batchStart = batch * 100;
				int batchEnd = Math.min(batchStart + 100, nSongs);
				// Now form a batch of 100
				String url = "https://api.spotify.com/v1/audio-features?ids=";
				int firstTime = 1;
				for (int i = batchStart; i < batchEnd; i++) {
					if (firstTime == 1) {
						url = url + dataLoaded.songs[i].get_track_uri()
								.split(":")[2];
						firstTime = 0;
					} else {
						url = url + "%2C" + dataLoaded.songs[i].get_track_uri()
								.split(":")[2];
					}

				}

				Request request = new Request.Builder().url(url)
						.addHeader("Authorization", AUTH_TOKEN).build();
				Response responses = null;
				String append = "[";
				String last = "]";

				try {
					responses = client.newCall(request).execute();
				} catch (IOException e) {
					e.printStackTrace();
				}
				String jsonData = responses.body().string();
				jsonData = append + jsonData + last;
				org.json.JSONArray jsonarray = new org.json.JSONArray(jsonData);

				if (jsonarray.getJSONObject(0).has("error")) {
					System.out.println("timed out pausing for a while.");
					try {
						Thread.sleep(4000 + 1000);

						try {
							responses = client.newCall(request).execute();
						} catch (IOException e) {
							e.printStackTrace();
						}
						jsonData = responses.body().string();
						jsonData = append + jsonData + last;
						jsonarray = new org.json.JSONArray(jsonData);
					} catch (InterruptedException e) {
						e.printStackTrace();
					}

				}
				if (jsonarray.getJSONObject(0).has("error")) {
					System.out.println(jsonarray.getJSONObject(0));
					// Now our key has timed out . SO lets just exit
					bw.close();
					System.out.println(
							"Please refresh your key as you timed out on batch:  "
									+ batch);
					System.exit(1);
				}

				org.json.JSONArray jsonobject = (org.json.JSONArray) jsonarray
						.getJSONObject(0).get("audio_features");
				String writeString = jsonobject.toString();
				if (batch == 0) {
					writeString = writeString.replace("]", ",");
				} else if (batch == batchSize - 1) {
					writeString = writeString.replace("[", "");
				} else {
					writeString = writeString.replace("[", "");
					writeString = writeString.replace("]", ",");
				}
				bw.write(writeString);

			}
			bw.close();

		} catch (IOException e) {

			e.printStackTrace();

		}
		System.out.println("Extraction complete.");
	}

	public static void main(final String[] args) {
		try {
			String trainPath = "/media/mvolkovs/external4TB/Data/recsys2018/data/train";
			String testFile = "/media/mvolkovs/external4TB/Data/recsys2018/data/test/challenge_set.json";
			String creativeTrackFile = "/media/mvolkovs/external4TB/Data/recsys2018/data/song_audio_features.txt";
			String pythonScriptPath = "/home/mvolkovs/projects/vl6_recsys2018/script/svd_py.py";
			String cachePath = "/media/mvolkovs/external4TB/Data/recsys2018/models/svd/";

			MLTimer timer = new MLTimer("main");
			timer.tic();

			XGBModelParams xgbParams = new XGBModelParams();
			xgbParams.doCreative = false;
			xgbParams.xgbModel = cachePath + "xgb.model";

			// load data
			Data data = DataLoader.load(trainPath, testFile);
			timer.toc("data loaded");

			// download creative track features if not there
			if (xgbParams.doCreative == true
					&& new File(creativeTrackFile).exists() == false) {
				downloadCreativeData(data, creativeTrackFile);
			}

			ParsedDataLoader loader = new ParsedDataLoader(data);
			loader.loadPlaylists();
			loader.loadSongs();
			if (xgbParams.doCreative == true) {
				loader.loadSongExtraInfo(creativeTrackFile);
			}
			ParsedData dataParsed = loader.dataParsed;
			timer.toc("data parsed");

			// generate split
			SplitterCF split = RecSysSplitter.getSplitMatching(dataParsed);
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
			latents.Ucnn = als.getU();
			latents.V = als.getV();
			latents.Vcnn = als.getV();

			// SVD on album, artist and name
			SVDModel svdModel = new SVDModel(dataParsed, split, latents);
			svdModel.factorizeAlbums(pythonScriptPath, cachePath);
			svdModel.factorizeArtists(pythonScriptPath, cachePath);
			svdModel.factorizeNames(pythonScriptPath, cachePath);
			timer.toc("latents computed");

			// train second stage model
			// Latents latents = new Latents(dataParsed);
			XGBModel model = new XGBModel(dataParsed, xgbParams, latents,
					split);
			model.extractFeatures2Stage(cachePath);
			model.trainModel(cachePath);
			model.inference2Stage();

		} catch (Exception e) {
			e.printStackTrace();
		}

	}
}
